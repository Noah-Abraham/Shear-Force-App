import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Shear Force & Bolt Load Calculator")

class Bolt:
    def __init__(self, x, y, ks, ka):
        self.x = x  # X-coordinate
        self.y = y  # Y-coordinate
        self.ks = ks  # Shear stiffness
        self.ka = ka  # Axial stiffness

    def position(self):
        return np.array([self.x, self.y])

# --- INPUT SECTION ---

st.subheader("Load Cases")
PX = st.number_input("External Force in X (PX) [kN]", value=0.0)
PY = st.number_input("External Force in Y (PY) [kN]", value=0.0)
PZ = st.number_input("Axial Force (PZ) [kN]", value=0.0)
LX = st.number_input("X Coordinate of Load Application Point", value=0.0)
LY = st.number_input("Y Coordinate of Load Application Point", value=0.0)
LZ = st.number_input("Z Coordinate (for Torsion)", value=0.0)
MX = st.number_input("External Moment about X-axis (MX) [kNm]", value=0.0)
MY = st.number_input("External Moment about Y-axis (MY) [kNm]", value=0.0)
MZ = st.number_input("External Moment about Z-axis (MZ) [kNm]", value=0.0)

st.subheader("Bolt Configuration")
num_bolts = st.number_input("Number of Bolts", min_value=1, step=1)
bolts = []

for i in range(num_bolts):
    st.markdown(f"**Bolt {i + 1}**")
    x = st.number_input(f"X Position", key=f"x{i}", format="%.6f")
    y = st.number_input(f"Y Position", key=f"y{i}", format="%.6f")
    ks = st.number_input(f"Shear Stiffness (KS)", key=f"ks{i}", min_value=0.0, format="%.6f")
    ka = st.number_input(f"Axial Stiffness (KA)", key=f"ka{i}", min_value=0.0, format="%.6f")
    bolts.append(Bolt(x, y, ks, ka))

# --- CALCULATION SECTION ---
def compute_centroids(bolts):
    total_shear_stiffness = sum(b.ks for b in bolts)
    total_axial_stiffness = sum(b.ka for b in bolts)
    x1 = sum(b.x * b.ks for b in bolts)
    y1 = sum(b.y * b.ks for b in bolts)
    xm1 = sum(b.x * b.ka for b in bolts)
    ym1 = sum(b.y * b.ka for b in bolts)
    XC = x1 / total_shear_stiffness if total_shear_stiffness else 0.0
    YC = y1 / total_shear_stiffness if total_shear_stiffness else 0.0
    XMC = xm1 / total_axial_stiffness if total_axial_stiffness else 0.0
    YMC = ym1 / total_axial_stiffness if total_axial_stiffness else 0.0
    return XC, YC, XMC, YMC, total_shear_stiffness, total_axial_stiffness

def compute_reference_inertias(bolts, XMC, YMC):
    IX = sum(b.ka * (b.y - YMC)**2 for b in bolts)
    IY = sum(b.ka * (b.x - XMC)**2 for b in bolts)
    IXY = sum(b.ka * (b.x - XMC) * (b.y - YMC) for b in bolts)
    return IX, IY, IXY

def compute_principal_axes(IX, IY, IXY):
    if abs(IXY) < 1e-4:
        theta = 0.0
    elif abs(IX - IY) < 1e-6:
        theta = 45.0
    else:
        theta = 0.5 * np.degrees(np.arctan2(2 * IXY, (IY - IX)))
    return theta

def compute_principal_moments(bolts, XMC, YMC, theta):
    theta_rad = np.radians(theta)
    IPX = 0.0
    IPY = 0.0
    for b in bolts:
        dx = b.x - XMC
        dy = b.y - YMC
        xp = dx * np.cos(theta_rad) + dy * np.sin(theta_rad)
        yp = -dx * np.sin(theta_rad) + dy * np.cos(theta_rad)
        IPX += b.ka * yp**2
        IPY += b.ka * xp**2
    return IPX, IPY

def compute_shear_forces(bolts, PX, PY, MZ, XC, YC, TK):
    RS = sum((np.hypot(b.x - XC, b.y - YC))**2 for b in bolts)
    forces = []
    for b in bolts:
        rx = b.x - XC
        ry = YC - b.y
        fx = PX * b.ks / TK if TK else 0.0
        fy = PY * b.ks / TK if TK else 0.0
        ftx = MZ * ry * b.ks / RS if RS else 0.0
        fty = MZ * rx * b.ks / RS if RS else 0.0
        vx = fx + ftx
        vy = fy + fty
        vz = PZ * b.ka / sum(b.ka for b in bolts) if KAT else 0.0
        forces.append((b.x, b.y, vx, vy, vz))
    return forces

# --- DISPLAY RESULTS ---
if bolts:
    XC, YC, XMC, YMC, TK, KAT = compute_centroids(bolts)
    IX, IY, IXY = compute_reference_inertias(bolts, XMC, YMC)
    theta = compute_principal_axes(IX, IY, IXY)
    IPX, IPY = compute_principal_moments(bolts, XMC, YMC, theta)

    st.subheader("Centroid Locations")
    st.write(f"Shear Centroid (XC, YC): ({XC:.6f}, {YC:.6f})")
    st.write(f"Axial Centroid (XMC, YMC): ({XMC:.6f}, {YMC:.6f})")

    st.subheader("Reference Axis Inertias")
    st.write(f"Moment of Inertia about X (IX): {IX:.6f}")
    st.write(f"Moment of Inertia about Y (IY): {IY:.6f}")
    st.write(f"Product of Inertia (IXY): {IXY:.6f}")

    st.subheader("Principal Axes")
    st.write(f"Rotation Angle to Principal Axis (θ): {theta:.6f} degrees")
    st.write(f"Principal Moment IPX: {IPX:.6f}")
    st.write(f"Principal Moment IPY: {IPY:.6f}")

    view_option = st.radio("Select Force View", ["XY View", "XZ View", "YZ View"])
    # Fixed normalized arrow scale based on layout, without slider
    layout_span = max(max(b.x for b in bolts) - min(b.x for b in bolts), max(b.y for b in bolts) - min(b.y for b in bolts), 1e-6)
    normalized_arrow_scale = 0.25 * layout_span  # Set arrow size relative to layout size

    # Normalize force vectors based on max force and layout span
    force_mags = [np.hypot(vx, vy) for _, _, vx, vy, _ in compute_shear_forces(bolts, PX, PY, MZ, XC, YC, TK)]
    max_force = max(force_mags) if force_mags else 1
    bolt_span = max(max(b.x for b in bolts) - min(b.x for b in bolts), max(b.y for b in bolts) - min(b.y for b in bolts), 1e-6)
    normalized_arrow_scale = (max_force / bolt_span) if max_force > 0 else 1
    vector_display_scale = 1 / (3 * normalized_arrow_scale)

    shear_forces = compute_shear_forces(bolts, PX, PY, MZ, XC, YC, TK)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    # Calculate bounding box based on bolt layout and scaled vector length
    vector_extent = []
    for i, (x, y, vx, vy, vz) in enumerate(shear_forces):
        if view_option == "XY View":
            vector_extent.append((x + vx / normalized_arrow_scale, y + vy / normalized_arrow_scale))
        elif view_option == "XZ View":
            vector_extent.append((x + vx / normalized_arrow_scale, vz / normalized_arrow_scale))
        elif view_option == "YZ View":
            vector_extent.append((y + vy / normalized_arrow_scale, vz / normalized_arrow_scale))

    all_x = [b.x for b in bolts]
    all_y = [b.y for b in bolts]

    if view_option == "XY View":
        all_x += [pt[0] for pt in vector_extent]
        all_y += [pt[1] for pt in vector_extent]
    elif view_option == "XZ View":
        all_x += [pt[0] for pt in vector_extent]
        all_y += [pt[1] for pt in vector_extent]
    elif view_option == "YZ View":
        all_x += [pt[0] for pt in vector_extent]
        all_y += [pt[1] for pt in vector_extent]

    x_margin = 0.1 * (max(all_x) - min(all_x) if max(all_x) != min(all_x) else 1)
    y_margin = 0.1 * (max(all_y) - min(all_y) if max(all_y) != min(all_y) else 1)

    ax.set_xlim(min(all_x) - 1.25 * x_margin, max(all_x) + 1.25 * x_margin)
    ax.set_ylim(min(all_y) - 1.25 * y_margin, max(all_y) + 1.25 * y_margin)
    ax.set_title(f"Bolt Force Vectors ({view_option})")
    ax.set_xlabel(view_option[0] + " Position")
    ax.set_ylabel(view_option[1] + " Position")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    for i, (x, y, vx, vy, vz) in enumerate(shear_forces):
        if view_option == "XY View":
            ax.quiver(x, y, vx * vector_display_scale, vy * vector_display_scale, angles='xy', scale_units='xy', scale=1, color='blue')
            ax.plot(x, y, 'ro')
            ax.text(x - 0.1, y - 0.1, f"Bolt" {i+1}
({vx:} kN, {vy:} kN), fontsize=7, color='black', ha='right', va='top')
            
        elif view_option == "XZ View":
            ax.quiver(x, 0, vx * vector_display_scale, vz * vector_display_scale, angles='xy', scale_units='xy', scale=1, color='green')
            ax.plot(x, 0, 'ro')
            ax.text(x - 0.1, -0.1, f"Bolt" {i+1}
({vx:} kN, {vz:} kN), fontsize=7, color='black', ha='right', va='top')
            
        elif view_option == "YZ View":
            ax.quiver(y, 0, vy * vector_display_scale, vz * vector_display_scale, angles='xy', scale_units='xy', scale=1, color='purple')
            ax.plot(y, 0, 'ro')
            ax.text(y - 0.1, -0.1, f"Bolt" {i+1}
({vy:} kN, {vz:} kN), fontsize=7, color='black', ha='right', va='top')
            

        # Plot centroid positions and label arrows
    if view_option == "XY View":
        ax.plot(XC, YC, 'bs', label='Shear Centroid')
        ax.plot(XMC, YMC, 'gs', label='Axial Centroid')
    elif view_option == "XZ View":
        ax.plot(XC, 0, 'bs', label='Shear Centroid')
        ax.plot(XMC, 0, 'gs', label='Axial Centroid')
    elif view_option == "YZ View":
        ax.plot(YC, 0, 'bs', label='Shear Centroid')
        ax.plot(YMC, 0, 'gs', label='Axial Centroid')

    ax.legend()
    ax.set_aspect('equal', 'box')
    st.pyplot(fig)

    # Optional: show a force summary table
    import pandas as pd
    force_df = pd.DataFrame(shear_forces, columns=["X", "Y", "VX", "VY", "VZ"])
    force_df.index = [f"Bolt {i+1}" for i in range(len(shear_forces))]
    st.subheader("Force Summary Table")
    st.dataframe(force_df.style.format("{:.3f}"))
