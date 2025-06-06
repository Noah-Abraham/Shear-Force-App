import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Shear Force & Axial Bolt Load Calculator")

class Bolt:
    def __init__(self, x, y, ks, ka):
        self.x = x  # X-coordinate
        self.y = y  # Y-coordinate
        self.ks = ks  # Shear stiffness
        self.ka = ka  # Axial stiffness

    def position(self):
        return np.array([self.x, self.y])
    
    def distance_from_centroid(self, XMC, YMC, XC, YC):
        self.adx = self.x - XMC
        self.ady = self.y - YMC
        self.sdx = self.x - XC
        self.sdy = self.y - YC

    def prime_distance_from_centroid(self, theta):
        theta_rad = np.radians(theta)
        r = np.hypot(self.adx, self.ady)
        phi = np.arctan2(self.ady, self.adx)
        self.addx = r * np.cos(theta_rad - phi)
        self.addy = r * np.sin(theta_rad - phi)

    def tensile_bolt_loads(self, POMX, POMY, IPX, IPY, PZ, num_bolts):
        # Direct tension from axial force
        VZ = PZ / num_bolts if num_bolts else 0.0
        # Moment-induced tension (principal axes)
        self.tblx = -POMX * self.addx / IPX if IPX != 0 else 0.0
        self.tbly = POMY * self.addy / IPY if IPY != 0 else 0.0
        # Total tension is algebraic sum (not vector sum)
        self.ttbl = VZ + self.tblx + self.tbly

    def secondary_shear(self, PX, PY, LX, LY, XC, YC, IT, num_bolts):
        # Torsional moment about shear centroid
        T = PY * (LX - XC) - PX * (LY - YC)
        Fx = T * (self.x - XC) / IT if IT != 0 else 0.0
        Fy = T * (self.y - YC) / IT if IT != 0 else 0.0
        Vx = Fx + PX / num_bolts
        Vy = Fy + PY / num_bolts
        self.bslx = Vx
        self.bsly = Vy
        self.tbsl = np.hypot(Vx, Vy)

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
num_bolts = int(st.number_input("Number of Bolts", min_value=1, step=1))
bolts = []

for i in range(num_bolts):
    st.markdown(f"**Bolt {i + 1}**")
    x = st.number_input(f"X Position", key=f"x{i}", format="%.6f")
    y = st.number_input(f"Y Position", key=f"y{i}", format="%.6f")
    ks = st.number_input(f"Shear Stiffness (KS)", key=f"ks{i}", min_value=0.0, format="%.6f", value=1.00)
    ka = st.number_input(f"Axial Stiffness (KA)", key=f"ka{i}", min_value=0.0, format="%.6f", value=1.00)
    bolts.append(Bolt(x, y, ks, ka))

# --- CALCULATION SECTION ---
def compute_centroids(bolts):
    TK = sum(b.ks for b in bolts)
    KAT = sum(b.ka for b in bolts)
    x1 = sum(b.x * b.ks for b in bolts)
    y1 = sum(b.y * b.ks for b in bolts)
    xm1 = sum(b.x * b.ka for b in bolts)
    ym1 = sum(b.y * b.ka for b in bolts)
    XC = x1 / TK if TK else 0.0
    YC = y1 / TK if TK else 0.0
    XMC = xm1 / KAT if KAT else 0.0
    YMC = ym1 / KAT if KAT else 0.0
    return XC, YC, XMC, YMC, TK, KAT

def compute_reference_inertias(bolts):
    IX = sum(b.ka * b.ady**2 for b in bolts)
    IY = sum(b.ka * b.adx**2 for b in bolts)
    IXY = sum(b.ka * b.adx * b.ady for b in bolts)
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
        xp = b.adx * np.cos(theta_rad) + b.ady * np.sin(theta_rad)
        yp = -b.adx * np.sin(theta_rad) + b.ady * np.cos(theta_rad)
        IPX += b.ka * xp**2
        IPY += b.ka * yp**2
    return IPX, IPY

def compute_shear_forces(bolts, PX, PY, MZ, XC, YC, TK, PZ):
    KAT = sum(b.ka for b in bolts)
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
        vz = PZ * b.ka / KAT if KAT else 0.0
        forces.append((b.x, b.y, vx, vy, vz))
    return forces

def overturning_moments(PX, PY, PZ, LX, LY, LZ, XMC, YMC):
    OMX = PX * LZ + PZ * (XMC - LX)
    OMY = PY * LZ + PZ * (YMC - LY)
    return OMX, OMY

def resolved_moments(OMX, OMY, theta):
    theta_rad = np.radians(theta)
    POMX = OMX * np.cos(theta_rad) + OMY * np.sin(theta_rad)
    POMY = -OMX * np.sin(theta_rad) + OMY * np.cos(theta_rad)
    return POMX, POMY

# --- DISPLAY RESULTS ---
if bolts:
    XC, YC, XMC, YMC, TK, KAT = compute_centroids(bolts)
    for b in bolts:
        b.distance_from_centroid(XMC, YMC)
    shear_forces = compute_shear_forces(bolts, PX, PY, MZ, XC, YC, TK, PZ)
    IX, IY, IXY = compute_reference_inertias(bolts)
    theta = compute_principal_axes(IX, IY, IXY)
    for b in bolts:
        b.prime_distance_from_centroid(theta)
    IT = np.sum(np.hypot(b.sdx, b.sdy)**2 for b in bolts)
    st.write(f"PX={PX}, PY={PY}, PZ={PZ}")
    st.write(f"LX={LX}, LY={LY}, LZ={LZ}")
    st.write(f"XMC={XMC}, YMC={YMC}")
    OMX, OMY = overturning_moments(PX, PY, PZ, LX, LY, LZ, XMC, YMC)
    POMX, POMY = resolved_moments(OMX, OMY, theta)
    IPX, IPY = compute_principal_moments(bolts, XMC, YMC, theta)

# Debug print for key values
    st.write(f"OMX: {OMX:.3f}, OMY: {OMY:.3f}")
    st.write(f"POMX: {POMX:.3f}, POMY: {POMY:.3f}")
    st.write(f"IPX: {IPX:.3f}, IPY: {IPY:.3f}")
    st.write(f"IT: {IT:.3f}")

    T = PY * (LX - XMC) - PX * (LY - YMC)
    st.write(f"T (Torsional moment about centroid): {T:.3f}")

    for i, b in enumerate(bolts):
        b.tensile_bolt_loads(POMX, POMY, IPX, IPY, PZ, num_bolts)
        b.secondary_shear(PX, PY, LX, LY, XC, YC, IT, num_bolts)

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
    layout_span = max(max(b.x for b in bolts) - min(b.x for b in bolts), max(b.y for b in bolts) - min(b.y for b in bolts), 1e-6)
    normalized_arrow_scale = 0.25 * layout_span

    force_mags = [np.hypot(vx, vy) for _, _, vx, vy, _ in shear_forces]
    max_force = max(force_mags) if force_mags else 1
    bolt_span = max(max(b.x for b in bolts) - min(b.x for b in bolts), max(b.y for b in bolts) - min(b.y for b in bolts), 1e-6)
    normalized_arrow_scale = (max_force / bolt_span) if max_force > 0 else 1
    vector_display_scale = 1 / (3 * normalized_arrow_scale)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    vector_extent = []
    from collections import defaultdict

    bolt_positions = defaultdict(list)
    for i, (x, y, vx, vy, vz) in enumerate(shear_forces):
        if view_option == "XZ View":
            key = (round(x, 3), 0)
        elif view_option == "YZ View":
            key = (round(y, 3), 0)
        else:
            key = (round(x, 3), round(y, 3))
        bolt_positions[key].append(i + 1)

    for i, (x, y, vx, vy, vz) in enumerate(shear_forces):
        if view_option == "XY View":
            vector_extent.append((x + vx / normalized_arrow_scale, y + vy / normalized_arrow_scale))
        elif view_option == "XZ View":
            vector_extent.append((x, vz / normalized_arrow_scale))
        elif view_option == "YZ View":
            vector_extent.append((y, vz / normalized_arrow_scale))

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

    x_margin = 0.2 * (max(all_x) - min(all_x) if max(all_x) != min(all_x) else 1)
    y_margin = 0.2 * (max(all_y) - min(all_y) if max(all_y) != min(all_y) else 1)

    ax.set_xlim(min(all_x) - 1.25 * x_margin, max(all_x) + 1.25 * x_margin)
    ax.set_ylim(min(all_y) - 1.25 * y_margin, max(all_y) + 1.25 * y_margin)
    ax.set_title(f"Bolt Force Vectors ({view_option})")
    ax.set_xlabel(view_option[0] + " Position")
    ax.set_ylabel(view_option[1] + " Position")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    for i, b in enumerate(bolts):
        if view_option == "XY View":
            # Plot shear vector (b.bslx, b.bsly) and label with tension and shear
            ax.plot(b.x, b.y, 'ro')
            ax.quiver(
                b.x, b.y,
                b.bslx * vector_display_scale, b.bsly * vector_display_scale,
                angles='xy', scale_units='xy', scale=1, color='blue'
            )
            label_text = f"{i+1}\nT: {b.ttbl:.2f} kN\nS: {b.tbsl:.2f} kN"
            offset_x = -0.25
            offset_y = -0.25
            ax.text(b.x + offset_x, b.y + offset_y, label_text, fontsize=7, color='black', ha='right', va='top')

        elif view_option == "XZ View":
            # Plot tension as vertical vector
            ax.plot(b.x, 0, 'ro')
            ax.quiver(
                b.x, 0,
                0, b.ttbl * vector_display_scale,
                angles='xy', scale_units='xy', scale=1, color='green'
            )
            label_text = f"{i+1}\nT: {b.ttbl:.2f} kN\nS: {b.tbsl:.2f} kN"
            offset_x = -0.25
            offset_z = -0.25
            ax.text(b.x + offset_x, 0 + offset_z, label_text, fontsize=7, color='black', ha='right', va='top')

        elif view_option == "YZ View":
            # Plot tension as vertical vector
            ax.plot(b.y, 0, 'ro')
            ax.quiver(
                b.y, 0,
                0, b.ttbl * vector_display_scale,
                angles='xy', scale_units='xy', scale=1, color='purple'
            )
            label_text = f"{i+1}\nT: {b.ttbl:.2f} kN\nS: {b.tbsl:.2f} kN"
            offset_y = -0.25
            offset_z = -0.25
            ax.text(b.y + offset_y, 0 + offset_z, label_text, fontsize=7, color='black', ha='right', va='top')

    if view_option == "XY View": 
        ax.plot(LX, LY, 'kx', label='Load Point')
        ax.plot(XC, YC, 'bs', label='Shear Centroid')
        ax.plot(XMC, YMC, 'gs', label='Axial Centroid')
    elif view_option == "YZ View":
        ax.plot(LY, LZ, 'kx', label='Load Point')
        ax.plot(YC, 0, 'bs', label='Shear Centroid')
        ax.plot(YMC, 0, 'gs', label='Axial Centroid')
    elif view_option == "XZ View":
        ax.plot(LX, LZ, 'kx', label='Load Point')
        ax.plot(XC, 0, 'bs', label='Shear Centroid')
        ax.plot(XMC, 0, 'gs', label='Axial Centroid')

    ax.legend()
    ax.set_aspect('equal', 'box')
    st.pyplot(fig)

    import pandas as pd
    st.subheader("Force Summary Table")

    force_df = pd.DataFrame({
        "Bolt ID": [f"{i+1}" for i in range(len(bolts))],
        "X": [b.x for b in bolts],
        "Y": [b.y for b in bolts],
        "Total Tensile Load (kN)": [round(b.ttbl, 3) for b in bolts],
        "Total Shear Load (kN)": [round(b.tbsl, 3) for b in bolts],
    })
    force_df.index = [f"Bolt {i+1}" for i in range(len(bolts))]
    st.dataframe(force_df.style.format({col: "{:.3f}" for col in force_df.select_dtypes(include=["float", "int"]).columns}))

    debug_df = pd.DataFrame({
        "Bolt ID": [f"{i+1}" for i in range(len(bolts))],
        "X": [b.x for b in bolts],
        "Y": [b.y for b in bolts],
        "sdx": [b.sdx for b in bolts],
        "sdy": [b.sdy for b in bolts],
        "adx": [b.adx for b in bolts],
        "ady": [b.ady for b in bolts],
        "addx": [getattr(b, 'addx', 0.0) for b in bolts],
        "addy": [getattr(b, 'addy', 0.0) for b in bolts],
        "tblx (Mx')": [getattr(b, 'tblx', 0.0) for b in bolts],
        "tbly (My')": [getattr(b, 'tbly', 0.0) for b in bolts],
        "ttbl (Total Tension)": [getattr(b, 'ttbl', 0.0) for b in bolts],
        "bslx (Sec. Shear X)": [getattr(b, 'bslx', 0.0) for b in bolts],
        "bsly (Sec. Shear Y)": [getattr(b, 'bsly', 0.0) for b in bolts],
        "tbsl (Total Shear)": [getattr(b, 'tbsl', 0.0) for b in bolts],
    })

    numeric_cols = ["X", "Y", "sdx", "sdy", "adx", "ady", "addx", "addy", "tblx (Mx')", "tbly (My')", "ttbl (Total Tension)", "bslx (Sec. Shear X)", "bsly (Sec. Shear Y)", "tbsl (Total Shear)"]
st.dataframe(debug_df.style.format({col: "{:.4f}" for col in numeric_cols}))
