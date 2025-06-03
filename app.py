import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.title("Shear Force & Axial Bolt Load Calculator")

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

def compute_shear_forces(bolts, PX, PY, MZ, XC, YC, TK, PZ, KAT):
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
        shear_mag = np.hypot(vx, vy)
        forces.append((b.x, b.y, vx, vy, vz, shear_mag))
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

    shear_forces = compute_shear_forces(bolts, PX, PY, MZ, XC, YC, TK, PZ, KAT)

    st.subheader("Bolt Load Summary")
    for i, (x, y, vx, vy, vz, shear_mag) in enumerate(shear_forces):
        st.write(f"Bolt {i+1}: Axial Load (VZ) = {vz:.3f} kN, Shear Load (√VX²+VY²) = {shear_mag:.3f} kN")

    force_df = pd.DataFrame(shear_forces, columns=["X", "Y", "VX", "VY", "VZ", "Shear Magnitude"])
    force_df.index = [f"Bolt {i+1}" for i in range(len(shear_forces))]
    st.subheader("Force Summary Table")
    st.dataframe(force_df.style.format("{:.3f}"))

    # Visual display
    st.subheader("Bolt Load Visualization")
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    for b, (_, _, vx, vy, vz, _) in zip(bolts, shear_forces):
        ax.plot(b.x, b.y, 'ko')
        ax.arrow(b.x, b.y, vx, vy, head_width=0.1, color='r', label='Shear')
        ax.arrow(b.x, b.y, 0, vz, head_width=0.1, color='b', label='Axial')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Bolt Shear and Axial Loads")
    st.pyplot(fig)
