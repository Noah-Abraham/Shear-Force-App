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
    x = st.number_input(f"X Position", key=f"x{i}")
    y = st.number_input(f"Y Position", key=f"y{i}")
    ks = st.number_input(f"Shear Stiffness (KS)", key=f"ks{i}", min_value=0.0)
    ka = st.number_input(f"Axial Stiffness (KA)", key=f"ka{i}", min_value=0.0)
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
        xp = dy * np.sin(theta_rad) - dx * np.cos(theta_rad)
        yp = dy * np.cos(theta_rad) + dx * np.sin(theta_rad)
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
        forces.append((b.x, b.y, vx, vy))
    return forces

# --- DISPLAY RESULTS ---
if bolts:
    XC, YC, XMC, YMC, TK, KAT = compute_centroids(bolts)
    IX, IY, IXY = compute_reference_inertias(bolts, XMC, YMC)
    theta = compute_principal_axes(IX, IY, IXY)
    IPX, IPY = compute_principal_moments(bolts, XMC, YMC, theta)

    st.subheader("Centroid Locations")
    st.write(f"Shear Centroid (XC, YC): ({XC:.2f}, {YC:.2f})")
    st.write(f"Axial Centroid (XMC, YMC): ({XMC:.2f}, {YMC:.2f})")

    st.subheader("Reference Axis Inertias")
    st.write(f"Moment of Inertia about X (IX): {IX:.2f}")
    st.write(f"Moment of Inertia about Y (IY): {IY:.2f}")
    st.write(f"Product of Inertia (IXY): {IXY:.2f}")

    st.subheader("Principal Axes")
    st.write(f"Rotation Angle to Principal Axis (θ): {theta:.2f} degrees")
    st.write(f"Principal Moment IPX: {IPX:.2f}")
    st.write(f"Principal Moment IPY: {IPY:.2f}")

    # --- FORCE VECTOR VISUALIZATION ---
    shear_forces = compute_shear_forces(bolts, PX, PY, MZ, XC, YC, TK)

    fig, ax = plt.subplots()
    ax.set_title("Bolt Shear Force Vectors")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True)

    for x, y, vx, vy in shear_forces:
        ax.quiver(x, y, vx, vy, angles='xy', scale_units='xy', scale=1, color='blue')
        ax.plot(x, y, 'ro')  # Bolt location

    ax.set_aspect('equal', 'box')
    st.pyplot(fig)
