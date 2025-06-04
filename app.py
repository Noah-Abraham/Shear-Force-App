import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Define the Bolt class ---
class Bolt:
    def __init__(self, x, y, ks=1.0, ka=1.0):
        self.x = x  # position x
        self.y = y  # position y
        self.ks = ks  # shear stiffness (not used here but placeholder)
        self.ka = ka  # axial stiffness (not used here but placeholder)
        
        # Variables to store results
        self.dx = 0.0
        self.dy = 0.0
        self.ddx = 0.0
        self.ddy = 0.0
        self.ttblx = 0.0
        self.ttbly = 0.0
        self.ttbl = 0.0
        self.bslx = 0.0
        self.bsly = 0.0
        self.tbsl = 0.0
    
    def distance_from_centroid(self, XMC, YMC):
        self.dx = self.x - XMC
        self.dy = self.y - YMC
    
    def prime_distance_from_centroid(self, theta):
        r = np.hypot(self.x, self.y)
        angle = np.arctan2(self.y, self.x)
        self.ddx = r * np.sin(np.radians(theta) - angle)
        self.ddy = r * np.cos(np.radians(theta) - angle)
    
    def tensile_bolt_loads(self, POMX, POMY, IPX, IPY, PZ, num_bolts):
        VZ = PZ / num_bolts if num_bolts else 0.0
        self.ttblx = (POMX * self.ddx / IPX) if IPX != 0 else 0.0
        self.ttbly = (POMY * self.ddy / IPY) if IPY != 0 else 0.0
        self.ttbl = self.ttblx + self.ttbly + VZ
    
    def secondary_shear(self, PX, PY, LX, LY, XMC, YMC, IT):
        T = PY * (LX - XMC) - PX * (LY - YMC)
        if IT == 0:
            self.bslx = 0.0
            self.bsly = 0.0
            self.tbsl = 0.0
        else:
            self.bslx = T * self.dx / IT
            self.bsly = T * self.dy / IT
            self.tbsl = np.hypot(self.bslx, self.bsly)

# --- Helper functions ---
def compute_centroid(bolts):
    XMC = np.mean([b.x for b in bolts])
    YMC = np.mean([b.y for b in bolts])
    return XMC, YMC

def compute_reference_inertias(bolts, XMC, YMC):
    IX = sum((b.y - YMC)**2 for b in bolts)
    IY = sum((b.x - XMC)**2 for b in bolts)
    IXY = sum((b.x - XMC)*(b.y - YMC) for b in bolts)
    return IX, IY, IXY

def compute_principal_axes(IX, IY, IXY):
    denom = IX - IY
    if denom == 0:
        theta = 45
    else:
        theta = 0.5 * np.degrees(np.arctan2(-2 * IXY, denom))
    return theta

def compute_principal_moments(bolts, XMC, YMC, theta):
    # Rotate coordinates by theta to principal axes
    IPX = 0.0
    IPY = 0.0
    for b in bolts:
        dx = b.x - XMC
        dy = b.y - YMC
        # Rotate
        dxp = dx * np.cos(np.radians(theta)) + dy * np.sin(np.radians(theta))
        dyp = -dx * np.sin(np.radians(theta)) + dy * np.cos(np.radians(theta))
        IPX += dxp**2
        IPY += dyp**2
    return IPX, IPY

def overturning_moments(PX, PY, PZ, LX, LY, LZ, XMC, YMC):
    # Moments about centroid due to loads at point (LX, LY, LZ)
    OMX = PY * (LZ - 0) - PZ * (LY - YMC)
    OMY = PZ * (LX - XMC) - PX * (LZ - 0)
    return OMX, OMY

def resolved_moments(OMX, OMY, theta):
    POMX = OMX * np.cos(np.radians(theta)) + OMY * np.sin(np.radians(theta))
    POMY = -OMX * np.sin(np.radians(theta)) + OMY * np.cos(np.radians(theta))
    return POMX, POMY

# --- Main program ---

# Example bolt positions (x,y)
bolt_coords = [(0.5, 1.0), (1.5, 1.0), (1.5, 0.0), (0.5, 0.0)]
bolts = [Bolt(x, y) for x, y in bolt_coords]

# External applied forces and load application point
PX, PY, PZ = 1000, 500, 300  # Forces in Newtons
LX, LY, LZ = 1.0, 0.5, 0.0  # Load application coordinates

num_bolts = len(bolts)

# Calculate centroid
XMC, YMC = compute_centroid(bolts)

# Calculate bolt distances from centroid
for b in bolts:
    b.distance_from_centroid(XMC, YMC)

# Calculate reference moments of inertia
IX, IY, IXY = compute_reference_inertias(bolts, XMC, YMC)

# Calculate principal axis angle
theta = compute_principal_axes(IX, IY, IXY)

# Calculate principal moments of inertia
IPX, IPY = compute_principal_moments(bolts, XMC, YMC, theta)

# Calculate moments about centroid
OMX, OMY = overturning_moments(PX, PY, PZ, LX, LY, LZ, XMC, YMC)

# Resolve moments in principal axes
POMX, POMY = resolved_moments(OMX, OMY, theta)

# Compute IT for secondary shear calculation
IT = sum(b.dx**2 + b.dy**2 for b in bolts)

# Calculate loads on each bolt
for b in bolts:
    b.prime_distance_from_centroid(theta)
    b.tensile_bolt_loads(POMX, POMY, IPX, IPY, PZ, num_bolts)
    b.secondary_shear(PX, PY, LX, LY, XMC, YMC, IT)

# --- Prepare DataFrame for display ---
force_data = {
    "X": [b.x for b in bolts],
    "Y": [b.y for b in bolts],
    "Total Tensile Load (ttbl)": [b.ttbl for b in bolts],
    "Total Shear Load (tbsl)": [b.tbsl for b in bolts],
}
force_df = pd.DataFrame(force_data, index=[f"Bolt {i+1}" for i in range(num_bolts)])

print(force_df.round(2))

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 6))

# Plot bolts
ax.plot([b.x for b in bolts], [b.y for b in bolts], 'ro', label='Bolts')

# Plot shear load vectors as arrows
vector_scale = 0.1  # Scale factor for arrows length
for b in bolts:
    if b.tbsl > 0:
        # Normalize shear vector direction
        vx = b.bslx / b.tbsl
        vy = b.bsly / b.tbsl
        # Scale vector by shear load magnitude for visibility
        ax.arrow(b.x, b.y, vx * b.tbsl * vector_scale, vy * b.tbsl * vector_scale,
                 head_width=0.05, head_length=0.1, fc='blue', ec='blue')

    # Annotate tensile load next to each bolt
    ax.text(b.x, b.y + 0.1, f"TTBL: {b.ttbl:.1f} N", color='red', fontsize=8, ha='center')

# Labels and grid
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Bolt Loads: Shear Vectors (Blue) & Tensile Load (Red text)')
ax.grid(True)
ax.legend()
plt.axis('equal')
plt.show()
