import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Parameters
# ------------------------
N_theta = 32  # total projections
K = 4  # number of interlaced loops
r_outer = 1.0  # radius for first loop
r_step = 0.15  # radial step between loops


# ------------------------
# Bit-reversal function
# ------------------------
def bit_reverse(x, bits):
    b = f'{x:0{bits}b}'
    return int(b[::-1], 2)


# ------------------------
# Compute acquisition angles in time order
# ------------------------
bits = int(np.log2(K))
angles = []
loop_indices = []

for n in range(N_theta):
    val = n * K + bit_reverse((n * K // N_theta) % K, bits)
    theta = val * 2 * np.pi / N_theta  # full 360° rotation
    angles.append(theta)

    # Determine which loop this acquisition belongs to
    loop = (n * K // N_theta) % K  # 0 to K-1
    loop_indices.append(loop)

angles = np.array(angles)
loop_indices = np.array(loop_indices)

# ------------------------
# Assign radius based on loop
# ------------------------
radii = r_outer - loop_indices * r_step  # all points in the same loop share radius

# ------------------------
# Plot acquisition sequence
# ------------------------
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, polar=True)
ax.set_title(f"TIMBIR Interlaced Acquisition (N={N_theta} - K={K})\nEach loop on its own circle", va='bottom',
             fontsize=13)

# Connect points in true acquisition order
ax.plot(angles, radii, '-o', lw=1.2, ms=5, alpha=0.8, color='tab:blue')

# Optional: annotate loop number
for i in range(N_theta):
    ax.text(angles[i], radii[i] + 0.03, str(loop_indices[i] + 1), ha='center', va='bottom', fontsize=8)

# Hide radial ticks
ax.set_rticks([])
plt.show()

#-------------------------------------------
#         ADD
#-------------------------------------------

# Convert angles to degrees
angles_deg = np.degrees(angles)

angles_deg_rounded = np.round(angles_deg, 4)

angles_deg_uniq = np.unique(angles_deg_rounded)

# Convert to comma-separated string
angles_str = ', '.join(f'{a:.2f}' for a in angles_deg)
angles_str_uniq = ', '.join(f'{a:.2f}' for a in angles_deg_uniq)
print("Acquisition angles (degrees) in time order:")
print(angles_str)
print("Acquisition angles uniq (degrees) in time order:")
print(angles_str_uniq)