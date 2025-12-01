import numpy as np
import matplotlib.pyplot as plt
from epics import PV

  #----------------------------------------------------
  # TIMBIR con EPICS 
  #----------------------------------------------------

N_theta = 32   # total projections
K = 4          # number of interlaced loops
r_outer = 1.0  # radius for first loop
r_step = 0.15  # radial step between loops

# PV che contiene il numero di impulsi per giro
pso_counts_pv = PV('PSOCountsPerRotation')

# ------------------------
# Bit-reversal function
# ------------------------
def bit_reverse(x, bits):
    b = f'{x:0{bits}b}'
    return int(b[::-1], 2)

# ------------------------
# Generate TIMBIR angles
# ------------------------
angles = []
loop_indices = []
bits = int(np.log2(K))

for n in range(N_theta):
    base = n * K
    loop = (base // N_theta) % K
    rev = bit_reverse(loop, bits)
    val = base + rev

    theta = val * 360.0 / N_theta
    theta = theta % 180.0  # 180° for tomography

    angles.append(theta)
    loop_indices.append(loop)

angles = np.array(angles)
loop_indices = np.array(loop_indices)


  #----------------------------------------------------
  # Prendi angles_deg e converti in impulsi reali  vedi formula su appunti
  #----------------------------------------------------
''' Converte gli angoli in impulsi PSO reali leggendo PSOCountsPerRotation dal PV EPICS '''

def angles_to_pulses_epics(angles_deg):
    """
    Convert angles (deg) to PSO pulses using EPICS counts per revolution.
    """
    counts_per_rev = float(pso_counts_pv.get())
    pulses = angles_deg * (counts_per_rev / 360.0)
    return np.round(pulses).astype(int)

pulses = angles_to_pulses_epics(angles)

# ------------------------
# Assign radius based on loop for plotting
# ------------------------
radii = r_outer - loop_indices * r_step

# ------------------------
# PLOT : visualizza e stampa in plot polare  e stampa angoli, plot e pulses
# ------------------------
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, polar=True)
ax.set_title(f"TIMBIR Interlaced Acquisition (N={N_theta} - K={K})\nEach loop on its own circle", va='bottom', fontsize=13)

# Connect points in true acquisition order
ax.plot(angles * np.pi/180, radii, '-o', lw=1.2, ms=5, alpha=0.8, color='tab:blue')  # convert deg -> rad

# Optional: annotate loop number
for i in range(N_theta):
    ax.text(angles[i]*np.pi/180, radii[i]+0.03, str(loop_indices[i]+1), ha='center', va='bottom', fontsize=8)

# Hide radial ticks
ax.set_rticks([])
plt.show()

# ------------------------
# Print results
# ------------------------
for i in range(N_theta):
    print(f"Angle {angles[i]:6.2f} deg -> Loop {loop_indices[i]} -> Pulse {pulses[i]}")
    

  #----------------------------------------------------
  # Calcola i ritardi tra un impulso e il successivo (memPulseSeq)
  #----------------------------------------------------







