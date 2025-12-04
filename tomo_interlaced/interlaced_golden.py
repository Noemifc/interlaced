import numpy as np
import matplotlib.pyplot as plt
from epics import PV   

# --- PVs ---
pv_N_theta = PV("2bmb:TomoScan:NTheta")
pv_K       = PV("2bmb:TomoScan:KLoops")

N_theta = int(pv_N_theta.get() or 32)  # default 32 se PV non risponde
K       = int(pv_K.get() or 4)         # default 4 se PV non risponde

# --- parametri ---
end_angle = 360.
golden_a = end_angle * (3 - np.sqrt(5)) / 2  # ≈ 111.246°
num_proj = N_theta  # numero di proiezioni per loop

# --- offset iniziali per interlacciamento  ---
theta_start = np.linspace(0, end_angle, K, endpoint=False)  # interlacciamento equispaziato

# --- generazione angoli golden interlacciati ---
golden_angles_tomo = np.mod(
    theta_start[:, None] + np.arange(num_proj) * golden_a,
    end_angle
).flatten()

# ordinamento e rimozione duplicati
golden_angles_interl = np.unique(np.sort(golden_angles_tomo))

# --- stampa risultati ---
print("N_theta:", N_theta, "K:", K)
print("Totale angoli generati:", len(golden_angles_tomo))
print("Totale angoli unici:", len(golden_angles_interl))
print("Primi 10 angoli:", np.round(golden_angles_interl[:10], 3))
print("\nTutti gli angoli (fino a 100):")
print(np.round(golden_angles_interl[:100], 3))

# --- Plot ---
plt.figure(figsize=(6, 6))
colors = plt.cm.tab10(np.linspace(0, 1, K))

for i, start in enumerate(theta_start):
    angles = np.mod(start + np.arange(num_proj) * golden_a, end_angle)
    plt.polar(np.deg2rad(angles), np.ones_like(angles), '.', alpha=0.6, color=colors[i])

plt.title(f"Interlaced golden-angle sampling (mod {end_angle}°)")
plt.legend([f"Loop {i+1}" for i in range(K)])
plt.show()

plt.show()
