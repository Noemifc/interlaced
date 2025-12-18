import numpy as np
import matplotlib.pyplot as plt
from Tomoscan_pso_interlaced import InterlacedScan

# Parametri di scansione
num_angles = 32
K_interlace = 4
PSOCountsPerRotation = 20000
modes = ["timbir", "golden", "kturns", "multiturns", "corput"]

results = {}

# Ciclo su tutte le modalità e calcolo dati
for mode in modes:
    scan = InterlacedScan(
        num_angles=num_angles,
        K_interlace=K_interlace,
        PSOCountsPerRotation=PSOCountsPerRotation
    )

    if mode == "timbir":
        scan.generate_interlaced_timbir()
    elif mode == "golden":
        angles_all = scan.generate_interlaced_goldenangle()
    elif mode == "kturns":
        angles_all = scan.generate_interlaced_kturns()
    elif mode == "multiturns":
        angles_all = scan.generate_interlaced_multiturns()
    elif mode == "corput":
        angles_all = scan.generate_interlaced_corput()

    # Calcoli comuni
    scan.compute_positions_PSO()
    scan.simulate_taxi_motion()
    scan.compute_real_motion()
    scan.convert_angles_to_counts()

    # Salvo dati
    results[mode] = {
        "theta": scan.theta_interlaced,
        "PSO_Final": scan.PSOCountsFinal
    }

# ============================================================================
# Plot miniplot: angolo vs impulsi per tutte le modalità
# ============================================================================

fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

for ax, mode, color in zip(axs, modes, colors):
    data = results[mode]
    ax.plot(data["theta"], data["PSO_Final"], 'o-', color=color, alpha=0.8)
    ax.set_title(mode.capitalize())
    ax.set_xlabel("Angolo [deg]")
    ax.grid(True)

axs[0].set_ylabel("Impulsi encoder")
plt.suptitle("Confronto modalità: Angolo vs Impulsi")
plt.tight_layout()
plt.show()
