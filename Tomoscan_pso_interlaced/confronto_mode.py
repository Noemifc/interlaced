# ============================================================================
#                SCRIPT CONFRONTO TUTTE LE MODE
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from Tomoscan_pso_interlaced import InterlacedScan

# Parametri di scansione
num_angles = 32
K_interlace = 4
PSOCountsPerRotation = 20000

# Lista delle modalità disponibili
modes = ["timbir", "golden", "kturns", "multiturns", "corput"]

# Dizionario per salvare i dati di ciascuna modalità
results = {}

# Ciclo su tutte le modalità
for mode in modes:
    print(f"\n--- ESECUZIONE MODALITÀ: {mode} ---")
    
    scan = InterlacedScan(
        num_angles=num_angles,
        K_interlace=K_interlace,
        PSOCountsPerRotation=PSOCountsPerRotation
    )

    if mode == "timbir":
        scan.generate_interlaced_timbir()
    elif mode == "golden":
        angles_all = scan.generate_interlaced_goldenangle()
        scan.print_angles_table(angles_all)
        scan.print_cumulative_angles_table(angles_all)
        scan.plot_interlaced_circles(angles_all)
    elif mode == "kturns":
        angles_all = scan.generate_interlaced_kturns()
        scan.plot_equally_loops_polar_kturns()
        scan.print_cumulative_angles_table_kturns(angles_all)
        scan.print_angles_table_kturns(angles_all)
    elif mode == "multiturns":
        angles_all = scan.generate_interlaced_multiturns()
        scan.plot_equally_loops_polar_multiturns()
        scan.print_cumulative_angles_table_multiturns(angles_all)
        scan.print_angles_table_multiturns(angles_all)
    elif mode == "corput":
        angles_all = scan.generate_interlaced_corput()
        scan.plot_equally_loops_polar_corput()
        scan.print_cumulative_angles_table_corput(angles_all)
        scan.print_angles_table_corput(angles_all)
        scan.plot_live_corput()

    # Calcoli generali e confronto
    scan.compute_positions_PSO()
    scan.simulate_taxi_motion()
    scan.compute_real_motion()
    scan.convert_angles_to_counts()

    # Salvo dati per confronto
    results[mode] = {
        "theta": scan.theta_interlaced,
        "theta_unwrapped": scan.theta_interlaced_unwrapped,
        "PSO_Ideal": scan.PSOCountsIdeal,
        "PSO_Real": scan.PSOCountsTaxiCorrected,
        "PSO_Final": scan.PSOCountsFinal
    }

# ============================================================================
# Confronto finale: tutte le modalità insieme
# ============================================================================

fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

for mode, color in zip(modes, ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']):
    data = results[mode]
    ideal = data["PSO_Ideal"]
    real = data["PSO_Real"]
    final = data["PSO_Final"]
    
    axs[0].plot(ideal, real, 'o-', alpha=0.7, label=f"{mode} Real vs Ideal", color=color)
    axs[1].plot(ideal, final, 'o-', alpha=0.7, label=f"{mode} FPGA vs Ideal", color=color)
    axs[2].plot(real, final, 'o-', alpha=0.7, label=f"{mode} FPGA vs Real", color=color)

axs[0].set_title("Ideale vs Reale")
axs[1].set_title("Ideale vs FPGA")
axs[2].set_title("Reale vs FPGA")
for ax in axs:
    ax.grid(True)
    ax.legend()
plt.xlabel("Indice angolo")
plt.tight_layout()
plt.show()
