import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------
# Parametri TIMBIR
#-----------------------------------------------------------------
N_theta = 32          # total projections
K = 4                 # number of interlaced loops
r_outer = 1.0         # radius for first loop (plot only)
r_step = 0.15         # radial step between loops

#-----------------------------------------------------------------
# Funzione di bit-reversal
#-----------------------------------------------------------------
def bit_reverse(x, bits):
    b = f'{x:0{bits}b}'
    return int(b[::-1], 2)

#-----------------------------------------------------------------
# Compute acquisition angles_timbir in time order
# Qui si generano gli angoli interlacciati TIMBIR
#-----------------------------------------------------------------
def generate_timbir_angles(N_theta, K):
    bits = int(np.log2(K))
    angles = []
    loops = []

    for n in range(N_theta):
        # Valore originale TIMBIR
        val = n * K + bit_reverse((n * K // N_theta) % K, bits)

        # Conversione in GRADI (non radianti)
        # full 360° rotation
        theta_deg = val * 360.0 / N_theta  
        angles.append(theta_deg)

        # Determine which loop this acquisition belongs to
        # Ogni proiezione appartiene a un loop 0..K-1
        loop = (n * K // N_theta) % K
        loops.append(loop)

    return np.array(angles), np.array(loops)

angles_timbir, loop_indices = generate_timbir_angles(N_theta, K)

print("Angoli interlacciati (°):")
print(angles_timbir)

#-----------------------------------------------------------------
# ogni loop ha un raggio diverso per visualizzazione
#-----------------------------------------------------------------
radii = r_outer - loop_indices * r_step

#-----------------------------------------------------------------
#  sequenza degli angoli nel piano polare
#-----------------------------------------------------------------
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, polar=True)
ax.set_title(
    f"TIMBIR Interlaced Acquisition (N={N_theta} - K={K})\nEach loop on its own circle",
    va='bottom',
    fontsize=13
)

# Connect points in true acquisition order
ax.plot(np.deg2rad(angles_timbir), radii, '-o', lw=1.2, ms=5, alpha=0.8)

# Optional: annotate loop number
for i in range(N_theta):
    ax.text(
        np.deg2rad(angles_timbir[i]),
        radii[i] + 0.03,
        str(loop_indices[i] + 1),
        ha='center',
        va='bottom',
        fontsize=8
    )

# Hide radial ticks
ax.set_rticks([])
plt.show()

#-----------------------------------------------------------------
# Simula Counts encoder e impulsi
# Qui il PSO converte gli angoli in impulsi
#-----------------------------------------------------------------
PSOCountsPerRotation = 20000  
# 1 impulso = 0.018° circa
# counts_per_rev = PV("2bmb:TomoScan:PSOCountsPerRotation") # Numero di impulsi per giro del PSO

encoder_multiply = PSOCountsPerRotation / 360.0  # fattore di conversione da angoli a impulsi

# angolo * fattore counts/degree
raw_delta_encoder_counts = angles_timbir * encoder_multiply

# Arrotondamento perché il PSO accetta solo valori interi
delta_encoder_counts = np.round(raw_delta_encoder_counts).astype(int)

# Rotation step reale (dopo arrotondamento): trasformo counts in angolo reale per vedere a quanti gradi reali corrispondono
rotation_step_real_deg = delta_encoder_counts / encoder_multiply

plt.figure(figsize=(12,5))

# --- PLOT ---

t = np.arange(len(angles_timbir))   # tempo / indice acquisizione

plt.figure(figsize=(14,6))

# --- Curve principali ---
plt.plot(t, rotation_step_real_deg, '.-', color='blue', label='Angolo reale (°)')
plt.plot(t, delta_encoder_counts, '.-', color='orange', label='Count PSO')

# --- Linee verticali che collegano i due valori ---
for i in range(len(t)):
    plt.plot([t[i], t[i]],
             [rotation_step_real_deg[i], delta_encoder_counts[i]],
             color='gray', alpha=0.4, linewidth=0.8)

plt.xlabel("Tempo / Indice acquisizione")
plt.ylabel("Valore")
plt.title("Angolo reale e Count PSO con linee verticali di corrispondenza")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
