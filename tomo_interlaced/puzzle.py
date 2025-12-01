import numpy as np
from epics import PV

# ----------------------------------------------------
# PARAMETRI DI BASE
# ----------------------------------------------------
N_theta = 32    # numero totale di proiezioni
K = 4           # numero di loop interlacciati
r_outer = 1.0   # raggio del primo loop (per plot)
r_step = 0.15   # passo radiale tra i loop (per plot)

# ----------------------------------------------------
# EPICS PVs
# ----------------------------------------------------
pv_start_taxi = PV("2bmb:TomoScan:PSOStartTaxi")         # Posizione di inizio taxi [deg]
pv_end_taxi   = PV("2bmb:TomoScan:PSOEndTaxi")           # Posizione di fine taxi [deg]
pv_counts     = PV("2bmb:TomoScan:PSOCountsPerRotation") # Numero di impulsi per giro del PSO

# Lettura dai PV
start_taxi     = pv_start_taxi.get()           # es: -0.749939 deg
end_taxi       = pv_end_taxi.get()             # es: 0.735 deg
counts_per_rev = pv_counts.get()               # es: 11_840_200 impulsi/giro

# ----------------------------------------------------
# BIT-REVERSAL
# ----------------------------------------------------
def bit_reverse(x, bits):
    """Inverte i bit di x su 'bits' bit"""
    b = f'{x:0{bits}b}'
    return int(b[::-1], 2)

# ----------------------------------------------------
# TIMBIR
# ----------------------------------------------------
angles_timbir = []
loop_indices = []
bits = int(np.log2(K))

for n in range(N_theta):
    base = n * K
    loop = (base // N_theta) % K
    rev = bit_reverse(loop, bits)
    val = base + rev

    theta = val * 360.0 / N_theta       # angolo 0-360°
    theta = theta % 180.0               # 0-180° per tomografia

    angles_timbir.append(theta)
    loop_indices.append(loop)

angles_timbir = np.array(angles_timbir)
loop_indices = np.array(loop_indices)

# ----------------------------------------------------
# FUNZIONE TAXI CORRECTION
# ----------------------------------------------------
def taxi_correct(angles_deg, start_taxi, end_taxi, counts_per_rev):
    """
    Corregge gli angoli TIMBIR considerando l'inizio taxi
    e la fine taxi, e li converte in impulsi PSO.
    
    Parametri:
        angles_deg      : array degli angoli TIMBIR [deg]
        start_taxi      : angolo di inizio taxi [deg]
        end_taxi        : angolo di fine taxi [deg]
        counts_per_rev  : impulsi per giro del PSO

    Ritorna:
        pulses_corrected      : array impulsi corretti per il PSO
        pulses_end_corrected  : impulso corrispondente alla fine taxi
        theta_corrected       : angoli TIMBIR corretti per start taxi
        theta_end_corrected   : angolo finale corretto per end taxi
    """
    pulse_per_deg = counts_per_rev / 360.0

    theta_corrected = []
    pulses_corrected = []

    # correzione start taxi: shift angolare
    for theta in angles_deg:
        theta_corr = theta + abs(start_taxi)
        theta_corrected.append(theta_corr)
        pulses_corrected.append(theta_corr * pulse_per_deg)

    # correzione fine taxi
    theta_end_corrected = 180.0 + end_taxi
    pulses_end_corrected = theta_end_corrected * pulse_per_deg

    return np.array(pulses_corrected, dtype=int), int(pulses_end_corrected), theta_corrected, theta_end_corrected

# ----------------------------------------------------
# Applico la correzione taxi
# ----------------------------------------------------
pulses_corrected, pulses_end_corrected, theta_corrected, theta_end_corrected = taxi_correct(
    angles_timbir, start_taxi, end_taxi, counts_per_rev
)

# ----------------------------------------------------
# FUNZIONE CONVERSIONE GENERICA ANGOLI → IMPULSI (EPICS)
# ----------------------------------------------------
def angles_to_pulses_epics(angles_deg):
    """
    Converte angoli in gradi in impulsi PSO leggendo PSOCountsPerRotation dal PV.
    """
    counts_per_rev = float(pso_counts_pv.get())
    pulses = angles_deg * (counts_per_rev / 360.0)
    return np.round(pulses).astype(int)

# Esempio: conversione angoli TIMBIR senza taxi
pulses_epics = angles_to_pulses_epics(angles)

# ----------------------------------------------------
# PLOT POLARE TIMBIR
# ----------------------------------------------------
radii = r_outer - loop_indices * r_step  # raggio per ogni loop

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, polar=True)
ax.set_title(f"TIMBIR Interlaced Acquisition (N={N_theta} - K={K})\nOgni loop su un cerchio separato", va='bottom', fontsize=13)

# Plot angoli con correzione taxi (deg → rad)
ax.plot(theta_corrected*np.pi/180, radii, '-o', lw=1.2, ms=5, alpha=0.8, color='tab:blue')

# Annotazione numero loop
for i in range(N_theta):
    ax.text(theta_corrected[i]*np.pi/180, radii[i]+0.03, str(loop_indices[i]+1), ha='center', va='bottom', fontsize=8)

ax.set_rticks([])
plt.show()

# ----------------------------------------------------
# PROFILO MOTORE E CALCOLO TRIGGER
# ----------------------------------------------------
omega_target = 5.0         # velocità angolare target [deg/s]
T_acc = 0.5                # tempo accelerazione [s]
T_dec = 0.5                # tempo decelerazione [s]

# Calcolo tratto piatto (utile)
T_flat = 180.0 / omega_target - T_acc - T_dec

# Funzione inversa t_real(θ) per trigger reali
def t_real(theta_target, T_acc, T_flat, T_dec, omega_target):
    alpha = omega_target / T_acc
    t_out = np.zeros_like(theta_target, dtype=float)
    theta_acc = 0.5 * alpha * T_acc**2
    theta_flat = theta_acc + omega_target * T_flat
    theta_total = theta_flat + 0.5*alpha*T_dec**2

    for i, th in enumerate(theta_target):
        if th <= theta_acc:
            t_out[i] = np.sqrt(2*th/alpha)
        elif th <= theta_flat:
            t_out[i] = T_acc + (th - theta_acc)/omega_target
        elif th <= theta_total:
            a = 0.5*alpha
            b = -omega_target
            c = theta_flat - th
            dt = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
            t_out[i] = T_acc + T_flat + dt
        else:
            t_out[i] = T_acc + T_flat + T_dec
    return t_out

# Calcolo tempo trigger per angoli corretti
t_triggers = t_real(theta_corrected, T_acc, T_flat, T_dec, omega_target)

# Velocità istantanea
def omega_inst(t, T_acc, T_flat, T_dec, omega_target):
    alpha = omega_target / T_acc
    omega = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < T_acc:
            omega[i] = alpha * ti
        elif ti < T_acc + T_flat:
            omega[i] = omega_target
        elif ti <= T_acc + T_flat + T_dec:
            dt = ti - T_acc - T_flat
            omega[i] = omega_target - alpha*dt
        else:
            omega[i] = 0.0
    return omega

omega_values = omega_inst(t_triggers, T_acc, T_flat, T_dec, omega_target)

# Impulsi reali PSO
pulses_real = theta_corrected * (counts_per_rev / 360.0)

# ----------------------------------------------------
# STAMPA RISULTATI
# ----------------------------------------------------
for i in range(len(theta_corrected)):
    print(f"Angle {theta_corrected[i]:6.2f} deg -> Loop {loop_indices[i]} -> Pulse {pulses_real[i]:.0f} -> t_trigger {t_triggers[i]:.4f} s -> omega {omega_values[i]:.2f} deg/s")

print(f"Fine taxi: {theta_end_corrected:.3f} deg -> Pulse {pulses_end_corrected}")
