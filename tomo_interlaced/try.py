import numpy as np
from epics import PV

# ------------------------------
# Parametri TIMBIR e PVs
# ------------------------------
pv_N_theta = PV("2bmb:TomoScan:NTheta")          
pv_K       = PV("2bmb:TomoScan:KLoops")         
pv_start_taxi = PV("2bmb:TomoScan:PSOStartTaxi")         
pv_end_taxi   = PV("2bmb:TomoScan:PSOEndTaxi")           
pv_counts     = PV("2bmb:TomoScan:PSOCountsPerRotation") 

N_theta = int(pv_N_theta.get() or 32)
K       = int(pv_K.get() or 4)
start_taxi = float(pv_start_taxi.get() or -0.75)
end_taxi   = float(pv_end_taxi.get() or 0.75)
counts_per_rev = float(pv_counts.get() or 11_840_200)

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
# ------------------------------
# Correzione taxi
# ------------------------------
def taxi_correct(angles_deg, start_taxi, end_taxi, counts_per_rev):
   
    pulse_per_deg = counts_per_rev / 360.0  # conversione in impulsi i

    theta_corrected = []
    pulses_corrected = []

    # correzione start taxi-> shift angolare
    
    for theta in angles_deg:
        theta_corr = theta + abs(start_taxi)
        theta_corrected.append(theta_corr)
        pulses_corrected.append(theta_corr * pulse_per_deg)

    # correzione fine taxi
    theta_end_corrected = 180.0 + end_taxi     # posizione angolare in cui il trigger deve fermarsi
    pulses_end_corrected = theta_end_corrected * pulse_per_deg    # dopo questo stop alla generezione di trigger 
    return np.array(pulses_corrected, dtype=int), int(pulses_end_corrected), theta_corrected, theta_end_corrected

# ----------------------------------------------------
# Applico la correzione taxi
# ----------------------------------------------------
pulses_corrected, pulses_end_corrected, theta_corrected, theta_end_corrected = taxi_correct(
    angles_timbir, start_taxi, end_taxi, counts_per_rev
)

# ------------------------------
# Calcolo ritardi Δpulses per memPulseSeq
# ------------------------------
def compute_deltas(pulses):
    pulses = np.array(pulses, dtype=int)
    deltas = [pulses[0]] + [pulses[i] - pulses[i-1] for i in range(1, len(pulses))]
    return np.array(deltas, dtype=int)

delta_pulses = compute_deltas(pulses_corrected)

# ------------------------------
#  accelerazione, plateau, decelerazione
# ------------------------------
def compute_real_timeline(theta_corrected, counts_per_rev):
    # PV reali del rotary
    pv_vmax = PV("2bmb:m102.VMAX")
    pv_velo = PV("2bmb:m102.VELO")
    pv_vbas = PV("2bmb:m102.VBAS")
    pv_accl = PV("2bmb:m102.ACCL")

    VELO = float(pv_velo.get() or 1.0)
    VBAS = float(pv_vbas.get() or 0.1)
    ACCL = float(pv_accl.get() or 0.1)

    pulse_per_deg = counts_per_rev / 360.0
    pulses_timeline = []

    theta_accel = (VELO**2 - VBAS**2) / (2 * ACCL)  # assumo accelerazione uniforme
    theta_decel = theta_accel
    theta_plateau = 180.0 - theta_accel - theta_decel

    for theta in theta_corrected:
        if theta <= theta_accel:
            # accelerazione
            pulses = theta * pulse_per_deg
        elif theta <= (theta_accel + theta_plateau):
            # plateau
            pulses = theta * pulse_per_deg
        else:
            # decelerazione
            pulses = theta * pulse_per_deg
        pulses_timeline.append(int(round(pulses)))

    return np.array(pulses_timeline, dtype=int)

pulses_real = compute_real_timeline(theta_corrected, counts_per_rev)

# ------------------------------
# Output finale
# ------------------------------
print("Angoli TIMBIR:", angles_timbir)
print("Angoli corretti:", theta_corrected)
print("Impulsi corretti:", pulses_corrected)
print("Δpulses:", delta_pulses)
print("Impulsi reali (timeline):", pulses_real)
print("Impulso stop:", pulses_end_corrected)
