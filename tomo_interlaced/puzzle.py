"""

angles_timbir        → angoli ideali TIMBIR
theta_corrected      → angoli reali corretti dal taxi
pulses_corrected     → impulsi assoluti
delta_pulses         → ritardi da dare al memPulseSeq (capisci se e' corretto)
pulses_end_corrected → impulso di stop

"""


import numpy as np
from epics import PV

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# TIMBIR e nuovi PVs 
# ------------------------------------------------------------------------------------------------------------------------------------------------------

pv_N_theta = PV("2bmb:TomoScan:NTheta")          # numero totale proiezioni  NUOVO PV
pv_K       = PV("2bmb:TomoScan:KLoops")         # numero di loop interlacciati  NUOVO PV

N_theta = int(pv_N_theta.get()) or 32  # numero totale proiezioni 
K       = int(pv_K.get()) or 4          # numero di loop interlacciati 

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
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNZIONE TAXI CORRECTION con theta_corrected angoli di timbir corretti 
# ------------------------------------------------------------------------------------------------------------------------------------------------------
 """
    Applica la correzione taxi (taxi_correct) per spostare gli angoli secondo inizio/fine taxi
    Sposta ogni angolo TIMBIR per il taxi iniziale
    
    Parametri:
        angles_deg      : array degli angoli TIMBIR [deg]

    Ritorna:
        pulses_corrected      : array impulsi corretti per il PSO
        pulses_end_corrected  : impulso corrispondente alla fine taxi
        theta_corrected       : angoli TIMBIR corretti per start taxi
        theta_end_corrected   : angolo finale corretto per end taxi
    """

# ----------------------------------------------------
# EPICS PVs per Taxi e PSO
# ----------------------------------------------------
pv_start_taxi = PV("2bmb:TomoScan:PSOStartTaxi")         # Posizione di inizio taxi [deg]
pv_end_taxi   = PV("2bmb:TomoScan:PSOEndTaxi")           # Posizione di fine taxi [deg]
pv_counts     = PV("2bmb:TomoScan:PSOCountsPerRotation") # Numero di impulsi per giro del PSO

# Lettura dai PV
start_taxi     = pv_start_taxi.get()           # es: -0.749939 deg angolo di inizio taxi
end_taxi       = pv_end_taxi.get()             # es: 0.735 deg angolo di fine taxi 
counts_per_rev = pv_counts.get()               # es: 11_840_200 impulsi/giro impulsi per giro del PSO



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

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Generazione ritardi Δpulses per memPulseSeq 
# ------------------------------------------------------------------------------------------------------------------------------------------------------
"""  rotazione continua PSO conta impulsi in ingresso, ogni volta che parte un trigger,
il contatore si resetta al nuovo valore N quindi deve sapere solo di quanto aumentare N per andare dal trigger A al trigger B
 
"""
def compute_deltas(pulses_corrected):
    pulses_corrected = np.array(pulses_corrected, dtype=int)

    # Primo ritardo: dal punto zero encoder al primo trigger
    deltas = [pulses_corrected[0]]

    # Tutti gli altri: differenze successive
    for i in range(1, len(pulses_corrected)):
        delta = pulses_corrected[i] - pulses_corrected[i-1]
        deltas.append(int(delta))

    return np.array(deltas, dtype=int)

# applico agli impulsi
# delta_end = pulses_end_corrected - pulses_corrected[-1] # solo se richiesto 
delta_pulses = compute_deltas(pulses_corrected)

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Restituisce gli impulsi reali 
# ------------------------------------------------------------------------------------------------------------------------------------------------------
"""
    Genera la timeline impulsi realistica considerando:
    accelerazione - plateau - decelerazione
    usando i PV reali del rotary stage
    
    theta_corrected = angoli TIMBIR corretti per start-taxi [degrees]
    counts_per_rev  = impulsi encoder per giro (PSO) → es. 532800 per 2-BM
    Tutte le velocità sono in °/sec, lette dai PV del rotary stage.

    Durante la rampa la velocità cresce da Vbas → Vmax usando:
    θ(t) = 1/2 * a * t^2   →   t = sqrt(2θ / a)


    PSO riceve impulsi che rispettano la cinematica reale del motore, quindi la sincronizzazione con il detector durante la fly-scan è fisicamente consistente

Usa PV reali (VMAX, VELO, VBAS, ACCL) per calcolare accelerazione e plateau

Divide la rampa in 3 fasi: accelerazione, plateau, decelerazione

Converte angoli corretti TIMBIR in impulsi usando le risoluzioni reali

    """
def compute_real_timeline(theta_corrected, counts_per_rev):
    """
Legge i parametri reali del motore (velocità, accelerazione, risoluzioni…)
Divide la rotazione in tre fasi: accelerazione – plateau – decelerazione
Calcola quanti impulsi PSO x tempo/angolo del trigger
    
    Tutte le velocità sono in °/sec, lette dai PV del rotary stage
    """

    # ------------------------------
    # IMPORT EPICS PV
    # ------------------------------
    
    pv_vmax = PV("2bmb:m102.VMAX")   # velocità massima (plateau)
    pv_velo = PV("2bmb:m102.VELO")   # velocità corrente (debug)
    pv_vbas = PV("2bmb:m102.VBAS")   # velocità base a inizio rampa
    pv_accl = PV("2bmb:m102.ACCL")   # accelerazione (deg/s^2 o tempo ramp)
    pv_mres = PV("2bmb:m102.MRES")   # motor resolution
    pv_eres = PV("2bmb:m102.ERES")   # encoder resolution
    pv_rres = PV("2bmb:m102.RRES")   # readback resolution

    # ------------------------------
    # LETTURA PV
    # ------------------------------
    VELO = pv_velo.get()        # plateau target [deg/s]
    VBAS = pv_vbas.get()        # base velocity [deg/s]
    ACCL = pv_accl.get()        # accelerazione [deg/s^2]
    mres = pv_mres.get()
    eres = pv_eres.get()
    rres = pv_rres.get()

    # ------------------------------
    # Conversione angoli -> impulsi
    # ------------------------------
    pulse_per_deg = counts_per_rev / 360.0
    pulses_timeline = []

    # ------------------------------
    # Calcolo parametri rampa
    # ------------------------------
    # Accelerazione uniforme da VBAS -> VELO
    a_acc = ACCL
    theta_accel = (VELO**2 - VBAS**2) / (2 * a_acc)  # angolo percorso in accelerazione da VBAS a VELO

    # Decelerazione uniforme: simmetrica accelerazione
    a_dec = a_acc
    theta_decel = theta_accel  # simmetrico

    # Plateau
    theta_plateau = 180.0 - theta_accel - theta_decel  # angolo percorso a VELO costante

    # ------------------------------
    # Loop su tutti gli angoli corretti : valuto ogni angolo dove si trova
    # ------------------------------
    for theta in theta_corrected:

        # -----------------------------------------------
        # 1) Fase accelerazione
        # -----------------------------------------------
        if theta <= theta_accel:
            # moto uni accelerato
            t = (np.sqrt(VBAS**2 + 2*a_acc*theta) - VBAS) / a_acc
            pulses = theta * pulse_per_deg
            pulses_timeline.append(int(pulses))
            continue

        # -----------------------------------------------
        # 2) Fase plateau
        # -----------------------------------------------
        elif theta <= (theta_accel + theta_plateau):
            # angolo rimanente dopo accelerazione
            theta_remain = theta - theta_accel
            t_plateau = theta_remain / VELO
            pulses = theta * pulse_per_deg
            pulses_timeline.append(int(pulses))
            continue

        # -----------------------------------------------
        # 3) Fase decelerazione
        # -----------------------------------------------
        else:
            # angolo rimanente da percorrere in decelerazione
            theta_remain = 180.0 - theta
            t_decel = (VELO - np.sqrt(VELO**2 - 2*a_dec*theta_remain)) / a_dec
            pulses = theta * pulse_per_deg
            pulses_timeline.append(int(pulses))
            continue

    return np.array(pulses_timeline, dtype=int)


# ----------------------------------------------------
# FUNZIONE DI CONVERSIONE ANGOLI CORRETTI -> IMPULSI PSO : converte solo in impulsi, mantenendo la correzione del taxi già applicata
# ----------------------------------------------------
def angles_corrected_to_pulses_epics(theta_corrected):
    """
    Converte un array di angoli TIMBIR corretti per il taxi (in gradi)
    in impulsi PSO leggendo PSOCountsPerRotation dal PV EPICS.

    Parametri:
        theta_corrected : array/lista di angoli TIMBIR corretti [deg]

    Ritorna:
        pulses : array di impulsi PSO interi
    """
    # Lettura numero di impulsi per giro dal PV
    counts_per_rev = float(pv_counts.get())

    # Assicuro che sia array NumPy
    theta_corrected = np.array(theta_corrected)

    # Conversione angoli -> impulsi
    pulses = theta_corrected * (counts_per_rev / 360.0)

    # Arrotondo e converto in interi
    return np.round(pulses).astype(int)

# ----------------------------------------------------

