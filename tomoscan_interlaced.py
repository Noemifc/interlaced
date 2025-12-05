import numpy as np
from epics import PV
# ----------------------------------------------------
# Lettura PV e Nuovi PV da aggiungere
# ----------------------------------------------------
pv_N_theta = PV("2bmb:TomoScan:NTheta")   # NUOVO PV
pv_Kloops = PV("2bmb:TomoScan:KloopsLoops")         # NUOVO PV  

# --------------------------------------------------------------------
# Interlaced TIMBIR
# --------------------------------------------------------------------
def generate_timbir_interlaced_angles():
    pv_N_theta = PV("2bmb:TomoScan:NTheta")   # numero totale proiezioni
    pv_Kloops = PV("2bmb:TomoScan:KloopsLoops")         # numero di loop interlacciati

    N_theta = int(pv_N_theta.get() or 32)  # numero totale proiezioni 
    Kloops = int(pv_Kloops.get() or 4)               # numero di loop interlacciati 
    # ----------------------------------------------------
    # BIT-REVERSAL
    # ----------------------------------------------------
    def bit_reverse(x, bits):
        b = f'{x:0{bits}b}'
        return int(b[::-1], 2)
    # ----------------------------------------------------
    # TIMBIR
    # ----------------------------------------------------
    angles_timbir = []
    loop_indices = []
    bits = int(np.log2(Kloops)) 

    for n in range(N_theta):
        base = n * Kloops
        loop = (base // N_theta) % Kloops
        rev = bit_reverse(loop, bits)
        val = base + rev

        theta = val * 360.0 / N_theta       # angolo 0-360°
        theta = theta % 180.0               # 0-180° per tomografia

        angles_timbir.append(theta)
        loop_indices.append(loop)

    angles_timbir = np.array(angles_timbir)
    loop_indices = np.array(loop_indices)

    print("Angles TIMBIR (degrees):")
    print(np.round(angles_timbir, 4))

    return angles_timbir, np.array(angles_timbir), loop_indices

#--------------------------------------------------------
''' se serve il passo angolare '''
# differenze tra angoli consecutivi
rotation_step_timbir = np.diff(angles_timbir)
rotation_step = np.mod(diffs, 180.0)

# ----------------------------------------------------
# EPICS PVs Pulses
# ----------------------------------------------------
'''

encoder_multiply =   pv_counts     = PV("2bmb:TomoScan:PSOCountsPerRotation") # Numero di impulsi per giro del PSO
raw_delta_encoder_counts = pulse_per_deg = counts_per_rev / 360.0  
delta_encoder_counts =   pulse_timbir


passo in gradi × counts_per_degree = passo in encoder counts
'''

   # Compute the actual delta to keep each interval an integer number of encoder counts
encoder_multiply = float(self.epics_pvs['PSOCountsPerRotation'].get()) / 360.                 # pulse_per_deg  quanti impulsi corrispondono a un grado , cioè quanti encoder counts corrispondono a un grado di rotazione
raw_delta_encoder_counts = self.rotation_step * encoder_multiply                              # quanti impulsi dovrebbe fare l’encoder per quel passo (rotation_step =rotation_step = passo angolare in gradi)
delta_encoder_counts = round(raw_delta_encoder_counts)          #why        pulse_timbir                    # arrotonda all’impulso intero più vicino


# che cosa si intende per rotation step perche arrotonda 
# sara' una nuova routine o si andra' a riconnettee a questa
# ----------------------------------------------------
self.epics_pvs['PSOEncoderCountsPerStep'].put(delta_encoder_counts)  # ogni step di rotazione vale X counts 
# Change the rotation step Python variable and PV
self.rotation_step = delta_encoder_counts / encoder_multiply #trasformo counts in angolo reale per vedere a quanti gradi reali corrispondono nel reale 
self.epics_pvs['RotationStep'].put(self.rotation_step)  # passo in gradi mandato ad episcs che viene utilizzato 


























