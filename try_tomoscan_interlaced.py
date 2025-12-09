"""
theta_interlaced           = angoli TIMBIR ordinati
theta_real_interlaced      = angoli TIMBIR corretti dal taxi model
pulses_interlaced_ideal    = conversione diretta
pulses_interlaced_real     = conversione taxi-corretta
PSOCountsFinal             = impulsi per FPGA
"""


import numpy as np
import math
import struct
import matplotlib.pyplot as plt

# ============================================================================
#                     CLASSE INTERLACED SCAN 
# ============================================================================

class InterlacedScan:
    """
        - logica nominale di TomoScanPSO 
        - generazione angoli interlacciati TIMBIR
        - correzione taxi  
        - conversione angoli to impulsi
        - esportazione impulsi in formato binario per memPulseSeq 
        - grafici di verifica 

     nomenclatura è coerente con TomoScanPSO
    """

    # ----------------------------------------------------------------------
    # init e parametri
    # ----------------------------------------------------------------------
    def __init__(self,
                 rotation_start = 0.0,
                 rotation_stop = 360.0,
                 num_angles = 32,        
                 PSOCountsPerRotation = 20000,
                 RotationDirection = 0,
                 RotationAccelTime = 0.15,
                 exposure = 0.01,
                 readout = 0.01,
                 readout_margin = 1,
                 K_interlace = 4):      

        # Parametri di scansione
        self.rotation_start = rotation_start   # angolo iniziale della scansione 
        self.rotation_stop  = rotation_stop    # angolo finale della scansione 
        self.num_angles     = num_angles       # num proiezioni, nuovo pv
        self.K_interlace    = K_interlace      # nuovo pv

        # Parametri hardware
        self.PSOCountsPerRotation = PSOCountsPerRotation  # num impulsi encoder generati per 360 deg
        self.RotationDirection    = RotationDirection     # direzione positiva o negativa del motore
        self.RotationAccelTime    = RotationAccelTime     # tempo [s] che il motore impiega per raggiungere la velocità di scansione

        # Parametri camera
        self.exposure = exposure
        self.readout  = readout
        self.readout_margin = readout_margin

        # Distanza angolare tra due proiezioni consecutive: lo spacing nominale tra gli angoli
        self.rotation_step = (rotation_stop - rotation_start) / (num_angles - 1)
'''
divide l'intervallo di scansione in N punti equidistanti: serve alla parte meccanica per stabilire la velocità motore, velocità taxi
 finestra PSO e conversione in impulsi 
 la cinematica del motore richiede comunque uno spacing nominale costante
'''
    # ----------------------------------------------------------------------
    #  TomoScanPSO.compute_senses()
    # ----------------------------------------------------------------------
'''Determina in che direzione il sistema encoder conterà gli impulsi durante la scansione: utile per PSO, taxi
Bisogna capire il il verso finale del movimento perchè può muoversi in diverso senso
'''
    def compute_senses(self):
      
        encoder_dir = 1 if self.PSOCountsPerRotation > 0 else -1
        motor_dir   = 1 if self.RotationDirection == 0 else -1
        user_dir    = 1 if self.rotation_stop > self.rotation_start else -1
        return encoder_dir * motor_dir * user_dir, user_dir

    # ----------------------------------------------------------------------
    #  Tempo per Frame
    # ----------------------------------------------------------------------
''' tempo totale richiesto dalla camera per acquisire una singola immagine, utile per calcolare la velocità di rotazione
    Tempo totale per frame = esposizione + readout
'''
    def compute_frame_time(self):
     
        return self.exposure + self.readout
        
''' nell exposure time il sensore vede il fascio, accumula fotoni e qui il 
movimento del campione dovrebbe essere lento o costante per evitare blur
+ readout dove la camera non può acquisire un nuovo frame
'''

    # ----------------------------------------------------------------------
    #  compute_positions_PSO() come TomoScan
    # ----------------------------------------------------------------------
'''  Come il motore si muove effettivamente  con   rotation_step in impulsi interi

'''
     def compute_positions_PSO(self):
        
        overall_sense, user_direction = self.compute_senses()  # determinare in quale direzione si muovono gli impulsi encoder, se aumentano o diminuiscono  e +/- 1 da user
        encoder_multiply = self.PSOCountsPerRotation / 360.0   # fattore di coversione  da angoli ad impulsi (impulsi per grado)

        '''  overall_sense= se impulsi encoder aumentano o diminuiscono 
            user_direction direzione della scansione in deg
        '''

        # Correzione step per avere impulsi interi
        raw_counts = self.rotation_step * encoder_multiply    # impulsi raw 
        delta_counts = round(raw_counts)                      # rounding : impulsi che effettivamente il motore farà
        self.rotation_step = delta_counts / encoder_multiply  # impulsi reali nell'angolo considerato

        # Velocità del motore
        dt = self.compute_frame_time()                        # tempo di un frame 
        self.motor_speed = abs(self.rotation_step) / dt       # motor_speed = step_angolare / tempo_per_frame

        # Distanza necessaria per accelerare s = 1/2 * v * t
         '''
         v = motor_speed = velocità finale da raggiungere
         t = RotationAccelTime = tempo necessario per raggiungerla
         s = accel_dist = distanza angolare necessaria per accelerare
         '''

        accel_dist = 0.5 * self.motor_speed * self.RotationAccelTime  #  distanza angolare necessaria per accelerare

        # Rotazione di partenza corretta e interpretazione corretta per il PSO
''' overall_sense > 0   =  impulsi encoder aumentano con la rotazione se < 0 è necessario nuovo offset
    vedere da quale angolo reale far partire il motore affinchè il primo impulso endìcoder sia coerente con la direz di conteggio 
'''
        if overall_sense > 0:
            self.rotation_start_new = self.rotation_start
        else:
            self.rotation_start_new = self.rotation_start - (2 - self.readout_margin) * self.rotation_step # impulsi che diminuiscono : encoder conta all’indietro mentre il motore va avanti

        # Taxi  : distanza necessaria all’accelerazione ->  numero intero di passi encoder
        taxi_steps = math.ceil((accel_dist / abs(self.rotation_step)) + 0.5)                           # = quanti step effettivi servono distanza continua[deg], ceil arrotonda per eccesso  + mergine sicurezza 0.5
        taxi_dist  = taxi_steps * abs(self.rotation_step)                                              # = la distanza reale di taxi[deg]= step_interi × step_angolare

        self.PSOStartTaxi = self.rotation_start_new - taxi_dist * user_direction
        self.rotation_stop_new = self.rotation_start_new + (self.num_angles - 1) * self.rotation_step
        self.PSOEndTaxi   = self.rotation_stop_new + taxi_dist * user_direction

        # Angoli "classici"
        self.theta_classic = self.rotation_start_new + np.arange(self.num_angles) * self.rotation_step

    # ----------------------------------------------------------------------
    # TIMBIR — bit reverse
    # ----------------------------------------------------------------------
    def bit_reverse(self, n, bits):
        return int(f"{n:0{bits}b}"[::-1], 2)

      # ----------------------------------------------------------------------
    #  
    # ----------------------------------------------------------------------
'''  
'''
 
    def generate_interlaced_angles(self):
        """
        Genera gli angoli TIMBIR interlacciati, ordinati in senso crescente.
        """
        bits = int(np.log2(self.num_angles))
        theta = np.array([
            self.bit_reverse(n, bits) * 360.0 / self.num_angles
            for n in range(self.num_angles)
        ])
        self.theta_interlaced = np.sort(theta)

    # ----------------------------------------------------------------------
    # Modello cinematico del taxi
    # ----------------------------------------------------------------------
    def simulate_taxi_motion(self, omega_target=10, dt=1e-4):
        """
        Simula accelerazione → regime → decelerazione
        per modellare angolo reale(t).
        """
        accel = decel = omega_target / self.RotationAccelTime

        T_acc = omega_target / accel
        t_acc = np.arange(0, T_acc, dt)
        theta_acc = 0.5 * accel * t_acc**2

        theta_flat_len = 360 - 2 * theta_acc[-1]
        T_flat = theta_flat_len / omega_target
        t_flat = np.arange(0, T_flat, dt)
        theta_flat = theta_acc[-1] + omega_target * t_flat

        T_dec = omega_target / decel
        t_dec = np.arange(0, T_dec, dt)
        theta_dec = theta_flat[-1] + omega_target*t_dec - 0.5*decel*t_dec**2

        self.t_vec = np.concatenate([t_acc, t_acc[-1] + t_flat,
                                     t_acc[-1] + t_flat[-1] + t_dec])
        self.theta_vec = np.concatenate([theta_acc, theta_flat, theta_dec])

    # ----------------------------------------------------------------------
    def compute_real_motion(self):
        """Trova i tempi reali in cui vengono raggiunti gli angoli TIMBIR."""
        self.t_real = np.interp(self.theta_interlaced, self.theta_vec, self.t_vec)
        self.theta_real = np.interp(self.t_real, self.t_vec, self.theta_vec)

    # ----------------------------------------------------------------------
    def convert_angles_to_counts(self):
        """Converte angoli → impulsi PSO assoluti."""
        pulses_per_degree = self.PSOCountsPerRotation / 360.0
        self.PSOCountsIdeal = np.round(self.theta_interlaced * pulses_per_degree).astype(int)
        self.PSOCountsTaxiCorrected = np.round(self.theta_real * pulses_per_degree).astype(int)
        self.PSOCountsFinal = self.PSOCountsTaxiCorrected.copy()

    # ----------------------------------------------------------------------
    # A — Salvataggio impulsi per memPulseSeq in formato BINARIO
    # ----------------------------------------------------------------------
    def save_pulses_bin(self, filename="pulses.bin"):
        """
        Salva gli impulsi assoluti in formato binario 32-bit LE,
        compatibile con memPulseSeq.
        """
        with open(filename, "wb") as f:
            for p in self.PSOCountsFinal:
                f.write(struct.pack("<i", int(p)))

        print(f"[OK] Salvato file binario per FPGA: {filename}")

    # ----------------------------------------------------------------------
    # B — Grafico diagnostico
    # ----------------------------------------------------------------------
    def plot_diagnostics(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.theta_interlaced, self.PSOCountsIdeal, 'o-', label="Impulsi ideali")
        plt.plot(self.theta_interlaced, self.PSOCountsTaxiCorrected, 'o-', label="Impulsi taxi-corretti")
        plt.plot(self.theta_interlaced, self.PSOCountsFinal, 'o-', label="Impulsi finali FPGA", linewidth=3, alpha=0.7)

        plt.xlabel("Angolo interlacciato (deg)")
        plt.ylabel("Impulsi PSO (assoluti)")
        plt.title("Diagnostica impulsi TIMBIR → FPGA")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


# ============================================================================
#                     ESEMPIO COMPLETO
# ============================================================================

if __name__ == "__main__":

    scan = InterlacedScan(num_angles=32, K_interlace=4)

    scan.compute_positions_PSO()
    scan.generate_interlaced_angles()
    scan.simulate_taxi_motion()
    scan.compute_real_motion()
    scan.convert_angles_to_counts()

    scan.plot_diagnostics()
    scan.save_pulses_bin("pulses.bin")
