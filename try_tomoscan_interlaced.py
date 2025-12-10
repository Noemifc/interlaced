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

        self.theta_interlaced_timbir = []
        self.PSOCountsIdeal          = []
        self.PSOCountsTaxiCorrected  = []
        self.PSOCountsFinal          = []
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

        #Flyscan logic 
        self.PSOStartTaxi = self.rotation_start_new - taxi_dist * user_direction                       # PSOStartTaxi = posizione da cui parte l’accelerazione
        self.rotation_stop_new = self.rotation_start_new + (self.num_angles - 1) * self.rotation_step  # rotation_stop_new = ultimo angolo di acquisizione
        self.PSOEndTaxi   = self.rotation_stop_new + taxi_dist * user_direction                        # PSOEndTaxi = posizione dove termina la decelerazione

        # Angoli "classici"
        self.theta_classic = self.rotation_start_new + np.arange(self.num_angles) * self.rotation_step # theta_classic = la lista di angoli equispaziati da acquisire

        ''' non sono angoli timbir ma quelli equispaziati per acquisizione standard , usati per calcolare correttamente rotation_step , v motore e correzioni taxi 
        '''

    # ----------------------------------------------------------------------
    # TIMBIR — bit reverse
    # ----------------------------------------------------------------------
    def bit_reverse(self, n, bits):
        return int(f"{n:0{bits}b}"[::-1], 2)        # Converte n in binario su   bit lo inverte e lo riporta a intero

    # ----------------------------------------------------------------------
    #   Genera gli angoli TIMBIR interlacciati 
    # ----------------------------------------------------------------------
    def generate_interlaced_timbir_angles(self):

        bits = int(np.log2(self.K_interlace))                         # bit necessari per rappresentare K_interlace
        theta = []                                                    # lista temporanea angoli TIMBIR

        for n in range(self.num_angles):
            group = (n * self.K_interlace // self.num_angles) % self.K_interlace      #  quale loop (0..K-1) contiene la proiezione n
            group_br = self.bit_reverse(group, bits)                                  # bit-reversal al numero del loop
            idx = n * self.K_interlace + group_br                                     #  TIMBIR interlacciato
            angle_deg = (idx % self.num_angles) * 360.0 / self.num_angles             # conversione dell’indice in angolo  
            theta.append(angle_deg)                                                   # aggiungi angolo alla lista
        self.theta_interlaced_timbir = np.sort(theta)                                 # ordina angoli



    # ----------------------------------------------------------------------
    # Modello  taxi
    # ----------------------------------------------------------------------
    '''
    omega_target = velocità angolare durante il tratto uniforme
    dt = passo temporale di simulazione

    con 
    - accelerazione = parte da 0 e arriva a ω_target
    - regime = ruota a ω_target (velocità costante)
    - decelerazione = rallenta da ω_target a 0
    '''
    def simulate_taxi_motion(self, omega_target=10, dt=1e-4):
       # accelerazione e decelerazione
        accel = decel = omega_target / self.RotationAccelTime
           
        T_acc = omega_target / accel                  
        t_acc = np.arange(0, T_acc, dt)                    # vettore di tempi che descrive tutti gli istanti durante la fase di accelerazione  da 0 a T_acc , usa passo dt
        theta_acc = 0.5 * accel * t_acc**2                 # moto uniformemente accelerato 
     
        # parte centrale ad omega cost
        theta_flat_len = 360 - 2 * theta_acc[-1]           # numero di gradi totali che il motore percorre nella fase a velocità costante
        T_flat = theta_flat_len / omega_target             # quanti secondi dura la fase a velocità costante
        t_flat = np.arange(0, T_flat, dt)                  # vettore dei tempi per T_flat
        theta_flat = theta_acc[-1] + omega_target * t_flat # calcolo angolo per ogni istante a v cost 
        #          = gradi raggiunti dopo accelerazione + v cost x tempo

      # speculare   
        T_dec = omega_target / decel
        t_dec = np.arange(0, T_dec, dt)
        theta_dec = theta_flat[-1] + omega_target*t_dec - 0.5*decel*t_dec**2

        self.t_vec = np.concatenate([t_acc, t_acc[-1] + t_flat,
                                     t_acc[-1] + t_flat[-1] + t_dec])                   # t_vec = vettore del tempo continuo da 0 fino alla fine del moto
        self.theta_vec = np.concatenate([theta_acc, theta_flat, theta_dec])             # theta_vec =  angoli reali generati dal modello cinematico
        
    ''' Shift:
    - accelerazione = finisce a t_acc[-1]
    - flat = deve iniziare esattamente dopo
    - decelerazione = deve iniziare esattamente alla fine del regime
    '''
        
    # ----------------------------------------------------------------------
    # tempo reale a cui il motore raggiunge ogni angolo TIMBIR
    # ----------------------------------------------------------------------
    ''' angoli TIMBIR = tempi reali
        tempi reali = angoli reali
    '''
    def compute_real_motion(self):
        
        self.t_real = np.interp(self.theta_interlaced_timbir, self.theta_vec, self.t_vec)    # tempo reale a cui il motore raggiunge ogni angolo TIMBIR
        self.theta_real = np.interp(self.t_real, self.t_vec, self.theta_vec)                 # angolo effettivamente raggiunto dal motore in quel momento
    
    # ----------------------------------------------------------------------
    # Converte angoli = impulsi PSO assoluti
    # ----------------------------------------------------------------------
    def convert_angles_to_counts(self):
      
        pulses_per_degree = self.PSOCountsPerRotation / 360.0
        self.PSOCountsIdeal = np.round(self.theta_interlaced_timbir * pulses_per_degree).astype(int)    # conversione degli angoli TIMBIR ideali in impulsi ideali
        self.PSOCountsTaxiCorrected = np.round(self.theta_real * pulses_per_degree).astype(int)         # converte angoli reali ( quelli corretti da taxi) in impulsi encoder assoluti
      
        self.PSOCountsFinal = self.PSOCountsTaxiCorrected.copy()         # copia gli impulsi taxi-corretti nel vettore finale da inviare alla FPGA

    '''impulsi che voglio inviare se il motore si muovesse perfettamente senza accelerazioni e ritardi'''

    # ----------------------------------------------------------------------
    # Grafico
    # ----------------------------------------------------------------------
    def plot_diagnostics(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.theta_interlaced_timbir, self.PSOCountsIdeal, 'o-', label="Impulsi ideali")
        plt.plot(self.theta_interlaced_timbir, self.PSOCountsTaxiCorrected, 'o-', label="Impulsi taxi-corretti")
        plt.plot(self.theta_interlaced_timbir, self.PSOCountsFinal, 'o-', label="Impulsi finali FPGA", linewidth=3, alpha=0.7)

        plt.xlabel("Angolo interlacciato (deg)")
        plt.ylabel("Impulsi PSO (assoluti)")
        plt.title("Diagnostica impulsi TIMBIR = FPGA")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    def show_list(self):
        print(self.PSOCountsIdeal)
        print(self.PSOCountsTaxiCorrected)
        print(self.PSOCountsFinal)
# ============================================================================
#                     ESEMPIO COMPLETO
# ============================================================================

if __name__ == "__main__":

    scan = InterlacedScan(num_angles=32, K_interlace=4)

    scan.compute_positions_PSO()
    # scan.generate_interlaced_angles()
    scan.generate_interlaced_timbir_angles()
    scan.simulate_taxi_motion()
    scan.compute_real_motion()
    scan.convert_angles_to_counts()

    scan.plot_diagnostics()
    scan.show_list()
    # scan.save_pulses_bin("pulses.bin")
