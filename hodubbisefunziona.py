import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================================================#
#                     CLASSE INTERLACED SCAN
# ============================================================================#

class InterlacedScan:
    # ---------------------------------------------------------------------- #
    # init e parametri
    # ---------------------------------------------------------------------- #
    def __init__(self,
                 rotation_start=0.0,
                 rotation_stop=360.0,
                 num_angles=32,
                 PSOCountsPerRotation=20000,
                 RotationDirection=0,
                 RotationAccelTime=0.15,
                 exposure=0.01,
                 readout=0.01,
                 readout_margin=1,
                 K_interlace=4):
        
        # Parametri di scansione
        self.rotation_start = rotation_start  # angolo iniziale della scansione
        self.rotation_stop = rotation_stop  # angolo finale della scansione
        self.num_angles = num_angles  # num proiezioni
        self.K_interlace = K_interlace  # interlacciamento

        # Parametri hardware
        self.PSOCountsPerRotation = PSOCountsPerRotation
        self.RotationDirection = RotationDirection
        self.RotationAccelTime = RotationAccelTime

        # Parametri camera
        self.exposure = exposure
        self.readout = readout
        self.readout_margin = readout_margin

        # Distanza angolare nominale
        self.rotation_step = (rotation_stop - rotation_start) / (num_angles - 1)

    ################################################################################ 
    # METHOD
    ################################################################################ 

    def select_interlacing_method(self, method_name="Timbir"):
        """
        Seleziona il metodo di interlacciamento e genera gli angoli interlaced_metodo
        """
        interlacing_methods = {
            "Timbir": self.generate_interlaced_timbir,
            "GoldenAngle": self.generate_interlaced_goldenangle
        }

        if method_name in interlacing_methods:
            print(f"Select method : {method_name}")
            interlacing_methods[method_name]()  # Chiama il metodo selezionato
        else:
            print(f"Method '{method_name}' not found!")

    ##################################################################################################
    #               METODI DI INTERLACCIAMENTO
    ##################################################################################################

   # TIMBIR

    def generate_interlaced_timbir(self):
        bits = int(np.log2(self.K_interlace))
        theta = []
        group_indices = []

        for n in range(self.num_angles):
            group = (n * self.K_interlace // self.num_angles) % self.K_interlace
            group_br = self.bit_reverse(group, bits)
            idx = n * self.K_interlace + group_br
            angle_deg = (idx % self.num_angles) * 360.0 / self.num_angles
            theta.append(angle_deg)
            group_indices.append(group)

        # Salva l'angolo interlacciato generato valido per ogni metodo da aggiungere
        self.theta_interlaced = np.sort(theta)
        self.theta_interlaced_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(theta)))

    def bit_reverse(self, n, bits):
        """Funzione per il reverse dei bit"""
        return int(f"{n:0{bits}b}"[::-1], 2)

    # GoLDEN ANGLE

    def generate_interlaced_goldenangle(self):
        golden_angle = 360 * (3 - math.sqrt(5)) / 2  # Golden Angle
        theta = [(i * golden_angle) % 360 for i in range(self.num_angles)]
        
        # Salva l'angolo interlacciato generato valido per ogni metodo da aggiungere
        self.theta_interlaced = np.sort(theta)
        self.theta_interlaced_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(theta)))

   

    ##################################################################################################
    #               FUNZIONI DI CALCOLO E PLOT
    ##################################################################################################

   
    def compute_real_motion(self):
        
        self.t_real = np.interp(self.theta_interlaced, self.theta_vec, self.t_vec)
        self.theta_real = np.interp(self.t_real, self.t_vec, self.theta_vec)

    def convert_angles_to_counts(self):
     
        pulses_per_degree = self.PSOCountsPerRotation / 360.0
        self.PSOCountsIdeal = np.round(self.theta_interlaced * pulses_per_degree).astype(int)
        self.PSOCountsTaxiCorrected = self.theta_real * pulses_per_degree
        self.PSOCountsFinal = self.PSOCountsTaxiCorrected.copy()

    def plot_all_comparisons(self):
      
        ideal = self.PSOCountsIdeal
        real = self.PSOCountsTaxiCorrected
        final = self.PSOCountsFinal

        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        axs[0].plot(ideal, ideal, 'o--', alpha=0.6, label="Ideal")
        axs[0].plot(ideal, real, 'o-', alpha=0.9, label="Real (Taxi)")
        axs[0].set_title("Ideale vs Reale")
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(ideal, ideal, 'o--', alpha=0.6, label="Ideal")
        axs[1].plot(ideal, final, 'o-', alpha=0.9, label="Final FPGA")
        axs[1].set_title("Ideale vs FPGA")
        axs[1].grid()
        axs[1].legend()

        axs[2].plot(real, real, 'o--', alpha=0.6, label="Real")
        axs[2].plot(real, final, 'o-', alpha=0.9, label="Final FPGA")
        axs[2].set_title("Reale vs FPGA")
        axs[2].grid()
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    def plot(self):
        """Grafico finale"""
        x1 = self.theta_interlaced
        x2 = self.theta_interlaced_unwrapped
        pulse_counts = np.round(self.theta_interlaced_unwrapped / 360.0 * self.PSOCountsPerRotation).astype(int)
        y = pulse_counts

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharey=True)

        # Plot 1: angoli interlacciati
        axs[0].plot(x1, y, 'o-', color='tab:blue', label='Impulsi vs Angolo')
        axs[0].set_title('Angoli TIMBIR vs Impulsi encoder')
        axs[0].set_xlabel('Angolo interlacciato [deg]')
        axs[0].set_ylabel('Impulsi encoder')
        axs[0].grid(True)
        axs[0].legend()

        # Plot 2: angoli unwrapped
        axs[1].plot(x2, y, 's-', color='tab:orange', label='Impulsi vs Angolo Unwrapped')
        axs[1].set_title('Angoli TIMBIR Unwrapped vs Impulsi encoder')
        axs[1].set_xlabel('Angolo interlacciato Unwrapped [deg]')
        axs[1].set_ylabel('Impulsi encoder')
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        plt.show()


# ============================================================================#

if __name__ == "__main__":
    # Creazione oggetto e selezione del metodo
    scan = InterlacedScan(num_angles=32, K_interlace=4, PSOCountsPerRotation=20000)
    
    # Seleziona il metodo di interlacciamento
    scan.select_interlacing_method("Timbir")  # Sostituire con "GoldenAngle" per il metodo Golden Angle

    # Esegui i calcoli e i plot
    scan.compute_real_motion()
    scan.convert_angles_to_counts()

    # Plotta i risultati
    scan.plot_all_comparisons()
    scan.plot()
