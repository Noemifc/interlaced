import numpy as np
import math
import struct
import matplotlib.pyplot as plt

# ============================================================================
#                     CLASSE INTERLACED SCAN
# ============================================================================

class InterlacedScan:
    """
        - logica TomoScanPSO e nomenclatura
        - generazione angoli interlacciati TIMBIR
        - correzione taxi
        - conversione angoli to impulsi
        - esportazione impulsi in formato binario per memPulseSeq
        - grafici di verifica
    """

    # ----------------------------------------------------------------------
    # init e parametri
    # ----------------------------------------------------------------------
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
        self.rotation_start = rotation_start   # angolo iniziale della scansione
        self.rotation_stop = rotation_stop     # angolo finale della scansione
        self.num_angles = num_angles           # num proiezioni
        self.K_interlace = K_interlace         # nuovo pv

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

        '''
        divide l'intervallo di scansione in N punti equidistanti: serve alla parte meccanica
        per stabilire la velocità motore, velocità taxi, finestra PSO e conversione in impulsi.
        La cinematica richiede comunque uno spacing nominale costante.
        '''

    # ----------------------------------------------------------------------
    #  TomoScanPSO.compute_senses()
    # ----------------------------------------------------------------------
    '''
    Determina in che direzione il sistema encoder conterà gli impulsi durante la scansione.
    Utile per PSO e taxi.
    '''
    def compute_senses(self):

        encoder_dir = 1 if self.PSOCountsPerRotation > 0 else -1
        motor_dir = 1 if self.RotationDirection == 0 else -1
        user_dir = 1 if self.rotation_stop > self.rotation_start else -1
        return encoder_dir * motor_dir * user_dir, user_dir

    # ----------------------------------------------------------------------
    #  Tempo per Frame
    # ----------------------------------------------------------------------
    '''
    Tempo totale richiesto dalla camera per acquisire una singola immagine.
    Tempo totale per frame = esposizione + readout
    '''
    def compute_frame_time(self):

        return self.exposure + self.readout

    '''
    Nell'exposure time il sensore vede il fascio e accumula fotoni.
    Durante il readout la camera non può acquisire un nuovo frame.
    '''

    # ----------------------------------------------------------------------
    #  compute_positions_PSO()
    # ----------------------------------------------------------------------
    '''
    Come il motore si muove effettivamente con rotation_step in impulsi interi.
    '''
    def compute_positions_PSO(self):

        overall_sense, user_direction = self.compute_senses()
        encoder_multiply = self.PSOCountsPerRotation / 360.0

        # Correzione step -> impulsi interi
        raw_counts = self.rotation_step * encoder_multiply
        delta_counts = round(raw_counts)
        self.rotation_step = delta_counts / encoder_multiply

        # Velocità motore
        dt = self.compute_frame_time()
        self.motor_speed = abs(self.rotation_step) / dt

        '''
        v = motor_speed (velocità finale)
        t = RotationAccelTime
        s = distanza necessaria per accelerare
        '''
        accel_dist = 0.5 * self.motor_speed * self.RotationAccelTime

        '''
        Se overall_sense<0 l'encoder conta al contrario rispetto al movimento.
        Serve offset.
        '''
        if overall_sense > 0:
            self.rotation_start_new = self.rotation_start
        else:
            self.rotation_start_new = self.rotation_start - (2 - self.readout_margin) * self.rotation_step

        # Taxi
        taxi_steps = math.ceil((accel_dist / abs(self.rotation_step)) + 0.5)
        taxi_dist = taxi_steps * abs(self.rotation_step)

        # Flyscan logic
        self.PSOStartTaxi = self.rotation_start_new - taxi_dist * user_direction
        self.rotation_stop_new = self.rotation_start_new + (self.num_angles - 1) * self.rotation_step
        self.PSOEndTaxi = self.rotation_stop_new + taxi_dist * user_direction

        # Angoli classici equispaziati
        self.theta_classic = self.rotation_start_new + np.arange(self.num_angles) * self.rotation_step

        '''
        Non sono angoli TIMBIR ma equispaziati
        '''

    # ----------------------------------------------------------------------
    # TIMBIR — bit reverse
    # ----------------------------------------------------------------------
    def bit_reverse(self, n, bits):
        return int(f"{n:0{bits}b}"[::-1], 2)

    # ----------------------------------------------------------------------
    #   Genera angoli TIMBIR interlacciati
    # ----------------------------------------------------------------------
    def generate_interlaced_angles(self):

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

        self.theta_interlaced = np.sort(theta)
        self.theta_interlaced_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(theta)))

        # Added plot to verify TIMBIR angle locations
        group_indices = np.array(group_indices)
        radii = 1 - group_indices * 0.15
        # Plot acquisition sequence
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(f"TIMBIR Interlaced Acquisition (N={self.num_angles} - K={self.K_interlace })\nEach loop on its own circle",
                     va='bottom', fontsize=13)

        ax.plot(np.deg2rad(theta), radii, '-o', lw=1.2, ms=5, alpha=0.8, color='tab:blue')

        for i in range(self.num_angles):
            ax.text(theta[i], radii[i] + 0.03,
                    str(group_indices[i] + 1), ha='center', va='bottom', fontsize=8)

        ax.set_rticks([])
        plt.show()

    # ----------------------------------------------------------------------
    # Modello taxi
    # ----------------------------------------------------------------------
    '''
    Simulazione moto con accelerazione = regime = decelerazione.
    '''
    def simulate_taxi_motion(self, omega_target=10, dt=1e-4):

        accel = decel = omega_target / self.RotationAccelTime

        # Accelerazione
        T_acc = omega_target / accel
        t_acc = np.arange(0, T_acc, dt)
        theta_acc = 0.5 * accel * t_acc**2

        # Tratto uniforme
        theta_flat_len = 360 - 2 * theta_acc[-1]
        T_flat = theta_flat_len / omega_target
        t_flat = np.arange(0, T_flat, dt)
        theta_flat = theta_acc[-1] + omega_target * t_flat

        # Decelerazione
        T_dec = omega_target / decel
        t_dec = np.arange(0, T_dec, dt)
        theta_dec = theta_flat[-1] + omega_target * t_dec - 0.5 * decel * t_dec**2

        # Concatenate
        self.t_vec = np.concatenate([t_acc,
                                     t_acc[-1] + t_flat,
                                     t_acc[-1] + t_flat[-1] + t_dec])

        self.theta_vec = np.concatenate([theta_acc, theta_flat, theta_dec])

    # ----------------------------------------------------------------------
    # tempi reali = angoli TIMBIR
    # ----------------------------------------------------------------------
    '''
    Interpola gli angoli TIMBIR sul moto reale simulato
    '''
    def compute_real_motion(self):

        self.t_real = np.interp(self.theta_interlaced, self.theta_vec, self.t_vec)
        self.theta_real = np.interp(self.t_real, self.t_vec, self.theta_vec)

    # ----------------------------------------------------------------------
    # Converte angoli = impulsi
    # ----------------------------------------------------------------------
    def convert_angles_to_counts(self):

        pulses_per_degree = self.PSOCountsPerRotation / 360.0

        self.PSOCountsIdeal = np.round(self.theta_interlaced * pulses_per_degree).astype(int)
        self.PSOCountsTaxiCorrected = np.round(self.theta_real * pulses_per_degree).astype(int)

        self.PSOCountsFinal = self.PSOCountsTaxiCorrected.copy()

        '''
        impulsi ideali se il motore fosse perfetto
        '''
        # ----------------------------------------------------------------------
        # Error between the angle in floating point and the angle rounded to the 
        # closest encoder pulse
        # ----------------------------------------------------------------------
        # 1) Closest integer pulse number
        pulse_counts = np.round(self.theta_interlaced / 360.0 * self.PSOCountsPerRotation).astype(int)

        # 2) Actual angle of those pulses (these are the one we want to save with the projections in the hdf file)
        actual_angles = pulse_counts / pulses_per_degree

        # 3) Angular error (actual - desired)
        angular_error = actual_angles - self.theta_interlaced

        # Print results nicely
        for a, p, act, err in zip(self.theta_interlaced, pulse_counts, actual_angles, angular_error):
            print(f"Target: {a:8.2f} deg | Pulse: {p:6d} | Actual: {act:9.6f} deg | Error: {err:+.6f} deg")  

        print('********************* unwrapped angles *********************')
        # same for unwrapped angles
        # 1) Closest integer pulse number
        pulse_counts = np.round(self.theta_interlaced_unwrapped / 360.0 * self.PSOCountsPerRotation).astype(int)

        # 2) Actual angle of those pulses (these are the one we want to save with the projections in the hdf file)
        actual_angles = pulse_counts / pulses_per_degree

        # 3) Angular error (actual - desired)
        angular_error = actual_angles - self.theta_interlaced_unwrapped

        # Print results nicely
        for a, p, act, err in zip(self.theta_interlaced_unwrapped, pulse_counts, actual_angles, angular_error):
            print(f"Target: {a:8.2f} deg | Pulse: {p:6d} | Actual: {act:9.6f} deg | Error: {err:+.6f} deg")  

 
    # ----------------------------------------------------------------------
    # Plot comparativi
    # ----------------------------------------------------------------------
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


# ============================================================================
# ============================================================================
if __name__ == "__main__":

    scan = InterlacedScan(num_angles=32, K_interlace=4, PSOCountsPerRotation=20000)

    scan.compute_positions_PSO()
    scan.generate_interlaced_angles()
    scan.simulate_taxi_motion()
    scan.compute_real_motion()
    scan.convert_angles_to_counts()

    scan.plot_all_comparisons()
