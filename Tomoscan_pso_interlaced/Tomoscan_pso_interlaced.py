import numpy as np
import math
import struct
import matplotlib.pyplot as plt
import argparse
# ============================================================================
#                     CLASSE INTERLACED SCAN
# ============================================================================

class InterlacedScan:

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
                 K_interlace=5):

        # Parametri di scansione
        self.rotation_start = rotation_start  # angolo iniziale della scansione
        self.rotation_stop = rotation_stop  # angolo finale della scansione
        self.num_angles = num_angles  # num proiezioni
        self.K_interlace = K_interlace  # nuovo pv

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

    ############################################################ MODE ####################################################################

    # ----------------------------------------------------------------------
    #   TIMBIR
    # ----------------------------------------------------------------------
    def generate_interlaced_timbir(self):
        '''il fattore di interlacciamento K deve essere una potenza di due affinché l’operazione di
        bit reversal produca una permutazione uniforme e completa degli indici di loop
        i valori non binari di K introducono aliasing e distribuzioni non omogenee degli angoli
        verificare i funzionamenti a vari k 
        '''

        bits = int(np.log2(self.K_interlace))
        theta = []
        group_indices = []
        assert (self.K_interlace & (self.K_interlace - 1)) == 0   #bit rev definito su n fisso di bit , la permutazione è completa solo con vincolo di pot di 2 


        for n in range(self.num_angles):
            group = (n * self.K_interlace // self.num_angles) % self.K_interlace
            group_br = self.bit_reverse(group, bits)
            idx = n * self.K_interlace + group_br
            angle_deg = (idx % self.num_angles) * 360.0 / self.num_angles
            theta.append(angle_deg)
            group_indices.append(group)

        self.theta_interlaced = np.sort(theta)
        self.theta_interlaced_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(theta)))

        group_indices = np.array(group_indices)
        radii = 1 - group_indices * 0.15

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"TIMBIR Interlaced Acquisition (N={self.num_angles} - K={self.K_interlace})\nEach loop on its own circle",
            va='bottom', fontsize=13)

        ax.plot(np.deg2rad(theta), radii, '-o', lw=1.2, ms=5, alpha=0.8, color='tab:blue')

        for i in range(self.num_angles):
            ax.text(theta[i], radii[i] + 0.03,
                    str(group_indices[i] + 1), ha='center', va='bottom', fontsize=8)

        ax.set_rticks([])
        plt.show()

    def bit_reverse(self, n, bits):
        return int(f"{n:0{bits}b}"[::-1], 2)


    # unwrap solo per analisi temporale, non rappresenta traiettoria fisica reale



    # ----------------------------------------------------------------------
    #   GOLDEN ANGLE
    # ----------------------------------------------------------------------
    def generate_interlaced_goldenangle(self):

        golden_angle = 360 * (3 - np.sqrt(5)) / 2
        phi_inv = (np.sqrt(5) - 1) / 2                         #utile per offests

        angles_all = []

        base = np.array([
            (self.rotation_start + i * golden_angle) % 360
            for i in range(self.num_angles)
        ])
        base = np.sort(base)
        angles_all.append(base)

        for k in range(1, self.K_interlace):
            offset = (k / (self.num_angles + 1)) * 360 * phi_inv
            angles_all.append(np.sort((base + offset) % 360))

        theta = np.sort(np.concatenate(angles_all))

        self.theta_interlaced = theta
        self.theta_interlaced_unwrapped = np.rad2deg(
            np.unwrap(np.deg2rad(theta))
        )

        return angles_all

    
    # Tabelle e plot Golden
    # ----------------------------------------------------------------------
    
    def print_angles_table(self, angles_all):
        print(f"{'Idx':>5}", end='')
        for k in range(len(angles_all)):
            print(f"{f'Loop {k + 1}':>12}", end='')
        print()

        for i in range(len(angles_all[0])):
            print(f"{i:5}", end='')
            for loop in angles_all:
                print(f"{loop[i]:12.3f}", end='')
            print()

    def print_cumulative_angles_table(self, angles_all):
        cumulative = [angles_all[0].copy()]

        for k in range(1, len(angles_all)):
            prev_max = cumulative[-1].max()
            cumulative.append(angles_all[k] + np.ceil(prev_max / 360) * 360)

        print(f"{'Idx':>5}", end='')
        for k in range(len(cumulative)):
            print(f"{f'Loop {k + 1}':>15}", end='')
        print()

        for i in range(len(cumulative[0])):
            print(f"{i:5}", end='')
            for loop in cumulative:
                print(f"{loop[i]:15.3f}", end='')
            print()

    def plot_interlaced_circles(self, angles_all):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        for k, angles in enumerate(angles_all):
            r = np.full_like(angles, 1 - k * 0.15)
            ax.plot(np.deg2rad(angles), r, 'o-', label=f'Loop {k + 1}')

        ax.set_rticks([])
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.legend()
        ax.set_title("Golden Angle – Interlaced (TIMBIR-style)")
        plt.show()

    # ----------------------------------------------------------------------
    #   EQUALLY SPACED – K-TURN
    # ----------------------------------------------------------------------
    def generate_interlaced_kturns(self, delta_theta=None):
        """
        θ_n = θ_start + n * dθ
        dθ = (θ_stop - θ_start) / (N - 1) o def da user
        """
       
        # Step
        
        if delta_theta is not None:
            delta_theta = float(delta_theta)
        else:
            delta_theta = (self.rotation_stop - self.rotation_start) / (self.num_angles - 1)

        self.rotation_step = delta_theta

        
        # single loop
       
        base = self.rotation_start + np.arange(self.num_angles) * delta_theta

        # multi-turn

        angles_all = []
        for k in range(self.K_interlace):
            angles_all.append(base + k * 360.0)

        # concateno tutti i loop
        theta_unwrapped = np.concatenate(angles_all)

        # versione modulo 360 (per PSO / FPGA)
        theta = np.mod(theta_unwrapped, 360.0)

         
        # for plot
       
        self.theta_interlaced = np.array(theta)
        self.theta_interlaced_unwrapped = np.array(theta_unwrapped)

        if self.K_interlace > 1:
            self.rotation_stop = theta_unwrapped[-1]   # motore ruota fino all'ultimo unwrapped

        return angles_all
    # round plot
    # stesso angolo viene acquisito a impulsi diversi in rotazioni fisiche successive nel 2 plot
    def plot_equally_loops_polar_kturns(self):

        # loop a partire da theta_unwrapped
        theta_unwrapped = self.theta_interlaced_unwrapped
        theta_mod = np.mod(theta_unwrapped, 360.0)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)

        ax.set_title(
            f"Equally Spaced Acquisition (N={self.num_angles}, K={self.K_interlace})\n"
            "Each loop on its own circle",
            va='bottom', fontsize=13
        )

        # Un cerchio per ogni loop
        for k in range(self.K_interlace):
            start = k * self.num_angles
            stop = (k + 1) * self.num_angles

            theta_k = theta_mod[start:stop]
            radii = np.full_like(theta_k, 1 - k * 0.15)

            ax.plot(
                np.deg2rad(theta_k),
                radii,
                '-o',
                lw=1.2,
                ms=5,
                alpha=0.85
            )

            # etichetta loop
            for i, ang in enumerate(theta_k):
                ax.text(
                    np.deg2rad(ang),
                    radii[i] + 0.03,
                    str(k + 1),
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        ax.set_rticks([])
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        plt.show()



    # ----------------------------------------------------------------------
    #   EQUALLY SPACED 
    # multi-turn acquisition (TIMBIR-like)
    # ----------------------------------------------------------------------

    def generate_interlaced_multiturns(self, delta_theta=None):

        N = self.num_angles
        K = self.K_interlace
       
        # Step
        if delta_theta is not None:
            delta_theta = float(delta_theta)
        else:
            delta_theta = (self.rotation_stop - self.rotation_start) / (N - 1)

        self.rotation_step = delta_theta

        n = np.arange(N)
        angles_all = []
        
        for k in range(K):
            theta_n = self.rotation_start + (n + k / K) * delta_theta   #  θ_n = θ_start + (n + k/Kloops) 
            angles_all.append(theta_n)

        # concateno tutti i loop
        theta_unwrapped = np.concatenate(angles_all)

        # versione modulo 360 (per PSO / FPGA)
        theta = np.mod(theta_unwrapped, 360.0)
         # for plot
        self.theta_interlaced = np.array(theta)
        self.theta_interlaced_unwrapped = np.array(theta_unwrapped)

        if self.K_interlace > 1:
            self.rotation_stop = theta_unwrapped[-1]   # motore ruota fino all'ultimo unwrapped

        return angles_all
    # round plot
    # stesso angolo viene acquisito a impulsi diversi in rotazioni fisiche successive nel 2 plot
    def plot_equally_loops_polar_multiturns(self):

        # loop a partire da theta_unwrapped
        theta_unwrapped = self.theta_interlaced_unwrapped
        theta_mod = np.mod(theta_unwrapped, 360.0)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)

        ax.set_title(
            f"Equally Spaced Acquisition (N={self.num_angles}, K={self.K_interlace})\n"
            "Each loop on its own circle",
            va='bottom', fontsize=13
        )

        # Un cerchio per ogni loop
        for k in range(self.K_interlace):
            start = k * self.num_angles
            stop = (k + 1) * self.num_angles

            theta_k = theta_mod[start:stop]
            radii = np.full_like(theta_k, 1 - k * 0.15)

            ax.plot(
                np.deg2rad(theta_k),
                radii,
                '-o',
                lw=1.2,
                ms=5,
                alpha=0.85
            )

            # etichetta loop
            for i, ang in enumerate(theta_k):
                ax.text(
                    np.deg2rad(ang),
                    radii[i] + 0.03,
                    str(k + 1),
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        ax.set_rticks([])
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        plt.show()



    # ----------------------------------------------------------------------
    #           FUNZIONI
    # ----------------------------------------------------------------------


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

        accel_dist = 0.5 * self.motor_speed * self.RotationAccelTime

        if overall_sense > 0:
            self.rotation_start_new = self.rotation_start
        else:
            self.rotation_start_new = self.rotation_start - (2 - self.readout_margin) * self.rotation_step

        taxi_steps = math.ceil((accel_dist / abs(self.rotation_step)) + 0.5)
        taxi_dist = taxi_steps * abs(self.rotation_step)

        self.PSOStartTaxi = self.rotation_start_new - taxi_dist * user_direction
        self.rotation_stop_new = self.rotation_start_new + (self.num_angles - 1) * self.rotation_step
        self.PSOEndTaxi = self.rotation_stop_new + taxi_dist * user_direction

        self.theta_classic = self.rotation_start_new + np.arange(self.num_angles) * self.rotation_step

    # ----------------------------------------------------------------------
    # Modello taxi
    # ----------------------------------------------------------------------
    def simulate_taxi_motion(self, omega_target=10, dt=1e-4):

        theta_required = self.theta_interlaced_unwrapped.max()
        theta_max = float(np.max(self.theta_interlaced_unwrapped))   #indipendenza dal metodo 
        #theta_max = self.theta_interlaced_unwrapped.max()   # rotazione tot del motore
        
        #accelerazione 
        accel = decel = omega_target / self.RotationAccelTime
        theta_acc = 0.5 * accel * t_acc**2
        theta_acc_end = theta_acc[-1]
        t_acc = np.arange(0, self.RotationAccelTime, dt)

        # plateau
        theta_flat_len = theta_max - 2 * theta_acc_end
        if theta_flat_len < 0:
            raise ValueError("Profilo di moto non realizzabile")

        t_flat = np.arange(0, theta_flat_len / omega_target, dt)
        theta_flat = theta_acc_end + omega_target * t_flat

        # decelerazione
        t_dec = np.arange(0, self.RotationAccelTime, dt)
        theta_dec = ( theta_flat[-1] + omega_target * t_dec - 0.5 * decel * t_dec**2 )
       
        self.theta_vec = np.concatenate([theta_acc, theta_flat, theta_dec])
        self.t_vec = np.concatenate([
        t_acc,
        t_acc[-1] + t_flat,
        t_acc[-1] + t_flat[-1] + t_dec ])
    
        
     
"""
        theta_flat_len = theta_max - 2 * theta_acc[-1]    # dovrebbe generalizzare meglio 
        #theta_flat_len = 360 - 2 * theta_acc[-1]         # forzato a 360 fallisce con golden
       
        
        T_flat = theta_flat_len / omega_target
        t_flat = np.arange(0, T_flat, dt)
        theta_flat = theta_acc[-1] + omega_target * t_flat

        T_dec = omega_target / decel
        t_dec = np.arange(0, T_dec, dt)
        theta_dec = theta_flat[-1] + omega_target * t_dec - 0.5 * decel * t_dec ** 2

        self.theta_vec = np.concatenate([theta_acc, theta_flat, theta_dec])

        self.t_vec = np.concatenate([t_acc,
                                     t_acc[-1] + t_flat,
                                     t_acc[-1] + t_flat[-1] + t_dec]) """


    # ----------------------------------------------------------------------
    # tempi reali = angoli TIMBIR
    # ----------------------------------------------------------------------
    def compute_real_motion(self):

        # self.t_real = np.interp(self.theta_interlaced_unwrapped, self.theta_vec, self.t_vec)   
        self.t_real = np.interp(self.theta_interlaced, self.theta_vec, self.t_vec)
        self.theta_real = np.interp(self.t_real, self.t_vec, self.theta_vec)

    # ----------------------------------------------------------------------
    # Converte angoli = impulsi
    # ----------------------------------------------------------------------
    def convert_angles_to_counts(self):

        pulses_per_degree = self.PSOCountsPerRotation / 360.0

        self.PSOCountsIdeal = np.round(self.theta_interlaced * pulses_per_degree).astype(int)
        self.PSOCountsTaxiCorrected = self.theta_real * pulses_per_degree

        self.PSOCountsFinal = self.PSOCountsTaxiCorrected.copy()

        pulse_counts = np.round(self.theta_interlaced / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_angles = pulse_counts / pulses_per_degree
        angular_error = actual_angles - self.theta_interlaced

        for a, p, act, err in zip(self.theta_interlaced, pulse_counts, actual_angles, angular_error):
            print(f"Target: {a:8.2f} deg | Pulse: {p:6d} | Actual: {act:9.6f} deg | Error: {err:+.6f} deg")

        print('********************* unwrapped angles *********************')
        pulse_counts = np.round(self.theta_interlaced_unwrapped / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_angles = pulse_counts / pulses_per_degree
        angular_error = actual_angles - self.theta_interlaced_unwrapped

        for a, p, act, err in zip(self.theta_interlaced_unwrapped, pulse_counts, actual_angles, angular_error):
            print(f"Target: {a:8.2f} deg | Pulse: {p:6d} | Actual: {act:9.6f} deg | Error: {err:+.6f} deg")

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

    def plot(self):
        x1 = self.theta_interlaced
        x2 = self.theta_interlaced_unwrapped
        pulse_counts = np.round(self.theta_interlaced_unwrapped / 360.0 * self.PSOCountsPerRotation).astype(int)
        y = pulse_counts

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharey=True)

        # Plot 1: angoli interlacciati
        axs[0].plot(x1, y, 'o-', color='tab:blue', label='Impulsi vs Angolo')
        axs[0].set_title('Angoli vs Impulsi encoder')
        axs[0].set_xlabel('Angoli interlacciato [deg]')
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


# ============================================================================
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run interlaced scan simulation.")
    parser.add_argument(
        "--num_angles",
        type=int,
        default=32,
        help="Number of angles (default: 32)",
    )
    parser.add_argument(
        "--K_interlace",
        type=int,
        default=4,
        help="Interlace factor K (default: 4)",
    )
    parser.add_argument(
        "--mode",
        choices=["timbir", "golden", "kturns", "multiturns"],
        default="timbir",
    )
    parser.add_argument(
        "--PSOCountsPerRotation",
        type=int,
        default=20,
        help="PSO counts per rotation (default: 20)",
    )

    args = parser.parse_args()

    scan = InterlacedScan(
        num_angles=args.num_angles,
        K_interlace=args.K_interlace,
        PSOCountsPerRotation=args.PSOCountsPerRotation,
    )

    # select method
    if args.mode == "timbir":
        scan.generate_interlaced_timbir()

    elif args.mode == "golden":
        angles_all = scan.generate_interlaced_goldenangle()
        scan.print_angles_table(angles_all)
        scan.print_cumulative_angles_table(angles_all)
        scan.plot_interlaced_circles(angles_all)
        
    elif args.mode == "kturns":
        scan.generate_interlaced_kturns()
        scan.plot_equally_loops_polar_kturns()
        
    elif args.mode == "multiturns":
        scan.generate_interlaced_multiturns()
        scan.plot_equally_loops_polar_multiturns()



    

    # sorted
    scan.compute_positions_PSO()
    scan.simulate_taxi_motion()
    scan.compute_real_motion()
    scan.convert_angles_to_counts()


    scan.plot_all_comparisons()
    scan.plot()



if __name__ == "__main__":
    main()
