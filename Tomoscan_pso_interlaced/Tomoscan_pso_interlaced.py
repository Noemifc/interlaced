import numpy as np
import math
import matplotlib.pyplot as plt
import argparse


# ============================================================================
#                     CLASSE INTERLACED SCAN
# ============================================================================

class InterlacedScan:

    # ----------------------------------------------------------------------
    # init e parametri
    # ----------------------------------------------------------------------
    def __init__(
        self,
        rotation_start=0.0,
        rotation_stop=360.0,
        num_angles=32,
        PSOCountsPerRotation=20000,
        PSOPulsePerRotation=11840158,
        RotationDirection=0,
        RotationAccelTime=0.15,
        exposure=0.01,
        readout=0.01,
        readout_margin=1,
        K_interlace=4
    ):

        # Parametri di scansione
        self.rotation_start = float(rotation_start)
        self.rotation_stop = float(rotation_stop)
        self.num_angles = int(num_angles)
        self.K_interlace = int(K_interlace)

        # Parametri hardware
        self.PSOCountsPerRotation = int(PSOCountsPerRotation)
        self.RotationDirection = int(RotationDirection)
        self.RotationAccelTime = float(RotationAccelTime)
        self.PSOPulsePerRotation = int(PSOPulsePerRotation)

        # Parametri camera
        self.exposure = float(exposure)
        self.readout = float(readout)
        self.readout_margin = float(readout_margin)

        # Distanza angolare nominale (solo riferimento; alcuni metodi la sovrascrivono)
        if self.num_angles > 1:
            self.rotation_step = (self.rotation_stop - self.rotation_start) / (self.num_angles - 1)
        else:
            self.rotation_step = 0.0

        # placeholder (evita attribute errors se chiami robe fuori ordine)
        self.theta_interlaced = None
        self.theta_interlaced_unwrapped = None
        self.theta_monotonic = None

        # motion placeholders
        self.theta_vec = None
        self.t_vec = None
        self.t_real = None
        self.theta_real = None

        # counts placeholders
        self.PSOCountsIdeal = None
        self.PSOCountsTaxiCorrected = None
        self.PSOCountsFinal = None

    # ----------------------------------------------------------------------
    # utility
    # ----------------------------------------------------------------------
    def bit_reverse(self, n, bits):
        return int(f"{n:0{bits}b}"[::-1], 2)

    def _ensure_power_of_two_K(self):
        assert (self.K_interlace & (self.K_interlace - 1)) == 0, "K_interlace deve essere potenza di 2"

    # =========================================================
    # MODE
    # =========================================================

    # ----------------------------------------------------------------------
    #   TIMBIR
    # ----------------------------------------------------------------------
    def generate_interlaced_timbir(self):

        self._ensure_power_of_two_K()
        bits = int(np.log2(self.K_interlace))

        theta_acq = []
        group_indices = []

        for n in range(self.num_angles):
            group = (n * self.K_interlace // self.num_angles) % self.K_interlace
            group_br = self.bit_reverse(group, bits)
            idx = n * self.K_interlace + group_br
            angle_deg = idx * 360.0 / self.num_angles
            theta_acq.append(angle_deg)
            group_indices.append(group)

        theta_acq = np.asarray(theta_acq, dtype=float)

        # self.theta_interlaced = angoli interlacciati in ordine di acquisizione
        self.theta_interlaced = theta_acq.astype(float)

        # self.theta_interlaced_unwrapped = angoli interlacciati unwrap in ordine di acquisizione
        self.theta_interlaced_unwrapped = np.rad2deg(
            np.unwrap(np.deg2rad(theta_acq))
        ).astype(float)

        # self.theta_monotonic = tutti gli angoli acquisiti in ordine crescente (lista da mandare al PSO)
        self.theta_monotonic = np.sort(self.theta_interlaced_unwrapped).astype(float)

        # plot (cerchi per group)
        group_indices = np.asarray(group_indices, dtype=int)
        radii = 1.0 - group_indices * 0.15

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"TIMBIR Interlaced Acquisition (N={self.num_angles} - K={self.K_interlace})\n"
            "Each loop on its own circle",
            va="bottom",
            fontsize=13
        )

        ax.plot(np.deg2rad(theta_acq), radii, "-o", lw=1.2, ms=5, alpha=0.8, color="tab:blue")

        for i in range(self.num_angles):
            ax.text(
                np.deg2rad(theta_acq[i]),
                radii[i] + 0.03,
                str(group_indices[i] + 1),
                ha="center",
                va="bottom",
                fontsize=8
            )

        ax.set_rticks([])
        plt.show()

    # ----------------------------------------------------------------------
    #   multi-TIMBIR
    # ----------------------------------------------------------------------
    def generate_interlaced_multitimbir(self):
        """Ordine acquisizione timbir-like: per ogni posizione i nel giro acquisisco K loop in ordine bit-rev.
        Totale N*K viste.
        """

        self._ensure_power_of_two_K()
        N = self.num_angles
        K = self.K_interlace
        bits = int(np.log2(K))

        theta_acq = []
        group_indices = []

        # N*K viste per ogni loop (giro)
        for loop_turn in range(K):  # questo è il giro fisico (0..K-1)
            base_turn = 360.0 * loop_turn

            for i in range(N):
                for g in range(K):
                    

                    # indice su griglia fine (0..N*K-1) dentro al giro
                    idx = i * K + subloop
                    angle_deg = idx * 360.0 / (N * K)  # in [0,360)

                    # unwrapped: aggiungi 360 per il giro fisico
                    angle_unwrapped = angle_deg + base_turn

                    theta_acq.append(angle_unwrapped)
                    group_indices.append(loop_turn)  # etichetta del giro (loop) per i cerchi

        theta_acq = np.asarray(theta_acq, dtype=float)

        # ordine acquisizioneconc ettuale
        self.theta_interlaced = theta_acq.astype(float)

        # già unwrapped per costruzione
        self.theta_interlaced_unwrapped = theta_acq.copy().astype(float)

        # SOLO qui monotono crescente per PSO
        self.theta_monotonic = np.sort(self.theta_interlaced_unwrapped).astype(float)

        # plot cerchi separati per loop
        group_indices = np.asarray(group_indices, dtype=int)
        step = 0.8 / max(self.K_interlace - 1, 1)
        radii = 1.0 - group_indices * step

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"Multi-TIMBIR: N={self.num_angles} per loop, K={self.K_interlace} → tot N·K={self.num_angles*self.K_interlace}\n"
            "Loop over separate circles with loop order = bit-reversal",
            va="bottom",
            fontsize=12
        )

        ax.plot(np.deg2rad(theta_acq), radii, "o", lw=1.2, ms=5, alpha=0.8)

        for ang, r, lp in zip(theta_acq, radii, group_indices):
            ax.text(
                np.deg2rad(ang),
                r + 0.03,
                str(lp + 1),
                ha="center",
                va="bottom",
                fontsize=8
            )

        ax.set_rticks([])
        plt.show()

    # ----------------------------------------------------------------------
    #   GOLDEN ANGLE
    # ----------------------------------------------------------------------
    def generate_interlaced_goldenangle(self):
        golden_angle = 360.0 * (3.0 - np.sqrt(5.0)) / 2.0
        phi_inv = (np.sqrt(5.0) - 1.0) / 2.0

        angles_all = []

        base = np.array([(self.rotation_start + i * golden_angle) % 360.0
                         for i in range(self.num_angles)], dtype=float)
        base = np.sort(base)
        angles_all.append(base)

        for k in range(1, self.K_interlace):
            offset = (k / (self.num_angles + 1.0)) * 360.0 * phi_inv
            angles_all.append(np.sort((base + offset) % 360.0))

        # ordine acquisizione: loop1 poi loop2 poi loop3 ...
        theta_acq = np.concatenate(angles_all).astype(float)

        # self.theta_interlaced = angoli interlacciati in ordine di acquisizione
        self.theta_interlaced = theta_acq.astype(float)

        # self.theta_interlaced_unwrapped = angoli interlacciati unwrap in ordine di acquisizione
        self.theta_interlaced_unwrapped = np.rad2deg(
            np.unwrap(np.deg2rad(theta_acq))
        ).astype(float)

        # self.theta_monotonic = tutti gli angoli acquisiti in ordine crescente (lista da mandare al PSO)
        self.theta_monotonic = np.sort(self.theta_interlaced_unwrapped).astype(float)

        return angles_all

    # ----------------------------------------------------------------------
    # Tabelle e plot Golden
    # ----------------------------------------------------------------------
    def print_angles_table(self, angles_all):
        print(f"{'Idx':>5}", end="")
        for k in range(len(angles_all)):
            print(f"{f'Loop {k + 1}':>12}", end="")
        print()
        for i in range(len(angles_all[0])):
            print(f"{i:5}", end="")
            for loop in angles_all:
                print(f"{loop[i]:12.3f}", end="")
            print()

    def print_cumulative_angles_table(self, angles_all):
        cumulative = [angles_all[0].copy()]
        for k in range(1, len(angles_all)):
            prev_max = float(np.max(cumulative[-1]))
            cumulative.append(angles_all[k] + np.ceil(prev_max / 360.0) * 360.0)

        print(f"{'Idx':>5}", end="")
        for k in range(len(cumulative)):
            print(f"{f'Loop {k + 1}':>15}", end="")
        print()
        for i in range(len(cumulative[0])):
            print(f"{i:5}", end="")
            for loop in cumulative:
                print(f"{loop[i]:15.3f}", end="")
            print()

    def plot_interlaced_circles(self, angles_all):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)

        for k, angles in enumerate(angles_all):
            r = np.full_like(angles, 1.0 - k * 0.15, dtype=float)
            ax.plot(np.deg2rad(angles), r, "o-", label=f"Loop {k + 1}")

        ax.set_rticks([])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.legend()
        ax.set_title("Golden Angle – Interlaced (TIMBIR-style)")
        plt.show()

    # ----------------------------------------------------------------------
    #   EQUALLY SPACED – K-TURN
    # ----------------------------------------------------------------------
    def generate_interlaced_kturns(self, delta_theta=None):

        if delta_theta is not None:
            delta_theta = float(delta_theta)
        else:
            delta_theta = (self.rotation_stop - self.rotation_start) / (self.num_angles - 1)

        self.rotation_step = float(delta_theta)

        base = self.rotation_start + np.arange(self.num_angles, dtype=float) * self.rotation_step

        angles_all = []
        for k in range(self.K_interlace):
            angles_all.append(base + 360.0 * k)

        theta_unwrapped_acq = np.concatenate(angles_all).astype(float)

        # self.theta_interlaced = angoli interlacciati in ordine di acquisizione
        # (qui è già continuo; NON facciamo mod perché perdi informazione sui giri)
        self.theta_interlaced = theta_unwrapped_acq.astype(float)

        # self.theta_interlaced_unwrapped = angoli interlacciati unwrapped in ordine di acquisizione
        self.theta_interlaced_unwrapped = theta_unwrapped_acq.astype(float)

        # self.theta_monotonic = tutti gli angoli acquisiti in ordine crescente (lista da mandare al PSO)
        self.theta_monotonic = np.sort(self.theta_interlaced_unwrapped).astype(float)

        if self.K_interlace > 1:
            self.rotation_stop = float(theta_unwrapped_acq[-1])

        return angles_all

    def plot_equally_loops_polar_kturns(self):
        theta_unwrapped = np.asarray(self.theta_interlaced_unwrapped, dtype=float)
        theta_mod = np.mod(theta_unwrapped, 360.0)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"K-Turn (N={self.num_angles}, K={self.K_interlace})\n"
            "Each loop on its own circle (mod 360)",
            va="bottom",
            fontsize=13
        )

        for k in range(self.K_interlace):
            start = k * self.num_angles
            stop = (k + 1) * self.num_angles
            theta_k = theta_mod[start:stop]
            radii = np.full_like(theta_k, 1.0 - k * 0.15, dtype=float)

            ax.plot(np.deg2rad(theta_k), radii, "-o", lw=1.2, ms=5, alpha=0.85)

            for i, ang in enumerate(theta_k):
                ax.text(np.deg2rad(ang), radii[i] + 0.03, str(k + 1), ha="center", va="bottom", fontsize=8)

        ax.set_rticks([])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.show()

    def print_angles_table_kturns(self, angles_all):
        print(f"{'Idx':>5}", end="")
        for k in range(len(angles_all)):
            print(f"{f'Loop {k + 1} K-Turn':>15}", end="")
        print()
        for i in range(len(angles_all[0])):
            print(f"{i:5}", end="")
            for loop in angles_all:
                print(f"{loop[i]:15.3f}", end="")
            print()

    def print_cumulative_angles_table_kturns(self, angles_all):
        cumulative = [angles_all[0].copy()]
        for k in range(1, len(angles_all)):
            prev_max = float(np.max(cumulative[-1]))
            cumulative.append(angles_all[k] + np.ceil(prev_max / 360.0) * 360.0)

        print(f"{'Idx':>5}", end="")
        for k in range(len(cumulative)):
            print(f"{f'Loop {k + 1} K-Turn':>18}", end="")
        print()
        for i in range(len(cumulative[0])):
            print(f"{i:5}", end="")
            for loop in cumulative:
                print(f"{loop[i]:18.3f}", end="")
            print()

    # ----------------------------------------------------------------------
    #   EQUALLY SPACED multi-turn acquisition (TIMBIR-like)
    # ----------------------------------------------------------------------
    def generate_interlaced_multiturns(self, delta_theta=None):
        """different offset from kturns """

        N = self.num_angles
        K = self.K_interlace

        if delta_theta is not None:
            delta_theta = float(delta_theta)
        else:
            delta_theta = (self.rotation_stop - self.rotation_start) / (N - 1)

        self.rotation_step = float(delta_theta)

        n = np.arange(N, dtype=float)
        angles_all = []

        for k in range(K):
            theta_n = self.rotation_start + (n + k / K) * self.rotation_step
            angles_all.append(theta_n)

        theta_unwrapped_acq = np.concatenate(angles_all).astype(float)

        # self.theta_interlaced = angoli interlacciati in ordine di acquisizione
        self.theta_interlaced = theta_unwrapped_acq.astype(float)

        # self.theta_interlaced_unwrapped = angoli interlacciati unwrapped in ordine di acquisizione
        self.theta_interlaced_unwrapped = theta_unwrapped_acq.astype(float)

        # self.theta_monotonic = tutti gli angoli acquisiti in ordine crescente (lista da mandare al PSO)
        self.theta_monotonic = np.sort(self.theta_interlaced_unwrapped).astype(float)

        if self.K_interlace > 1:
            self.rotation_stop = float(theta_unwrapped_acq[-1])

        return angles_all

    def plot_equally_loops_polar_multiturns(self):
        theta_unwrapped = np.asarray(self.theta_interlaced_unwrapped, dtype=float)
        theta_mod = np.mod(theta_unwrapped, 360.0)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"Multi-Turn (N={self.num_angles}, K={self.K_interlace})\n"
            "Each loop on its own circle (mod 360)",
            va="bottom",
            fontsize=13
        )

        for k in range(self.K_interlace):
            start = k * self.num_angles
            stop = (k + 1) * self.num_angles
            theta_k = theta_mod[start:stop]
            radii = np.full_like(theta_k, 1.0 - k * 0.15, dtype=float)

            ax.plot(np.deg2rad(theta_k), radii, "-o", lw=1.2, ms=5, alpha=0.85)

            for i, ang in enumerate(theta_k):
                ax.text(np.deg2rad(ang), radii[i] + 0.03, str(k + 1), ha="center", va="bottom", fontsize=8)

        ax.set_rticks([])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.show()

    def print_angles_table_multiturns(self, angles_all):
        print(f"{'Idx':>5}", end="")
        for k in range(len(angles_all)):
            print(f"{f'Loop {k + 1} Multi-Turn':>18}", end="")
        print()
        for i in range(len(angles_all[0])):
            print(f"{i:5}", end="")
            for loop in angles_all:
                print(f"{loop[i]:18.3f}", end="")
            print()

    def print_cumulative_angles_table_multiturns(self, angles_all):
        cumulative = [angles_all[0].copy()]
        for k in range(1, len(angles_all)):
            prev_max = float(np.max(cumulative[-1]))
            cumulative.append(angles_all[k] + np.ceil(prev_max / 360.0) * 360.0)

        print(f"{'Idx':>5}", end="")
        for k in range(len(cumulative)):
            print(f"{f'Loop {k + 1} Multi-Turn':>20}", end="")
        print()
        for i in range(len(cumulative[0])):
            print(f"{i:5}", end="")
            for loop in cumulative:
                print(f"{loop[i]:20.3f}", end="")
            print()


    # =========================================================
    # VAN DER CORPUT INTERLACED – K-TURN
    # =========================================================
    def generate_interlaced_corput(self, delta_theta=None):

        if delta_theta is not None:
            delta_theta = float(delta_theta)
        else:
            delta_theta = (self.rotation_stop - self.rotation_start) / (self.num_angles - 1)

        self.rotation_step = delta_theta

        base = self.rotation_start + np.arange(self.num_angles) * delta_theta

        K = self.K_interlace
        bitsK = int(np.ceil(np.log2(K)))
        MK = 1 << bitsK

        p_corput = np.array([self.bit_reverse(i, bitsK) for i in range(MK)])
        p_corput = p_corput[p_corput < K]
        assert len(p_corput) == K

        offsets = (p_corput / K) * delta_theta

        angles_all = []

        bits = int(np.ceil(np.log2(self.num_angles)))
        M = 1 << bits

        indices = np.array([self.bit_reverse(i, bits) for i in range(M)])
        indices = indices[indices < self.num_angles]

        for k in range(K):
            offset = offsets[k]
            loop_angles = base[indices] + offset

            loop_angles_mod = np.mod(loop_angles - self.rotation_start, 360.0) + self.rotation_start
            loop_angles_unwrapped = loop_angles_mod + 360.0 * k

            angles_all.append(loop_angles_unwrapped)

        theta_unwrapped_unsorted = np.concatenate(angles_all)
        theta_unsorted = np.mod(theta_unwrapped_unsorted - self.rotation_start, 360.0) + self.rotation_start

        theta_unwrapped = np.sort(theta_unwrapped_unsorted)
        theta = np.sort(theta_unsorted)

        self.theta_interlaced = np.array(theta)
        self.theta_interlaced_unwrapped = np.array(theta_unwrapped)
        self.theta_monotonic = np.sort(self.theta_interlaced_unwrapped)

        if self.K_interlace > 1:
            self.rotation_stop = float(theta_unwrapped[-1])

        return angles_all

    def plot_equally_loops_polar_corput(self):

        theta_unwrapped = self.theta_interlaced_unwrapped
        theta_mod = np.mod(theta_unwrapped, 360.0)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"Interlaced Van der Corput K-Turn (N={self.num_angles}, K={self.K_interlace})",
            va="bottom",
            fontsize=13
        )

        for k in range(self.K_interlace):
            start = k * self.num_angles
            stop = (k + 1) * self.num_angles

            theta_k = theta_mod[start:stop]
            radii = np.full_like(theta_k, 1 - k * 0.15)

            ax.plot(np.deg2rad(theta_k), radii, "-o", lw=1.2, ms=5, alpha=0.85)

            for i, ang in enumerate(theta_k):
                ax.text(np.deg2rad(ang), radii[i] + 0.03, str(k + 1), ha="center", va="bottom", fontsize=8)

        ax.set_rticks([])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.show()

    def print_angles_table_corput(self, angles_all):
        print(f"{'Idx':>5}", end="")
        for k in range(len(angles_all)):
            print(f"{f'Loop {k + 1} K-Turn':>15}", end="")
        print()

        for i in range(len(angles_all[0])):
            print(f"{i:5}", end="")
            for loop in angles_all:
                print(f"{loop[i]:15.3f}", end="")
            print()

    def print_cumulative_angles_table_corput(self, angles_all):
        cumulative = [angles_all[0].copy()]
        for k in range(1, len(angles_all)):
            prev_max = cumulative[-1].max()
            cumulative.append(angles_all[k] + np.ceil(prev_max / 360) * 360)

        print(f"{'Idx':>5}", end="")
        for k in range(len(cumulative)):
            print(f"{f'Loop {k + 1} K-Turn':>18}", end="")
        print()

        for i in range(len(cumulative[0])):
            print(f"{i:5}", end="")
            for loop in cumulative:
                print(f"{loop[i]:18.3f}", end="")
            print()

    def plot_live_corput(self):

        theta_unwrapped = self.theta_interlaced_unwrapped
        theta_mod = np.mod(theta_unwrapped, 360.0)

        fig, ax = plt.subplots(figsize=(9, 5))

        n_total = len(theta_mod)
        indices = np.arange(n_total)

        for k in range(self.K_interlace):
            start = k * self.num_angles
            stop = (k + 1) * self.num_angles

            ax.scatter(indices[start:stop], theta_mod[start:stop], s=18, alpha=0.85, label=f"Loop {k+1}")
            ax.plot(indices[start:stop], theta_mod[start:stop], lw=0.6, alpha=0.4)

        ax.set_title(
            f"Live Acquisition Order – Van der Corput (N={self.num_angles}, K={self.K_interlace})",
            fontsize=13
        )

        ax.set_xlabel("Acquisition index")
        ax.set_ylabel("Angle [deg]")
        ax.set_ylim(0, 360)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()
    # ----------------------------------------------------------------------
    #   plot delta (Δθ consecutivi, ordine sort)
    # ----------------------------------------------------------------------
    def plotdelta(self, metodo=""):
        theta = np.asarray(self.theta_monotonic, dtype=float)
        dtheta = np.diff(theta)

        print(f"\n--- Δθ ({metodo}) ---")
        for i, d in enumerate(dtheta):
            print(f"{i:4d} -> {i+1:4d}: {d:9.3f} deg")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.arange(1, len(theta)), dtheta, "o")
        ax.set_title(f"Δθ tra angoli consecutivi sort – {metodo}")
        ax.set_xlabel("Indice acquisizione")
        ax.set_ylabel("Δθ [deg]")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # =========================================================
    # FUNZIONI (motion + PSO)
    # =========================================================
    def compute_senses(self):
        encoder_dir = 1 if self.PSOCountsPerRotation > 0 else -1
        motor_dir = 1 if self.RotationDirection == 0 else -1
        user_dir = 1 if self.rotation_stop > self.rotation_start else -1
        return encoder_dir * motor_dir * user_dir, user_dir

    def compute_frame_time(self):
        return float(self.exposure + self.readout)

    def compute_positions_PSO(self):
        overall_sense, user_direction = self.compute_senses()
        encoder_multiply = self.PSOCountsPerRotation / 360.0

        raw_counts = self.rotation_step * encoder_multiply
        delta_counts = round(raw_counts)
        self.rotation_step = delta_counts / encoder_multiply

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

        self.theta_classic = self.rotation_start_new + np.arange(self.num_angles, dtype=float) * self.rotation_step

    # ----------------------------------------------------------------------
    # Modello taxi
    # ----------------------------------------------------------------------
    def simulate_taxi_motion(self, omega_target=10, dt=1e-4):
        theta_max = float(np.max(np.asarray(self.theta_monotonic, dtype=float)))

        accel = decel = float(omega_target) / float(self.RotationAccelTime)

        # accelerazione
        t_acc = np.arange(0, self.RotationAccelTime, dt)
        theta_acc = 0.5 * accel * t_acc ** 2
        theta_acc_end = float(theta_acc[-1]) if len(theta_acc) > 0 else 0.0

        # plateau
        theta_flat_len = theta_max - 2 * theta_acc_end
        if theta_flat_len < 0:
            raise ValueError("Profilo di moto non realizzabile (theta_max troppo piccolo rispetto accel/decel)")

        t_flat = np.arange(0, theta_flat_len / omega_target, dt) if omega_target > 0 else np.array([0.0])
        theta_flat = theta_acc_end + omega_target * t_flat

        # decelerazione
        t_dec = np.arange(0, self.RotationAccelTime, dt)
        last_flat = float(theta_flat[-1]) if len(theta_flat) > 0 else theta_acc_end
        theta_dec = last_flat + omega_target * t_dec - 0.5 * decel * t_dec ** 2

        self.theta_vec = np.concatenate([theta_acc, theta_flat, theta_dec]).astype(float)
        self.t_vec = np.concatenate([
            t_acc,
            (t_acc[-1] if len(t_acc) > 0 else 0.0) + t_flat,
            (t_acc[-1] if len(t_acc) > 0 else 0.0) + (t_flat[-1] if len(t_flat) > 0 else 0.0) + t_dec
        ]).astype(float)

    def compute_real_motion(self):
        theta_target = np.asarray(self.theta_monotonic, dtype=float)

        self.t_real = np.interp(theta_target, self.theta_vec, self.t_vec).astype(float)
        self.theta_real = np.interp(self.t_real, self.t_vec, self.theta_vec).astype(float)

    def convert_angles_to_counts(self):
        pulses_per_degree = self.PSOCountsPerRotation / 360.0

        theta_target = np.asarray(self.theta_monotonic, dtype=float)  # PSO input
        self.PSOCountsIdeal = np.round(theta_target * pulses_per_degree).astype(int)

        if np.any(np.diff(self.PSOCountsIdeal) <= 0):
            print("WARNING: counts non strettamente crescenti (duplicati/inversioni).")

        self.PSOCountsTaxiCorrected = np.round(np.asarray(self.theta_real, dtype=float) * pulses_per_degree).astype(int)
        self.PSOCountsFinal = self.PSOCountsTaxiCorrected.copy()

        # diagnostica: stampa errori sia mod360 sia unwrapped(acq)
        print("\n********************* MOD 360 diagnostica *********************")
        theta_mod = np.mod(np.asarray(self.theta_interlaced, dtype=float), 360.0)
        pulse_counts_mod = np.round(theta_mod / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_mod = pulse_counts_mod / pulses_per_degree
        err_mod = actual_mod - theta_mod
        for a, p, act, err in zip(theta_mod, pulse_counts_mod, actual_mod, err_mod):
            print(f"Target(mod): {a:8.3f} deg | Pulse: {p:6d} | Actual: {act:9.6f} deg | Error: {err:+.6f} deg")

        print("\n********************* unwrapped angles (ordine acquisizione) *********************")
        theta_unw = np.asarray(self.theta_interlaced_unwrapped, dtype=float)
        pulse_counts_unw = np.round(theta_unw / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_unw = pulse_counts_unw / pulses_per_degree
        err_unw = actual_unw - theta_unw
        for a, p, act, err in zip(theta_unw, pulse_counts_unw, actual_unw, err_unw):
            print(f"Target(unw): {a:8.3f} deg | Pulse: {p:6d} | Actual: {act:9.6f} deg | Error: {err:+.6f} deg")

        print("\n********************* sort angles (ordine monotono) *********************")
        theta_unw = np.asarray(self.theta_monotonic, dtype=float)
        pulse_counts_unw = np.round(theta_unw / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_unw = pulse_counts_unw / pulses_per_degree
        err_unw = actual_unw - theta_unw
        for a, p, act, err in zip(theta_unw, pulse_counts_unw, actual_unw, err_unw):
            print(f"Target(unw): {a:8.3f} deg | Pulse: {p:6d} | Actual: {act:9.6f} deg | Error: {err:+.6f} deg")

    # ----------------------------------------------------------------------
    # Plot comparazioni counts
    # ----------------------------------------------------------------------
    def plot_all_comparisons(self):
        ideal = np.asarray(self.PSOCountsIdeal, dtype=float)
        real = np.asarray(self.PSOCountsTaxiCorrected, dtype=float)
        final = np.asarray(self.PSOCountsFinal, dtype=float)

        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        axs[0].plot(ideal, ideal, "o--", alpha=0.6, label="Ideal")
        axs[0].plot(ideal, real, "o-", alpha=0.9, label="Real (Taxi)")
        axs[0].set_title("Ideale vs Reale")
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(ideal, ideal, "o--", alpha=0.6, label="Ideal")
        axs[1].plot(ideal, final, "o-", alpha=0.9, label="Final FPGA")
        axs[1].set_title("Ideale vs FPGA")
        axs[1].grid()
        axs[1].legend()

        axs[2].plot(real, real, "o--", alpha=0.6, label="Real")
        axs[2].plot(real, final, "o-", alpha=0.9, label="Final FPGA")
        axs[2].set_title("Reale vs FPGA")
        axs[2].grid()
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------------
    # Plot angolo vs impulsi (mod e unwrapped)
    # ----------------------------------------------------------------------
    def plot(self):
        pulses_per_degree = self.PSOCountsPerRotation / 360.0

        x1 = np.mod(np.asarray(self.theta_interlaced, dtype=float), 360.0)
        y1 = np.round(x1 * pulses_per_degree).astype(int)

        x2 = np.asarray(self.theta_interlaced_unwrapped, dtype=float)
        y2 = np.round(x2 * pulses_per_degree).astype(int)

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        axs[0].plot(x1, y1, "o-")
        axs[0].set_title("MOD 360: Angolo vs Impulsi")
        axs[0].set_xlabel("Angolo [deg] (mod 360)")
        axs[0].set_ylabel("Pulses Encoder")
        axs[0].grid(True)

        axs[1].plot(x2, y2, "s-")
        axs[1].set_title("UNWRAPPED: Angles vs Pulses")
        axs[1].set_xlabel("Unwrapped Angles  [deg]")
        axs[1].set_ylabel("Pulses Encoder")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------------
    # Export Excel
    # ----------------------------------------------------------------------
    def export_to_excel(self, filename="risultati.xlsx"):
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("Manca pandas. Installa con: pip install pandas") from e

        try:
            import openpyxl  # noqa: F401
        except ImportError as e:
            raise ImportError("Manca openpyxl. Installa con: pip install openpyxl") from e

        pulses_per_degree = self.PSOCountsPerRotation / 360.0

        theta_mod = np.mod(np.asarray(self.theta_interlaced, dtype=float), 360.0)
        pulse_counts_mod = np.round(theta_mod / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_mod = pulse_counts_mod / pulses_per_degree
        err_mod = actual_mod - theta_mod

        df_mod = pd.DataFrame({
            "target_deg_mod360": theta_mod,
            "pulse": pulse_counts_mod,
            "actual_deg": actual_mod,
            "error_deg": err_mod
        })

        theta_unw = np.asarray(self.theta_interlaced_unwrapped, dtype=float)
        pulse_counts_unw = np.round(theta_unw / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_unw = pulse_counts_unw / pulses_per_degree
        err_unw = actual_unw - theta_unw

        df_unw = pd.DataFrame({
            "target_deg_unwrapped_acq": theta_unw,
            "pulse": pulse_counts_unw,
            "actual_deg": actual_unw,
            "error_deg": err_unw
        })

        theta_pso = np.asarray(self.theta_monotonic, dtype=float)
        df_counts = pd.DataFrame({
            "theta_monotonic_deg": theta_pso,
            "counts_ideal": np.asarray(self.PSOCountsIdeal, dtype=int),
            "counts_taxi": np.asarray(self.PSOCountsTaxiCorrected, dtype=int),
            "counts_final": np.asarray(self.PSOCountsFinal, dtype=int)
        })

        df_delta = pd.DataFrame({
            "i": np.arange(len(theta_pso) - 1),
            "theta_i": theta_pso[:-1],
            "theta_ip1": theta_pso[1:],
            "delta_theta": np.diff(theta_pso)
        })

        with pd.ExcelWriter(filename, engine="openpyxl") as w:
            df_mod.to_excel(w, sheet_name="MOD360", index=False)
            df_unw.to_excel(w, sheet_name="UNWRAPPED_ACQ", index=False)
            df_counts.to_excel(w, sheet_name="COUNTS_PSO", index=False)
            df_delta.to_excel(w, sheet_name="DELTA_THETA_PSO", index=False)

        print(f"Creato: {filename}")


# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run interlaced scan simulation.")
    parser.add_argument("--num_angles", type=int, default=32, help="Number of angles (default: 32)")
    parser.add_argument("--K_interlace", type=int, default=4, help="Interlace factor K (default: 4)")
    parser.add_argument(
        "--mode",
        choices=["timbir", "multitimbir", "golden", "kturns", "multiturns", "corput"],
        default="timbir"
    )
    parser.add_argument("--PSOCountsPerRotation", type=int, default=20000, help="PSO counts per rotation (default: 20000)")

    args = parser.parse_args()

    scan = InterlacedScan(
        num_angles=args.num_angles,
        K_interlace=args.K_interlace,
        PSOCountsPerRotation=args.PSOCountsPerRotation
    )

    if args.mode == "timbir":
        scan.generate_interlaced_timbir()
        scan.plotdelta("timbir")

    elif args.mode == "multitimbir":
        scan.generate_interlaced_multitimbir()
        scan.plotdelta("multitimbir")

    elif args.mode == "golden":
        angles_all = scan.generate_interlaced_goldenangle()
        scan.print_angles_table(angles_all)
        scan.print_cumulative_angles_table(angles_all)
        scan.plot_interlaced_circles(angles_all)
        scan.plotdelta("golden")

    elif args.mode == "kturns":
        angles_all = scan.generate_interlaced_kturns()
        scan.plot_equally_loops_polar_kturns()
        scan.print_cumulative_angles_table_kturns(angles_all)
        scan.print_angles_table_kturns(angles_all)
        scan.plotdelta("kturns")

    elif args.mode == "multiturns":
        angles_all = scan.generate_interlaced_multiturns()
        scan.plot_equally_loops_polar_multiturns()
        scan.print_cumulative_angles_table_multiturns(angles_all)
        scan.print_angles_table_multiturns(angles_all)
        scan.plotdelta("multiturns")

    elif args.mode == "corput":
        angles_all = scan.generate_interlaced_corput()
        scan.plot_equally_loops_polar_corput()
        scan.print_cumulative_angles_table_corput(angles_all)
        scan.print_angles_table_corput(angles_all)
        scan.plot_live_corput()
        scan.plotdelta("corput")

    # motion / counts (usa theta_monotonic come stringa da mandare al PSO)
    scan.compute_positions_PSO()
    scan.simulate_taxi_motion()
    scan.compute_real_motion()
    scan.convert_angles_to_counts()

    scan.plot_all_comparisons()
    scan.plot()

    filename = f"risultati_{args.mode}_N{args.num_angles}_K{args.K_interlace}_PSO{args.PSOCountsPerRotation}.xlsx"
    scan.export_to_excel(filename)


if __name__ == "__main__":
    main()


"""
 python Tomoscan_pso_interlaced.py --mode timbir     --num_angles 32 --K_interlace 4 --PSOCountsPerRotation 20000
 
 python Tomoscan_pso_interlaced.py --mode multitimbir --num_angles 32 --K_interlace 4 --PSOCountsPerRotation 20000
 
 python Tomoscan_pso_interlaced.py --mode golden     --num_angles 32 --K_interlace 4 --PSOCountsPerRotation 20000
 
 python Tomoscan_pso_interlaced.py --mode kturns     --num_angles 32 --K_interlace 4 --PSOCountsPerRotation 20000
 
 python Tomoscan_pso_interlaced.py --mode multiturns --num_angles 32 --K_interlace 4 --PSOCountsPerRotation 20000
 
 python Tomoscan_pso_interlaced.py --mode corput     --num_angles 32 --K_interlace 4 --PSOCountsPerRotation 20000
"""
