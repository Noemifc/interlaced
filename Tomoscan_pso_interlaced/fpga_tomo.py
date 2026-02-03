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
        RotationDirection=0,
        RotationAccelTime=0.15,
        exposure=0.01,
        readout=0.01,
        readout_margin=1,
        K_interlace=4
    ):

        # Parametri di scansione
        self.rotation_start = rotation_start
        self.rotation_stop = rotation_stop
        self.num_angles = num_angles
        self.K_interlace = K_interlace

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

        # --- Nuove liste  ---
        self.theta_fpga = None
        self.theta_acq_mod = None
        self.theta_sorted_mod = None
        self.theta_acq_unwrapped_monotono = None

        # Alias
        self.theta_interlaced = None
        self.theta_interlaced_unwrapped = None

        # Per plot che usano i loop
        self.angles_all = None


    # =========================================================
    # UTILITY
    # =========================================================
    def bit_reverse(self, n, bits):
        return int(f"{n:0{bits}b}"[::-1], 2)


    # =========================================================
    # MODE
    # =========================================================

    # ----------------------------------------------------------------------
    #   TIMBIR
    # ----------------------------------------------------------------------
    def generate_interlaced_timbir(self):

        bits = int(np.log2(self.K_interlace))
        theta = []
        group_indices = []

        assert (self.K_interlace & (self.K_interlace - 1)) == 0

        for n in range(self.num_angles):
            group = (n * self.K_interlace // self.num_angles) % self.K_interlace
            group_br = self.bit_reverse(group, bits)
            idx = n * self.K_interlace + group_br
            angle_deg = (idx % self.num_angles) * 360.0 / self.num_angles
            theta.append(angle_deg)
            group_indices.append(group)

        # -----------------------------
        #
        # -----------------------------
        theta_acq_mod_list = theta  # ordine acquisizione (MOD360)

        theta_acq_mod = np.mod(np.array(theta_acq_mod_list, dtype=float), 360.0)
        theta_sorted_mod = np.sort(theta_acq_mod)

        theta_unw = np.empty_like(theta_acq_mod, dtype=float)
        if len(theta_acq_mod) > 0:
            offset = 0.0
            theta_unw[0] = theta_acq_mod[0]
            for i in range(1, len(theta_acq_mod)):
                if theta_acq_mod[i] <= theta_acq_mod[i - 1]:
                    offset += 360.0
                theta_unw[i] = theta_acq_mod[i] + offset

        self.theta_acq_mod = theta_acq_mod
        self.theta_sorted_mod = theta_sorted_mod
        self.theta_acq_unwrapped_monotono = theta_unw
        self.theta_fpga = theta_unw.copy()

        # Alias legacy
        self.theta_interlaced = self.theta_sorted_mod
        self.theta_interlaced_unwrapped = self.theta_fpga

        # -----------------------------
        # Plot polare (come prima)
        # -----------------------------
        group_indices = np.array(group_indices)
        radii = 1 - group_indices * 0.15

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"TIMBIR Interlaced Acquisition (N={self.num_angles} - K={self.K_interlace})\n"
            "Each loop on its own circle",
            va="bottom",
            fontsize=13
        )

        ax.plot(np.deg2rad(theta), radii, "-o", lw=1.2, ms=5, alpha=0.8, color="tab:blue")

        for i in range(self.num_angles):
            ax.text(
                np.deg2rad(theta[i]),
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

        bits = int(np.log2(self.K_interlace))
        theta = []
        group_indices = []

        assert (self.K_interlace & (self.K_interlace - 1)) == 0

        # i = indice nel loop da 0 a num_angles
        # g = indice temporale nel loop da 0 a K loops
        for i in range(self.num_angles):
            for g in range(self.K_interlace):
                loop = self.bit_reverse(g, bits)                 # ordine temporale (bit-reversal)
                idx = i * self.K_interlace + loop                # 0..N*K-1
                angle_deg = idx * 360.0 / (self.num_angles * self.K_interlace)
                theta.append(angle_deg)
                group_indices.append(loop)

        # -----------------------------
        # NEW: 4 liste standard
        # -----------------------------
        theta_acq_mod_list = theta

        theta_acq_mod = np.mod(np.array(theta_acq_mod_list, dtype=float), 360.0)
        theta_sorted_mod = np.sort(theta_acq_mod)

        theta_unw = np.empty_like(theta_acq_mod, dtype=float)
        if len(theta_acq_mod) > 0:
            offset = 0.0
            theta_unw[0] = theta_acq_mod[0]
            for i in range(1, len(theta_acq_mod)):
                if theta_acq_mod[i] <= theta_acq_mod[i - 1]:
                    offset += 360.0
                theta_unw[i] = theta_acq_mod[i] + offset

        self.theta_acq_mod = theta_acq_mod
        self.theta_sorted_mod = theta_sorted_mod
        self.theta_acq_unwrapped_monotono = theta_unw
        self.theta_fpga = theta_unw.copy()

        # Alias legacy
        self.theta_interlaced = self.theta_sorted_mod
        self.theta_interlaced_unwrapped = self.theta_fpga

        # -----------------------------
        # Plot polare (come prima)
        # -----------------------------
        theta = np.array(theta, dtype=float)
        group_indices = np.array(group_indices, dtype=int)

        step = 0.8 / max(self.K_interlace - 1, 1)
        radii = 1.0 - group_indices * step

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"Multi-TIMBIR: N={self.num_angles} per loop, K={self.K_interlace} → totale N·K={self.num_angles*self.K_interlace} angoli\n"
            "Loop su cerchi separati con ordine loop = bit-reversal",
            va="bottom",
            fontsize=12
        )

        ax.plot(np.deg2rad(theta), radii, "o", lw=1.2, ms=5, alpha=0.8)

        for ang, r, lp in zip(theta, radii, group_indices):
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

        golden_angle = 360 * (3 - np.sqrt(5)) / 2
        phi_inv = (np.sqrt(5) - 1) / 2

        # base in ordine di acquisizione (NON sort)
        base_acq = np.array([(self.rotation_start + i * golden_angle) % 360
                             for i in range(self.num_angles)], dtype=float)

        loops_acq = [base_acq]
        angles_all_sorted = [np.sort(base_acq)]  # per tabelle/plot

        for k in range(1, self.K_interlace):
            offset = (k / (self.num_angles + 1)) * 360 * phi_inv
            loop_acq = (base_acq + offset) % 360
            loops_acq.append(loop_acq)
            angles_all_sorted.append(np.sort(loop_acq))

        # ordine temporale dei loop: bit-reversal se K è potenza di 2
        if (self.K_interlace & (self.K_interlace - 1)) == 0:
            bits = int(np.log2(self.K_interlace))
            loop_order = [self.bit_reverse(g, bits) for g in range(self.K_interlace)]
        else:
            loop_order = list(range(self.K_interlace))

        # acquisizione interlacciata i-major
        theta_acq_mod_list = []
        for i in range(self.num_angles):
            for g in loop_order:
                theta_acq_mod_list.append(loops_acq[g][i])

        # -----------------------------
        # NEW: 4 liste standard
        # -----------------------------
        theta_acq_mod = np.mod(np.array(theta_acq_mod_list, dtype=float), 360.0)
        theta_sorted_mod = np.sort(theta_acq_mod)

        theta_unw = np.empty_like(theta_acq_mod, dtype=float)
        if len(theta_acq_mod) > 0:
            offset = 0.0
            theta_unw[0] = theta_acq_mod[0]
            for i in range(1, len(theta_acq_mod)):
                if theta_acq_mod[i] <= theta_acq_mod[i - 1]:
                    offset += 360.0
                theta_unw[i] = theta_acq_mod[i] + offset

        self.theta_acq_mod = theta_acq_mod
        self.theta_sorted_mod = theta_sorted_mod
        self.theta_acq_unwrapped_monotono = theta_unw
        self.theta_fpga = theta_unw.copy()

        # Alias legacy
        self.theta_interlaced = self.theta_sorted_mod
        self.theta_interlaced_unwrapped = self.theta_fpga

        # salva per plot loop
        self.angles_all = angles_all_sorted

        return angles_all_sorted


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
            prev_max = cumulative[-1].max()
            cumulative.append(angles_all[k] + np.ceil(prev_max / 360) * 360)

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
            r = np.full_like(angles, 1 - k * 0.15)
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

        self.rotation_step = delta_theta

        base = self.rotation_start + np.arange(self.num_angles) * delta_theta

        angles_all = []
        for k in range(self.K_interlace):
            angles_all.append(base + k * 360.0)

        theta_fpga = np.concatenate(angles_all)  # già monotono crescente

        # -----------------------------
        # NEW: 4 liste standard
        # -----------------------------
        self.theta_fpga = np.array(theta_fpga, dtype=float)
        self.theta_acq_unwrapped_monotono = self.theta_fpga.copy()
        self.theta_acq_mod = np.mod(self.theta_fpga, 360.0)
        self.theta_sorted_mod = np.sort(self.theta_acq_mod)

        # Alias legacy
        self.theta_interlaced = self.theta_sorted_mod
        self.theta_interlaced_unwrapped = self.theta_fpga

        self.angles_all = angles_all

        if self.K_interlace > 1:
            self.rotation_stop = float(self.theta_fpga[-1])

        return angles_all

    def plot_equally_loops_polar_kturns(self):

        if self.angles_all is None:
            raise RuntimeError("Prima chiama generate_interlaced_kturns().")

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)

        ax.set_title(
            f"Equally Spaced Acquisition (N={self.num_angles}, K={self.K_interlace})\n"
            "Each loop on its own circle",
            va="bottom",
            fontsize=13
        )

        for k, loop_angles_unw in enumerate(self.angles_all):
            theta_k = np.mod(np.array(loop_angles_unw, dtype=float), 360.0)
            radii = np.full_like(theta_k, 1 - k * 0.15)

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


    # ----------------------------------------------------------------------
    #   EQUALLY SPACED multi-turn acquisition (TIMBIR-like)
    # ----------------------------------------------------------------------
    def generate_interlaced_multiturns(self, delta_theta=None):

        N = self.num_angles
        K = self.K_interlace

        if delta_theta is not None:
            delta_theta = float(delta_theta)
        else:
            delta_theta = (self.rotation_stop - self.rotation_start) / (N - 1)

        self.rotation_step = delta_theta

        n = np.arange(N)
        angles_all = []

        for k in range(K):
            theta_n = self.rotation_start + (n + k / K) * delta_theta  # già unwrapped
            angles_all.append(theta_n)

        # ordine acquisizione TIMBIR-like: i-major (interleaving)
        theta_fpga_list = []
        for i in range(N):
            for k in range(K):
                theta_fpga_list.append(angles_all[k][i])

        theta_fpga = np.array(theta_fpga_list, dtype=float)  # monotono

        # -----------------------------
        # NEW: 4 liste standard
        # -----------------------------
        self.theta_fpga = theta_fpga
        self.theta_acq_unwrapped_monotono = self.theta_fpga.copy()
        self.theta_acq_mod = np.mod(self.theta_fpga, 360.0)
        self.theta_sorted_mod = np.sort(self.theta_acq_mod)

        # Alias legacy
        self.theta_interlaced = self.theta_sorted_mod
        self.theta_interlaced_unwrapped = self.theta_fpga

        self.angles_all = angles_all

        if self.K_interlace > 1:
            self.rotation_stop = float(self.theta_fpga[-1])

        return angles_all

    def plot_equally_loops_polar_multiturns(self):

        if self.angles_all is None:
            raise RuntimeError("Prima chiama generate_interlaced_multiturns().")

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)

        ax.set_title(
            f"Equally Spaced Multi-Turn (N={self.num_angles}, K={self.K_interlace})\n"
            "Each loop on its own circle",
            va="bottom",
            fontsize=13
        )

        for k, loop_angles_unw in enumerate(self.angles_all):
            theta_k = np.mod(np.array(loop_angles_unw, dtype=float), 360.0)
            radii = np.full_like(theta_k, 1 - k * 0.15)

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
            prev_max = cumulative[-1].max()
            cumulative.append(angles_all[k] + np.ceil(prev_max / 360) * 360)

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

        # ordine acquisizione "teorico": loop-major, dentro loop = bit-reversal indices
        theta_acq_mod_list = []
        for loop_unw in angles_all:
            theta_acq_mod_list.extend(np.mod(loop_unw, 360.0))

        # -----------------------------
        # NEW: 4 liste standard (unwrap monotono)
        # -----------------------------
        theta_acq_mod = np.mod(np.array(theta_acq_mod_list, dtype=float), 360.0)
        theta_sorted_mod = np.sort(theta_acq_mod)

        theta_unw = np.empty_like(theta_acq_mod, dtype=float)
        if len(theta_acq_mod) > 0:
            offset = 0.0
            theta_unw[0] = theta_acq_mod[0]
            for i in range(1, len(theta_acq_mod)):
                if theta_acq_mod[i] <= theta_acq_mod[i - 1]:
                    offset += 360.0
                theta_unw[i] = theta_acq_mod[i] + offset

        self.theta_acq_mod = theta_acq_mod
        self.theta_sorted_mod = theta_sorted_mod
        self.theta_acq_unwrapped_monotono = theta_unw
        self.theta_fpga = theta_unw.copy()

        # Alias legacy
        self.theta_interlaced = self.theta_sorted_mod
        self.theta_interlaced_unwrapped = self.theta_fpga

        self.angles_all = angles_all

        if self.K_interlace > 1 and len(self.theta_fpga) > 0:
            self.rotation_stop = float(self.theta_fpga[-1])

        return angles_all


    def plot_equally_loops_polar_corput(self):

        if self.angles_all is None:
            raise RuntimeError("Prima chiama generate_interlaced_corput().")

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"Interlaced Van der Corput K-Turn (N={self.num_angles}, K={self.K_interlace})",
            va="bottom",
            fontsize=13
        )

        for k, loop_unw in enumerate(self.angles_all):
            theta_k = np.mod(np.array(loop_unw, dtype=float), 360.0)
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

        # ordine acquisizione = self.theta_acq_mod
        theta_mod = np.array(self.theta_acq_mod, dtype=float)

        fig, ax = plt.subplots(figsize=(9, 5))

        n_total = len(theta_mod)
        indices = np.arange(n_total)

        # segmentazione per loop (assume loop-major: N angoli per loop)
        for k in range(self.K_interlace):
            start = k * self.num_angles
            stop = (k + 1) * self.num_angles
            if stop > n_total:
                break

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
    #   plot delta (Δθ consecutivi, ordine acquisizione)  <-- usa theta_fpga
    # ----------------------------------------------------------------------
    def plotdelta(self, metodo=""):

        theta = np.array(self.theta_fpga, dtype=float)
        dtheta = np.diff(theta)

        print(f"\n--- Δθ (acq order, theta_fpga) ({metodo}) ---")
        for i, d in enumerate(dtheta):
            print(f"{i:4d} -> {i+1:4d}: {d:9.3f} deg")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.arange(1, len(theta)), dtheta, "o")
        ax.set_title(f"Δθ tra angoli consecutivi (theta_fpga) – {metodo}")
        ax.set_xlabel("Indice acquisizione")
        ax.set_ylabel("Δθ [deg]")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    # ----------------------------------------------------------------------
    #   plot delta (Δθ su angoli ordinati)  <-- usa theta_sorted_mod
    # ----------------------------------------------------------------------
    def plotdelta_sort(self, metodo="", include_wrap=True):

        theta = np.array(self.theta_sorted_mod, dtype=float)

        if len(theta) < 2:
            print("Pochi angoli: niente Δθ da plottare.")
            return

        if include_wrap:
            theta_ext = np.append(theta, theta[0] + 360.0)
            dtheta = np.diff(theta_ext)
            x = np.arange(1, len(theta_ext))
        else:
            dtheta = np.diff(theta)
            x = np.arange(1, len(theta))

        print(f"\n--- Δθ (sorted MOD360) ({metodo}) ---")
        for i, d in enumerate(dtheta):
            print(f"{i:4d} -> {i+1:4d}: {d:9.3f} deg")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(x, dtheta, "o")
        ax.set_title(f"Δθ su lista ordinata (theta_sorted_mod) – {metodo}")
        ax.set_xlabel("Indice nella lista ordinata")
        ax.set_ylabel("Δθ [deg]")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    # ----------------------------------------------------------------------
    #           FUNZIONI
    # ----------------------------------------------------------------------
    def compute_senses(self):

        encoder_dir = 1 if self.PSOCountsPerRotation > 0 else -1
        motor_dir = 1 if self.RotationDirection == 0 else -1
        user_dir = 1 if self.rotation_stop > self.rotation_start else -1
        return encoder_dir * motor_dir * user_dir, user_dir

    def compute_frame_time(self):
        return self.exposure + self.readout

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

        self.theta_classic = self.rotation_start_new + np.arange(self.num_angles) * self.rotation_step


    # ----------------------------------------------------------------------
    # Modello taxi
    # ----------------------------------------------------------------------
    def simulate_taxi_motion(self, omega_target=10, dt=1e-4):

        theta_max = float(np.max(self.theta_fpga))

        accel = decel = omega_target / self.RotationAccelTime

        # accelerazione
        t_acc = np.arange(0, self.RotationAccelTime, dt)
        theta_acc = 0.5 * accel * t_acc ** 2
        theta_acc_end = theta_acc[-1]

        # plateau
        theta_flat_len = theta_max - 2 * theta_acc_end
        if theta_flat_len < 0:
            raise ValueError("Profilo di moto non realizzabile")

        t_flat = np.arange(0, theta_flat_len / omega_target, dt)
        theta_flat = theta_acc_end + omega_target * t_flat

        # decelerazione
        t_dec = np.arange(0, self.RotationAccelTime, dt)
        theta_dec = theta_flat[-1] + omega_target * t_dec - 0.5 * decel * t_dec ** 2

        self.theta_vec = np.concatenate([theta_acc, theta_flat, theta_dec])
        self.t_vec = np.concatenate([t_acc, t_acc[-1] + t_flat, t_acc[-1] + t_flat[-1] + t_dec])


    def compute_real_motion(self):

        theta_ref = np.array(self.theta_fpga, dtype=float)
        self.t_real = np.interp(theta_ref, self.theta_vec, self.t_vec)
        self.theta_real = np.interp(self.t_real, self.t_vec, self.theta_vec)


    def convert_angles_to_counts(self):

        pulses_per_degree = self.PSOCountsPerRotation / 360.0

        theta_fpga = np.array(self.theta_fpga, dtype=float)

        self.PSOCountsIdeal = np.round(theta_fpga * pulses_per_degree).astype(int)

        if np.any(np.diff(self.PSOCountsIdeal) <= 0):
            print("WARNING: counts non strettamente crescenti (duplicati/inversioni).")

        self.PSOCountsTaxiCorrected = np.round(self.theta_real * pulses_per_degree).astype(int)
        self.PSOCountsFinal = self.PSOCountsTaxiCorrected.copy()

        # --- errori MOD360 (usa theta_sorted_mod)
        pulse_counts_mod = np.round(self.theta_sorted_mod / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_mod = pulse_counts_mod / pulses_per_degree
        angular_error_mod = actual_mod - self.theta_sorted_mod

        for a, p, act, err in zip(self.theta_sorted_mod, pulse_counts_mod, actual_mod, angular_error_mod):
            print(f"Target(sorted MOD): {a:8.2f} deg | Pulse: {p:6d} | Actual: {act:9.6f} deg | Error: {err:+.6f} deg")

        print("********************* unwrapped FPGA angles *********************")
        pulse_counts_unw = np.round(theta_fpga / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_unw = pulse_counts_unw / pulses_per_degree
        angular_error_unw = actual_unw - theta_fpga

        for a, p, act, err in zip(theta_fpga, pulse_counts_unw, actual_unw, angular_error_unw):
            print(f"Target(FPGA): {a:8.2f} deg | Pulse: {p:6d} | Actual: {act:9.6f} deg | Error: {err:+.6f} deg")


    def plot_all_comparisons(self):

        ideal = self.PSOCountsIdeal
        real = self.PSOCountsTaxiCorrected
        final = self.PSOCountsFinal

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


    def plot(self):

        pulses_per_degree = self.PSOCountsPerRotation / 360.0

        # plot MOD 360 (ordinati)
        x1 = np.array(self.theta_sorted_mod, dtype=float)
        y1 = np.round(x1 * pulses_per_degree).astype(int)

        # plot FPGA (unwrapped monotono)
        x2 = np.array(self.theta_fpga, dtype=float)
        y2 = np.round(x2 * pulses_per_degree).astype(int)

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        axs[0].plot(x1, y1, "o-")
        axs[0].set_title("MOD 360 (sorted): Angolo vs Impulsi")
        axs[0].set_xlabel("Angolo [deg]")
        axs[0].set_ylabel("Impulsi encoder")
        axs[0].grid(True)

        axs[1].plot(x2, y2, "s-")
        axs[1].set_title("FPGA (unwrapped monotono): Angolo vs Impulsi")
        axs[1].set_xlabel("Angolo unwrapped [deg]")
        axs[1].set_ylabel("Impulsi encoder")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()


    # ----------------------------------------------------------------------
    #   Export Excel
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

        # -------- MOD 360 (ordinati) --------
        pulse_counts_mod = np.round(self.theta_sorted_mod / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_mod = pulse_counts_mod / pulses_per_degree
        err_mod = actual_mod - self.theta_sorted_mod

        df_mod = pd.DataFrame({
            "target_deg_sorted_mod": self.theta_sorted_mod,
            "pulse": pulse_counts_mod,
            "actual_deg": actual_mod,
            "error_deg": err_mod
        })

        # -------- FPGA unwrapped (ordine acquisizione, monotono) --------
        theta_fpga = np.array(self.theta_fpga, dtype=float)
        pulse_counts_unw = np.round(theta_fpga / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_unw = pulse_counts_unw / pulses_per_degree
        err_unw = actual_unw - theta_fpga

        df_unw = pd.DataFrame({
            "target_deg_fpga": theta_fpga,
            "pulse": pulse_counts_unw,
            "actual_deg": actual_unw,
            "error_deg": err_unw
        })

        # -------- Confronto counts (ideal/taxi/final) --------
        df_counts = pd.DataFrame({
            "theta_fpga_deg": theta_fpga,
            "counts_ideal": self.PSOCountsIdeal,
            "counts_taxi": self.PSOCountsTaxiCorrected,
            "counts_final": self.PSOCountsFinal
        })

        # -------- Δθ su theta_fpga --------
        df_delta = pd.DataFrame({
            "i": np.arange(len(theta_fpga) - 1),
            "theta_i": theta_fpga[:-1],
            "theta_ip1": theta_fpga[1:],
            "delta_theta": np.diff(theta_fpga)
        })

        with pd.ExcelWriter(filename, engine="openpyxl") as w:
            df_mod.to_excel(w, sheet_name="MOD360_SORTED", index=False)
            df_unw.to_excel(w, sheet_name="FPGA_UNWRAPPED", index=False)
            df_counts.to_excel(w, sheet_name="COUNTS", index=False)
            df_delta.to_excel(w, sheet_name="DELTA_THETA_FPGA", index=False)

        print(f"Creato: {filename}")


# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run interlaced scan simulation.")
    parser.add_argument("--num_angles", type=int, default=32, help="Number of angles (default: 32)")
    parser.add_argument("--K_interlace", type=int, default=4, help="Interlace factor K (default: 4)")
    parser.add_argument(
        "--mode",
        choices=["timbir", "golden", "kturns", "multiturns", "corput", "multitimbir"],
        default="timbir"
    )
    parser.add_argument("--PSOCountsPerRotation", type=int, default=20, help="PSO counts per rotation (default: 20)")

    args = parser.parse_args()

    scan = InterlacedScan(
        num_angles=args.num_angles,
        K_interlace=args.K_interlace,
        PSOCountsPerRotation=args.PSOCountsPerRotation
    )

    # select method
    if args.mode == "timbir":
        scan.generate_interlaced_timbir()
        scan.plotdelta("timbir")
        # scan.plotdelta_sort("timbir")

    elif args.mode == "multitimbir":
        scan.generate_interlaced_multitimbir()
        scan.plotdelta("multitimbir")
        # scan.plotdelta_sort("multitimbir")

    elif args.mode == "golden":
        angles_all = scan.generate_interlaced_goldenangle()
        scan.print_angles_table(angles_all)
        scan.print_cumulative_angles_table(angles_all)
        scan.plot_interlaced_circles(angles_all)
        scan.plotdelta("golden")
        # scan.plotdelta_sort("golden")

    elif args.mode == "kturns":
        angles_all = scan.generate_interlaced_kturns()
        scan.plot_equally_loops_polar_kturns()
        scan.print_cumulative_angles_table_kturns(angles_all)
        scan.print_angles_table_kturns(angles_all)
        scan.plotdelta("kturns")
        # scan.plotdelta_sort("kturns")

    elif args.mode == "multiturns":
        angles_all = scan.generate_interlaced_multiturns()
        scan.plot_equally_loops_polar_multiturns()
        scan.print_cumulative_angles_table_multiturns(angles_all)
        scan.print_angles_table_multiturns(angles_all)
        scan.plotdelta("multiturns")
        # scan.plotdelta_sort("multiturns")

    elif args.mode == "corput":
        angles_all = scan.generate_interlaced_corput()
        scan.plot_equally_loops_polar_corput()
        scan.print_cumulative_angles_table_corput(angles_all)
        scan.print_angles_table_corput(angles_all)
        scan.plot_live_corput()
        scan.plotdelta("corput")
        # scan.plotdelta_sort("corput")

    # sorted / motion
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

# python Tomoscan_pso_interlaced.py  --mode multitimbir  --num_angles 32 --K_interlace 4 --PSOCountsPerRotation 20000  > multitimbir_32_4_20000_output.txt


