```python
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

        # Liste angoli utili
        self.theta_fpga = None                    # angoli finali da mandare alla FPGA (monotoni nel tempo)
        self.theta_acq_mod = None                 # angoli in ordine reale di acquisizione (mod 360)
        self.theta_sorted_mod = None              # angoli (mod 360) ordinati (solo analisi copertura)
        self.theta_acq_unwrapped_monotono = None  # unwrap monotono (0..360..720..)

        # Alias/compatibilità (tenuti ma non usati come sorgente nelle funzioni successive)
        self.theta_interlaced = None
        self.theta_interlaced_unwrapped = None

        # contenitore completo per export/debug
        self.angles_all = None

        # variabili moto/counts
        self.theta_vec = None
        self.t_vec = None
        self.t_real = None
        self.theta_real = None

        self.PSOCountsIdeal = None
        self.PSOCountsTaxiCorrected = None
        self.PSOCountsFinal = None

    # ----------------------------------------------------------------------
    # helper: unwrap monotono (FPGA-style)
    # ----------------------------------------------------------------------
    def unwrap_monotone(self, theta_mod):
        out = np.asarray(theta_mod, dtype=float).copy()
        for i in range(1, len(out)):
            while out[i] < out[i - 1]:
                out[i] += 360.0
        return out

    def _require_theta_fpga(self):
        if self.theta_fpga is None:
            raise ValueError("theta_fpga è None: devi chiamare prima un generate_*().")

    # ----------------------------------------------------------------------
    # bit reverse
    # ----------------------------------------------------------------------
    def bit_reverse(self, n, bits):
        return int(f"{n:0{bits}b}"[::-1], 2) if bits > 0 else int(n)

    # =========================================================
    # MODE
    # =========================================================

    # ----------------------------------------------------------------------
    #   TIMBIR
    # ----------------------------------------------------------------------
    def generate_interlaced_timbir(self):

        bits = int(np.log2(self.K_interlace)) if self.K_interlace > 1 else 0
        theta = []
        group_indices = []

        assert (self.K_interlace & (self.K_interlace - 1)) == 0

        for n in range(self.num_angles):
            group = (n * self.K_interlace // self.num_angles) % self.K_interlace
            group_br = self.bit_reverse(group, bits) if bits > 0 else group
            idx = n * self.K_interlace + group_br
            angle_deg = (idx % self.num_angles) * 360.0 / self.num_angles
            theta.append(angle_deg)
            group_indices.append(group)

        theta = np.asarray(theta, dtype=float)

        # ordine reale (mod 360)
        self.theta_acq_mod = theta.copy()

        # per copertura
        self.theta_sorted_mod = np.sort(self.theta_acq_mod)

        # FPGA: unwrap monotono nel tempo
        self.theta_fpga = self.unwrap_monotone(self.theta_acq_mod)
        self.theta_acq_unwrapped_monotono = self.theta_fpga.copy()

        # alias/compatibilità
        self.theta_interlaced = self.theta_sorted_mod
        self.theta_interlaced_unwrapped = self.theta_fpga

        # pacchetto export/debug
        self.angles_all = {
            "n": np.arange(self.num_angles),
            "group": np.asarray(group_indices, dtype=int),
            "theta_acq_mod": self.theta_acq_mod,
            "theta_sorted_mod": self.theta_sorted_mod,
            "theta_acq_unwrapped_monotono": self.theta_acq_unwrapped_monotono,
            "theta_fpga": self.theta_fpga,
        }

        # plot
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

        ax.plot(np.deg2rad(self.theta_acq_mod), radii, "-o", lw=1.2, ms=5, alpha=0.8, color="tab:blue")

        for i in range(self.num_angles):
            ax.text(
                np.deg2rad(self.theta_acq_mod[i]),
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

        bits = int(np.log2(self.K_interlace)) if self.K_interlace > 1 else 0

        theta = []
        loop_br_list = []
        i_list = []
        g_list = []
        idx_list = []

        assert (self.K_interlace & (self.K_interlace - 1)) == 0

        # i = step (0..N-1), g = loop nominale (0..K-1)
        for i in range(self.num_angles):
            for g in range(self.K_interlace):
                loop_br = self.bit_reverse(g, bits) if bits > 0 else g
                idx = i * self.K_interlace + loop_br
                angle_deg = idx * 360.0 / (self.num_angles * self.K_interlace)

                theta.append(angle_deg)
                loop_br_list.append(loop_br)
                i_list.append(i)
                g_list.append(g)
                idx_list.append(idx)

        theta = np.asarray(theta, dtype=float)
        loop_br_list = np.asarray(loop_br_list, dtype=int)
        i_list = np.asarray(i_list, dtype=int)
        g_list = np.asarray(g_list, dtype=int)
        idx_list = np.asarray(idx_list, dtype=int)

        # ordine reale (mod 360)
        self.theta_acq_mod = np.mod(theta, 360.0)

        # per copertura
        self.theta_sorted_mod = np.sort(self.theta_acq_mod)

        # FPGA: unwrap monotono nel tempo
        self.theta_fpga = self.unwrap_monotone(self.theta_acq_mod)
        self.theta_acq_unwrapped_monotono = self.theta_fpga.copy()

        # alias/compatibilità
        self.theta_interlaced = self.theta_sorted_mod
        self.theta_interlaced_unwrapped = self.theta_fpga

        self.angles_all = {
            "acq_n": np.arange(self.num_angles * self.K_interlace),
            "i": i_list,
            "g": g_list,
            "loop_br": loop_br_list,
            "idx": idx_list,
            "theta_acq_mod": self.theta_acq_mod,
            "theta_sorted_mod": self.theta_sorted_mod,
            "theta_acq_unwrapped_monotono": self.theta_acq_unwrapped_monotono,
            "theta_fpga": self.theta_fpga,
        }

        # plot (cerchi per loop_br)
        step = 0.8 / max(self.K_interlace - 1, 1)
        radii = 1.0 - loop_br_list * step

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"Multi-TIMBIR: N={self.num_angles} per loop, K={self.K_interlace} → totale N·K={self.num_angles*self.K_interlace} angoli\n"
            "Loop su cerchi separati con ordine loop = bit-reversal",
            va="bottom",
            fontsize=12
        )

        ax.plot(np.deg2rad(self.theta_acq_mod), radii, "o", lw=1.2, ms=5, alpha=0.8)

        for ang, r, lp in zip(self.theta_acq_mod, radii, loop_br_list):
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

        # Loop 0 (base in [0,360))
        base = np.array([(self.rotation_start + i * golden_angle) % 360.0
                         for i in range(self.num_angles)], dtype=float)
        base = np.sort(base)
        angles_all.append(base)

        # Loop 1..K-1: offset deterministici
        for k in range(1, self.K_interlace):
            offset = (k / (self.num_angles + 1.0)) * 360.0 * phi_inv
            angles_all.append(np.sort((base + offset) % 360.0))

        # ordine di acquisizione (time): interleaving loop
        theta_time = []
        loop_time = []

        loop_order = list(range(self.K_interlace))  # puoi cambiarlo (es. bit-reversal) se vuoi
        for i in range(self.num_angles):
            for k in loop_order:
                theta_time.append(angles_all[k][i])
                loop_time.append(k)

        theta_time = np.asarray(theta_time, dtype=float)
        loop_time = np.asarray(loop_time, dtype=int)

        # ordine reale (mod 360)
        self.theta_acq_mod = np.mod(theta_time, 360.0)

        # per copertura
        self.theta_sorted_mod = np.sort(self.theta_acq_mod)

        # FPGA: unwrap monotono nel tempo
        self.theta_fpga = self.unwrap_monotone(self.theta_acq_mod)
        self.theta_acq_unwrapped_monotono = self.theta_fpga.copy()

        # alias/compatibilità
        self.theta_interlaced = self.theta_sorted_mod
        self.theta_interlaced_unwrapped = self.theta_fpga

        self.angles_all = {
            "acq_n": np.arange(self.num_angles * self.K_interlace),
            "loop": loop_time,
            "theta_acq_mod": self.theta_acq_mod,
            "theta_sorted_mod": self.theta_sorted_mod,
            "theta_acq_unwrapped_monotono": self.theta_acq_unwrapped_monotono,
            "theta_fpga": self.theta_fpga,
        }

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

        theta_unwrapped = np.concatenate(angles_all)
        theta_mod = np.mod(theta_unwrapped, 360.0)

        # ordine reale (mod 360)
        self.theta_acq_mod = theta_mod.copy()

        # per copertura
        self.theta_sorted_mod = np.sort(self.theta_acq_mod)

        # FPGA: unwrap monotono nel tempo
        self.theta_fpga = self.unwrap_monotone(self.theta_acq_mod)
        self.theta_acq_unwrapped_monotono = self.theta_fpga.copy()

        # alias/compatibilità
        self.theta_interlaced = self.theta_sorted_mod
        self.theta_interlaced_unwrapped = self.theta_fpga

        if self.K_interlace > 1:
            self.rotation_stop = float(self.theta_fpga[-1])

        self.angles_all = {
            "acq_n": np.arange(self.num_angles * self.K_interlace),
            "theta_acq_mod": self.theta_acq_mod,
            "theta_sorted_mod": self.theta_sorted_mod,
            "theta_fpga": self.theta_fpga,
        }

        return angles_all

    def plot_equally_loops_polar_kturns(self):

        self._require_theta_fpga()

        theta_unwrapped = self.theta_fpga
        theta_mod = np.mod(theta_unwrapped, 360.0)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"Equally Spaced Acquisition (N={self.num_angles}, K={self.K_interlace})\n"
            "Each loop on its own circle",
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
                ax.text(np.deg2rad(ang), radii[i] + 0.03, str(k + 1),
                        ha="center", va="bottom", fontsize=8)

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
            theta_n = self.rotation_start + (n + k / K) * delta_theta
            angles_all.append(theta_n)

        theta_unwrapped = np.concatenate(angles_all)
        theta_mod = np.mod(theta_unwrapped, 360.0)

        # ordine reale (mod 360)
        self.theta_acq_mod = theta_mod.copy()

        # per copertura
        self.theta_sorted_mod = np.sort(self.theta_acq_mod)

        # FPGA: unwrap monotono nel tempo
        self.theta_fpga = self.unwrap_monotone(self.theta_acq_mod)
        self.theta_acq_unwrapped_monotono = self.theta_fpga.copy()

        # alias/compatibilità
        self.theta_interlaced = self.theta_sorted_mod
        self.theta_interlaced_unwrapped = self.theta_fpga

        if self.K_interlace > 1:
            self.rotation_stop = float(self.theta_fpga[-1])

        self.angles_all = {
            "acq_n": np.arange(self.num_angles * self.K_interlace),
            "theta_acq_mod": self.theta_acq_mod,
            "theta_sorted_mod": self.theta_sorted_mod,
            "theta_fpga": self.theta_fpga,
        }

        return angles_all

    def plot_equally_loops_polar_multiturns(self):

        self._require_theta_fpga()

        theta_unwrapped = self.theta_fpga
        theta_mod = np.mod(theta_unwrapped, 360.0)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"Equally Spaced Acquisition (N={self.num_angles}, K={self.K_interlace})\n"
            "Each loop on its own circle",
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
                ax.text(np.deg2rad(ang), radii[i] + 0.03, str(k + 1),
                        ha="center", va="bottom", fontsize=8)

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
    # VAN DER CORPUT INTERLACED (ordine temporale + theta_fpga)
    # =========================================================
    def generate_interlaced_corput(self, delta_theta=None):

        if delta_theta is not None:
            delta_theta = float(delta_theta)
        else:
            delta_theta = (self.rotation_stop - self.rotation_start) / (self.num_angles - 1)

        self.rotation_step = delta_theta

        base = self.rotation_start + np.arange(self.num_angles) * delta_theta

        K = self.K_interlace
        bitsK = int(np.ceil(np.log2(K))) if K > 1 else 0
        MK = 1 << bitsK if bitsK > 0 else 1

        p_corput = np.array([self.bit_reverse(i, bitsK) for i in range(MK)], dtype=int)
        p_corput = p_corput[p_corput < K]
        assert len(p_corput) == K

        offsets = (p_corput / K) * delta_theta if K > 0 else np.array([0.0])

        bitsN = int(np.ceil(np.log2(self.num_angles))) if self.num_angles > 1 else 0
        MN = 1 << bitsN if bitsN > 0 else 1
        indices = np.array([self.bit_reverse(i, bitsN) for i in range(MN)], dtype=int)
        indices = indices[indices < self.num_angles]

        angles_all = []
        for k in range(K):
            offset = offsets[k]
            loop_angles = base[indices] + offset
            loop_angles_mod = np.mod(loop_angles - self.rotation_start, 360.0) + self.rotation_start
            angles_all.append(loop_angles_mod)

        # ordine reale di acquisizione (qui: concateno loop 0,1,2,...)
        theta_time_mod = np.concatenate(angles_all)

        self.theta_acq_mod = np.mod(theta_time_mod, 360.0)
        self.theta_sorted_mod = np.sort(self.theta_acq_mod)

        self.theta_fpga = self.unwrap_monotone(self.theta_acq_mod)
        self.theta_acq_unwrapped_monotono = self.theta_fpga.copy()

        self.theta_interlaced = self.theta_sorted_mod
        self.theta_interlaced_unwrapped = self.theta_fpga

        if self.K_interlace > 1:
            self.rotation_stop = float(self.theta_fpga[-1])

        self.angles_all = {
            "acq_n": np.arange(self.num_angles * self.K_interlace),
            "theta_acq_mod": self.theta_acq_mod,
            "theta_sorted_mod": self.theta_sorted_mod,
            "theta_fpga": self.theta_fpga,
            "indices_br": indices,
            "offsets": offsets,
            "p_corput": p_corput,
        }

        return angles_all

    def plot_equally_loops_polar_corput(self):

        self._require_theta_fpga()

        theta_unwrapped = self.theta_fpga
        theta_mod = np.mod(theta_unwrapped, 360.0)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"Interlaced Van der Corput (time-order) (N={self.num_angles}, K={self.K_interlace})",
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
                ax.text(np.deg2rad(ang), radii[i] + 0.03, str(k + 1),
                        ha="center", va="bottom", fontsize=8)

        ax.set_rticks([])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.show()

    def print_angles_table_corput(self, angles_all):
        print(f"{'Idx':>5}", end="")
        for k in range(len(angles_all)):
            print(f"{f'Loop {k + 1}':>15}", end="")
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
            print(f"{f'Loop {k + 1}':>18}", end="")
        print()

        for i in range(len(cumulative[0])):
            print(f"{i:5}", end="")
            for loop in cumulative:
                print(f"{loop[i]:18.3f}", end="")
            print()

    def plot_live_corput(self):

        self._require_theta_fpga()

        theta_mod = np.mod(self.theta_fpga, 360.0)
        n_total = len(theta_mod)
        indices = np.arange(n_total)

        fig, ax = plt.subplots(figsize=(9, 5))

        for k in range(self.K_interlace):
            start = k * self.num_angles
            stop = (k + 1) * self.num_angles

            ax.scatter(indices[start:stop], theta_mod[start:stop], s=18, alpha=0.85, label=f"Loop {k+1}")
            ax.plot(indices[start:stop], theta_mod[start:stop], lw=0.6, alpha=0.4)

        ax.set_title(
            f"Live Acquisition Order – Van der Corput (time-order) (N={self.num_angles}, K={self.K_interlace})",
            fontsize=13
        )
        ax.set_xlabel("Acquisition index")
        ax.set_ylabel("Angle [deg] (mod 360)")
        ax.set_ylim(0, 360)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------------
    #   Δθ consecutivi (ordine acquisizione FPGA = theta_fpga)
    # ----------------------------------------------------------------------
    def plotdelta(self, metodo=""):

        self._require_theta_fpga()

        theta = np.asarray(self.theta_fpga, dtype=float)
        dtheta = np.diff(theta)

        print(f"\n--- Δθ (theta_fpga) ({metodo}) ---")
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
    #   Δθ sort (copertura su mod 360, derivato da theta_fpga)
    # ----------------------------------------------------------------------
    def plotdelta_sort(self, metodo=""):

        self._require_theta_fpga()

        theta_mod_sorted = np.sort(np.mod(np.asarray(self.theta_fpga, dtype=float), 360.0))
        dtheta = np.diff(theta_mod_sorted)

        print(f"\n--- Δθ sort (mod da theta_fpga) ({metodo}) ---")
        for i, d in enumerate(dtheta):
            print(f"{i:4d} -> {i+1:4d}: {d:9.3f} deg")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.arange(1, len(theta_mod_sorted)), dtheta, "o")
        ax.set_title(f"Δθ sort (mod da theta_fpga) – {metodo}")
        ax.set_xlabel("Indice (sorted)")
        ax.set_ylabel("Δθ [deg]")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------------
    #           FUNZIONI (moto/counts)
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
    # Modello taxi (usa theta_fpga)
    # ----------------------------------------------------------------------
    def simulate_taxi_motion(self, omega_target=10, dt=1e-4):

        self._require_theta_fpga()

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

        self._require_theta_fpga()

        self.t_real = np.interp(self.theta_fpga, self.theta_vec, self.t_vec)
        self.theta_real = np.interp(self.t_real, self.t_vec, self.theta_vec)

    def convert_angles_to_counts(self):

        self._require_theta_fpga()

        pulses_per_degree = self.PSOCountsPerRotation / 360.0

        theta_unw = np.asarray(self.theta_fpga, dtype=float)
        theta_mod = np.mod(theta_unw, 360.0)

        # counts ideal (basati su theta_fpga)
        self.PSOCountsIdeal = np.round(theta_unw * pulses_per_degree).astype(int)

        if np.any(np.diff(self.PSOCountsIdeal) <= 0):
            print("WARNING: counts non strettamente crescenti (duplicati/inversioni).")

        # counts corretti dal taxi model
        self.PSOCountsTaxiCorrected = np.round(self.theta_real * pulses_per_degree).astype(int)
        self.PSOCountsFinal = self.PSOCountsTaxiCorrected.copy()

        # stampa MOD 360 (derivato da theta_fpga)
        pulse_counts_mod = np.round(theta_mod / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_mod = pulse_counts_mod / pulses_per_degree
        err_mod = actual_mod - theta_mod

        print("********************* mod 360 (da theta_fpga) *********************")
        for a, p, act, err in zip(theta_mod, pulse_counts_mod, actual_mod, err_mod):
            print(f"Target: {a:8.2f} deg | Pulse: {p:6d} | Actual: {act:9.6f} deg | Error: {err:+.6f} deg")

        # stampa UNWRAPPED (theta_fpga)
        pulse_counts_unw = np.round(theta_unw / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_unw = pulse_counts_unw / pulses_per_degree
        err_unw = actual_unw - theta_unw

        print("********************* unwrapped (theta_fpga) *********************")
        for a, p, act, err in zip(theta_unw, pulse_counts_unw, actual_unw, err_unw):
            print(f"Target: {a:8.2f} deg | Pulse: {p:6d} | Actual: {act:9.6f} deg | Error: {err:+.6f} deg")

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

        self._require_theta_fpga()

        pulses_per_degree = self.PSOCountsPerRotation / 360.0

        x2 = np.asarray(self.theta_fpga, dtype=float)  # UNWRAPPED = FPGA
        x1 = np.mod(x2, 360.0)                         # MOD 360 derivato

        y1 = np.round(x1 * pulses_per_degree).astype(int)
        y2 = np.round(x2 * pulses_per_degree).astype(int)

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        axs[0].plot(x1, y1, "o-")
        axs[0].set_title("MOD 360 (da theta_fpga): Angolo vs Impulsi")
        axs[0].set_xlabel("Angolo [deg]")
        axs[0].set_ylabel("Impulsi encoder")
        axs[0].grid(True)

        axs[1].plot(x2, y2, "s-")
        axs[1].set_title("UNWRAPPED (theta_fpga): Angolo vs Impulsi")
        axs[1].set_xlabel("Angolo unwrapped [deg]")
        axs[1].set_ylabel("Impulsi encoder")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------------------------
    #   Export Excel (usa theta_fpga)
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

        self._require_theta_fpga()

        pulses_per_degree = self.PSOCountsPerRotation / 360.0

        theta_unw = np.asarray(self.theta_fpga, dtype=float)
        theta_mod = np.mod(theta_unw, 360.0)

        # -------- MOD 360 (da theta_fpga) --------
        pulse_counts_mod = np.round(theta_mod / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_mod = pulse_counts_mod / pulses_per_degree
        err_mod = actual_mod - theta_mod

        df_mod = pd.DataFrame({
            "target_deg": theta_mod,
            "pulse": pulse_counts_mod,
            "actual_deg": actual_mod,
            "error_deg": err_mod
        })

        # -------- UNWRAPPED (theta_fpga) --------
        pulse_counts_unw = np.round(theta_unw / 360.0 * self.PSOCountsPerRotation).astype(int)
        actual_unw = pulse_counts_unw / pulses_per_degree
        err_unw = actual_unw - theta_unw

        df_unw = pd.DataFrame({
            "target_deg": theta_unw,
            "pulse": pulse_counts_unw,
            "actual_deg": actual_unw,
            "error_deg": err_unw
        })

        # -------- Confronto counts --------
        df_counts = pd.DataFrame({
            "theta_fpga_unwrapped_deg": theta_unw,
            "counts_ideal": self.PSOCountsIdeal,
            "counts_taxi": self.PSOCountsTaxiCorrected,
            "counts_final": self.PSOCountsFinal
        })

        # -------- Δθ --------
        df_delta = pd.DataFrame({
            "i": np.arange(len(theta_unw) - 1),
            "theta_i": theta_unw[:-1],
            "theta_ip1": theta_unw[1:],
            "delta_theta": np.diff(theta_unw)
        })

        with pd.ExcelWriter(filename, engine="openpyxl") as w:
            df_mod.to_excel(w, sheet_name="MOD360", index=False)
            df_unw.to_excel(w, sheet_name="UNWRAPPED", index=False)
            df_counts.to_excel(w, sheet_name="COUNTS", index=False)
            df_delta.to_excel(w, sheet_name="DELTA_THETA", index=False)

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
    parser.add_argument("--PSOCountsPerRotation", type=int, default=20000,
                        help="PSO counts per rotation (default: 20000)")

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

    # motion + counts (ora TUTTO basato su theta_fpga)
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

# Esempi:
# python Tomoscan_pso_interlaced.py --mode multitimbir --num_angles 32 --K_interlace 4 --PSOCountsPerRotation 20000 > multitimbir_32_4_20000_output.txt
# python -m py_compile Tomoscan_pso_interlaced.py
```
