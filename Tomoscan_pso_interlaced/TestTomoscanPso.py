import numpy as np
import math
import struct
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

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

    ############################################################ MODE ####################################################################

    # ----------------------------------------------------------------------
    #   TIMBIR
    # ----------------------------------------------------------------------
    def generate_interlaced_timbir(self):

        bits = int(np.log2(self.K_interlace)) if self.K_interlace > 1 else 0
        theta = []
        group_indices = []
        group_br_list = []
        idx_list = []

        assert (self.K_interlace & (self.K_interlace - 1)) == 0

        for n in range(self.num_angles):
            group = (n * self.K_interlace // self.num_angles) % self.K_interlace
            group_br = self.bit_reverse(group, bits)
            idx = n * self.K_interlace + group_br
            angle_deg = (idx % self.num_angles) * 360.0 / self.num_angles

            theta.append(angle_deg)
            group_indices.append(group)
            group_br_list.append(group_br)
            idx_list.append(idx)

        # --- Salvo in attributi "dedicati" TIMBIR (acq order)
        self.theta_timbir_acq = np.array(theta, dtype=float)
        self.theta_timbir_acq_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(self.theta_timbir_acq)))
        self.group_timbir = np.array(group_indices, dtype=int)
        self.group_br_timbir = np.array(group_br_list, dtype=int)
        self.idx_timbir = np.array(idx_list, dtype=int)

        # --- Se ti servono ancora questi per plot/altro
        self.theta_interlaced = np.sort(theta)
        self.theta_interlaced_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(theta)))

        group_indices = np.array(group_indices)
        radii = 1 - group_indices * 0.15

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"TIMBIR Interlaced Acquisition (N={self.num_angles} - K={self.K_interlace})\nEach loop on its own circle",
            va='bottom', fontsize=13
        )

        ax.plot(np.deg2rad(theta), radii, '-o', lw=1.2, ms=5, alpha=0.8, color='tab:blue')

        for i in range(self.num_angles):
            ax.text(np.deg2rad(theta[i]), radii[i] + 0.03,
                    str(group_indices[i] + 1), ha='center', va='bottom', fontsize=8)

        ax.set_rticks([])
        plt.show()

    def print_delta_angles_timbir(self, out_dir="angles_out", print_first=20):
        """
          - CSV in ordine di acquisizione con angoli (wrapped/unwrapped) e delta tra step consecutivi
          - CSV con angoli ordinati (0..360) e delta ciclico tra vicini
        """
        if not hasattr(self, "theta_timbir_acq"):
            raise RuntimeError("Prima chiama generate_interlaced_timbir(), poi print_delta_angles_timbir().")

        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        ang_w = self.theta_timbir_acq % 360.0
        ang_u = self.theta_timbir_acq_unwrapped

        delta_u = np.diff(ang_u)
        delta_u_full = np.concatenate([delta_u, [np.nan]])

        acq_path = os.path.join(out_dir, f"timbir_acq_angles_N{self.num_angles}_K{self.K_interlace}_{ts}.csv")
        header = "n,angle_deg_wrapped,angle_deg_unwrapped,delta_next_deg_unwrapped,group,group_br,idx"
        data = np.column_stack([
            np.arange(len(ang_w)),
            ang_w,
            ang_u,
            delta_u_full,
            self.group_timbir,
            self.group_br_timbir,
            self.idx_timbir
        ])
        np.savetxt(acq_path, data, delimiter=",", header=header, comments="", fmt="%.10g")

        ang_sorted = np.sort(ang_w)
        delta_sorted = np.diff(ang_sorted, append=ang_sorted[0] + 360.0)
        sort_path = os.path.join(out_dir, f"timbir_sorted_deltas_N{self.num_angles}_K{self.K_interlace}_{ts}.csv")
        header2 = "k,angle_deg_sorted,delta_to_next_deg_cyclic"
        data2 = np.column_stack([np.arange(len(ang_sorted)), ang_sorted, delta_sorted])
        np.savetxt(sort_path, data2, delimiter=",", header=header2, comments="", fmt="%.10g")

        print(f"[TIMBIR] Salvato: {acq_path}")
        print(f"[TIMBIR] Salvato: {sort_path}")

        print("\n[TIMBIR] Prime righe (acq order):")
        for i in range(min(print_first, len(ang_w))):
            print(f"  n={i:3d}  ang(w)={ang_w[i]:9.4f}°  ang(u)={ang_u[i]:9.4f}°  "
                  f"Δnext={delta_u_full[i]:9.4f}°  group={self.group_timbir[i]}  br={self.group_br_timbir[i]}")

        finite_du = delta_u[np.isfinite(delta_u)]
        print("\n[TIMBIR] Stat Δ (acq, unwrapped):",
              f"min={np.min(finite_du):.6g}°  max={np.max(finite_du):.6g}°  mean={np.mean(finite_du):.6g}°  std={np.std(finite_du):.6g}°")

    def bit_reverse(self, n, bits):
        return int(f"{n:0{bits}b}"[::-1], 2)

    # ----------------------------------------------------------------------
    #   Multi-TIMBIR
    # ----------------------------------------------------------------------
    def generate_interlaced_multitimbir(self):

        bits = int(np.log2(self.K_interlace)) if self.K_interlace > 1 else 0
        theta = []
        group_indices = []
        i_list = []
        g_list = []
        idx_list = []

        assert (self.K_interlace & (self.K_interlace - 1)) == 0

        for i in range(self.num_angles):
            for g in range(self.K_interlace):
                loop = self.bit_reverse(g, bits)
                idx = i * self.K_interlace + loop
                angle_deg = idx * 360.0 / (self.num_angles * self.K_interlace)

                theta.append(angle_deg)
                group_indices.append(loop)
                i_list.append(i)
                g_list.append(g)
                idx_list.append(idx)

        self.theta_multitimbir_acq = np.array(theta, dtype=float)
        self.theta_multitimbir_acq_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(self.theta_multitimbir_acq)))
        self.loop_multitimbir = np.array(group_indices, dtype=int)
        self.i_multitimbir = np.array(i_list, dtype=int)
        self.g_multitimbir = np.array(g_list, dtype=int)
        self.idx_multitimbir = np.array(idx_list, dtype=int)

        self.theta_interlaced = np.sort(theta)
        self.theta_interlaced_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(theta)))

        theta = np.array(theta, dtype=float)
        group_indices = np.array(group_indices, dtype=int)

        step = 0.8 / max(self.K_interlace - 1, 1)
        radii = 1.0 - group_indices * step

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title(
            f"Multi-TIMBIR: N={self.num_angles} per loop, K={self.K_interlace} → totale N·K={self.num_angles*self.K_interlace} angoli\n"
            f"Loop su cerchi separati con ordine loop = bit-reversal",
            va='bottom', fontsize=12
        )
        ax.plot(np.deg2rad(theta), radii, '-o', lw=1.2, ms=5, alpha=0.8)

        for ang, r, lp in zip(theta, radii, group_indices):
            ax.text(np.deg2rad(ang), r + 0.03, str(lp + 1),
                    ha='center', va='bottom', fontsize=8)

        ax.set_rticks([])
        plt.show()

    def print_delta_angles_multitimbir(self, out_dir="angles_out", print_first=20):
        """
          - CSV in ordine di acquisizione con angoli (wrapped/unwrapped) e delta tra step consecutivi
          - CSV con angoli ordinati (0..360) e delta ciclico tra vicini
        """
        if not hasattr(self, "theta_multitimbir_acq"):
            raise RuntimeError("Prima chiama generate_interlaced_multitimbir(), poi print_delta_angles_multitimbir().")

        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        ang_w = self.theta_multitimbir_acq % 360.0
        ang_u = self.theta_multitimbir_acq_unwrapped

        delta_u = np.diff(ang_u)
        delta_u_full = np.concatenate([delta_u, [np.nan]])

        acq_path = os.path.join(out_dir, f"multitimbir_acq_angles_N{self.num_angles}_K{self.K_interlace}_{ts}.csv")
        header = "n,i,g,loop_bitrev,idx,angle_deg_wrapped,angle_deg_unwrapped,delta_next_deg_unwrapped"
        data = np.column_stack([
            np.arange(len(ang_w)),
            self.i_multitimbir,
            self.g_multitimbir,
            self.loop_multitimbir,
            self.idx_multitimbir,
            ang_w,
            ang_u,
            delta_u_full
        ])
        np.savetxt(acq_path, data, delimiter=",", header=header, comments="", fmt="%.10g")

        ang_sorted = np.sort(ang_w)
        delta_sorted = np.diff(ang_sorted, append=ang_sorted[0] + 360.0)
        sort_path = os.path.join(out_dir, f"multitimbir_sorted_deltas_N{self.num_angles}_K{self.K_interlace}_{ts}.csv")
        header2 = "k,angle_deg_sorted,delta_to_next_deg_cyclic"
        data2 = np.column_stack([np.arange(len(ang_sorted)), ang_sorted, delta_sorted])
        np.savetxt(sort_path, data2, delimiter=",", header=header2, comments="", fmt="%.10g")

        print(f"[Multi-TIMBIR] Salvato: {acq_path}")
        print(f"[Multi-TIMBIR] Salvato: {sort_path}")

        print("\n[Multi-TIMBIR] Prime righe (acq order):")
        for i in range(min(print_first, len(ang_w))):
            print(f"  n={i:3d}  i={self.i_multitimbir[i]:3d} g={self.g_multitimbir[i]:2d} loop={self.loop_multitimbir[i]:2d}  "
                  f"ang(w)={ang_w[i]:9.4f}°  ang(u)={ang_u[i]:9.4f}°  Δnext={delta_u_full[i]:9.4f}°")

        finite_du = delta_u[np.isfinite(delta_u)]
        print("\n[Multi-TIMBIR] Stat Δ (acq, unwrapped):",
              f"min={np.min(finite_du):.6g}°  max={np.max(finite_du):.6g}°  mean={np.mean(finite_du):.6g}°  std={np.std(finite_du):.6g}°")

    #*******************************************************************************
    # ----------------------------------------------------------------------
    #   GOLDEN ANGLE
    # ----------------------------------------------------------------------
    def generate_interlaced_goldenangle(self):

        golden_angle = 360 * (3 - np.sqrt(5)) / 2
        phi_inv = (np.sqrt(5) - 1) / 2

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

    # ... (tutto il resto dei tuoi metodi resta uguale e già indentato correttamente)
    # ----------------------------------------------------------------------
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

    # ... continua con i tuoi metodi già corretti ...


# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run interlaced scan simulation.")
    parser.add_argument("--num_angles", type=int, default=32, help="Number of angles (default: 32)")
    parser.add_argument("--K_interlace", type=int, default=4, help="Interlace factor K (default: 4)")
    parser.add_argument("--mode", choices=["timbir", "golden", "kturns", "multiturns", "corput", "multitimbir"], default="timbir")
    parser.add_argument("--PSOCountsPerRotation", type=int, default=20, help="PSO counts per rotation (default: 20)")

    args = parser.parse_args()

    scan = InterlacedScan(
        num_angles=args.num_angles,
        K_interlace=args.K_interlace,
        PSOCountsPerRotation=args.PSOCountsPerRotation,
    )

    # select method
    if args.mode == "timbir":
        scan.generate_interlaced_timbir()
        scan.print_delta_angles_timbir(out_dir="debug_angles")

    elif args.mode == "multitimbir":
        scan.generate_interlaced_multitimbir()
        scan.print_delta_angles_multitimbir(out_dir="debug_angles")

    elif args.mode == "golden":
        angles_all = scan.generate_interlaced_goldenangle()
        scan.print_angles_table(angles_all)
        scan.print_cumulative_angles_table(angles_all)
        scan.plot_interlaced_circles(angles_all)

    elif args.mode == "kturns":
        angles_all = scan.generate_interlaced_kturns()
        scan.plot_equally_loops_polar_kturns()
        scan.print_cumulative_angles_table_kturns(angles_all)
        scan.print_angles_table_kturns(angles_all)

    elif args.mode == "multiturns":
        angles_all = scan.generate_interlaced_multiturns()
        scan.plot_equally_loops_polar_multiturns()
        scan.print_cumulative_angles_table_multiturns(angles_all)
        scan.print_angles_table_multiturns(angles_all)

    elif args.mode == "corput":
        angles_all = scan.generate_interlaced_corput()
        scan.plot_equally_loops_polar_corput()
        scan.print_cumulative_angles_table_corput(angles_all)
        scan.print_angles_table_corput(angles_all)
        scan.plot_live_corput()

    # sorted
    scan.compute_positions_PSO()
    scan.simulate_taxi_motion()
    scan.compute_real_motion()
    scan.convert_angles_to_counts()

    scan.plot_all_comparisons()
    scan.plot()


if __name__ == "__main__":
    main()
