import numpy as np
import struct
import math
import matplotlib.pyplot as plt


# ============================================================================
#                     CLASSE InterlacedScan (UFFICIALE)
# ============================================================================

class InterlacedScan:

    """
    Classe finale che produce:
    - theta_interlaced ordinati (TIMBIR)
    - theta_interlaced_real (corretti tramite taxi)
    - pulses_interlaced_ideal
    - pulses_interlaced_real
    - file pulses.bin compatibile con FPGA
    - grafici diagnostici

    Indipendente da EPICS e Tomoscan.
    """

    def __init__(self,
                 N_theta=32,
                 K=4,
                 PSOCountsPerRotation=20000,
                 accel=5,
                 decel=5,
                 omega_target=10,
                 dt=1e-4):

        self.N_theta = N_theta
        self.K = K
        self.PSOCountsPerRotation = PSOCountsPerRotation

        self.accel = accel
        self.decel = decel
        self.omega_target = omega_target
        self.dt = dt

        self.pulses_per_degree = PSOCountsPerRotation / 360.0

    # ============================================================================
    #                       BIT–REVERSAL (TIMBIR)
    # ============================================================================
    def bit_reverse(self, n, bits):
        b = f'{n:0{bits}b}'
        return int(b[::-1], 2)

    def generate_timbir_angles(self):
        bits = int(np.log2(self.N_theta))
        theta = np.array([self.bit_reverse(n, bits) * 360.0 / self.N_theta
                          for n in range(self.N_theta)])
        return np.sort(theta)   # ORDINATI QUI

    # ============================================================================
    #                       TAXI MODEL (θ(t))
    # ============================================================================
    def simulate_taxi_motion(self):
        accel = self.accel
        decel = self.decel
        omega_target = self.omega_target
        dt = self.dt
        theta_total = 360.0

        # Accelerazione
        T_acc = omega_target / accel
        t_acc = np.arange(0, T_acc, dt)
        theta_acc = 0.5 * accel * t_acc**2

        # Plateau
        theta_flat_len = theta_total - 2 * theta_acc[-1]
        T_flat = theta_flat_len / omega_target
        t_flat = np.arange(0, T_flat, dt)
        theta_flat = theta_acc[-1] + omega_target * t_flat

        # Decelerazione
        T_dec = omega_target / decel
        t_dec = np.arange(0, T_dec, dt)
        theta_dec = theta_flat[-1] + omega_target*t_dec - 0.5*decel*t_dec**2

        # Concatenazione
        t_vec = np.concatenate([t_acc,
                                t_acc[-1]+t_flat,
                                t_acc[-1]+t_flat[-1]+t_dec])

        theta_vec = np.concatenate([theta_acc, theta_flat, theta_dec])

        return t_vec, theta_vec

    # ============================================================================
    #                     INVERSIONE θ(t) → t(θ)
    # ============================================================================
    def invert_theta(self, theta_vec, t_vec, theta_targets):
        return np.interp(theta_targets, theta_vec, t_vec)

    # ============================================================================
    #                    ANGOLO → IMPULSI ASSOLUTI
    # ============================================================================
    def convert_to_counts(self, theta):
        return np.round(theta * self.pulses_per_degree).astype(np.uint32)

    # ============================================================================
    #                          PIPELINE COMPLETA
    # ============================================================================
    def compute(self):

        # --- TIMBIR + ordinamento ---
        self.theta_interlaced = self.generate_timbir_angles()

        # --- TAXI MODEL ---
        t_vec, theta_vec = self.simulate_taxi_motion()

        # tempo in cui viene raggiunto ogni angolo
        t_real = self.invert_theta(theta_vec, t_vec, self.theta_interlaced)

        # angolo reale calcolato
        self.theta_interlaced_real = np.interp(t_real, t_vec, theta_vec)

        # --- IMPULSI ---
        self.pulses_interlaced_ideal = self.convert_to_counts(self.theta_interlaced)
        self.pulses_interlaced_real  = self.convert_to_counts(self.theta_interlaced_real)

        return self

    # ============================================================================
    #                    A) SCRITTURA pulses.bin
    # ============================================================================
    def save_pulses_bin(self, filename="pulses.bin", use_real=True):

        data = self.pulses_interlaced_real if use_real else self.pulses_interlaced_ideal

        with open(filename, "wb") as f:
            for val in data:
                # uint32 little-endian per FPGA
                f.write(struct.pack("<I", int(val)))

        print(f"\n✔ File '{filename}' salvato ({len(data)} impulsi).")

    # ============================================================================
    #             B) GRAFICI DIAGNOSTICI
    # ============================================================================
    def plot_diagnostics(self):

        err_deg = self.theta_interlaced_real - self.theta_interlaced
        err_counts = self.pulses_interlaced_real - self.pulses_interlaced_ideal
        t = np.arange(len(self.theta_interlaced))

        plt.figure(figsize=(14, 6))
        plt.plot(t, self.theta_interlaced, 'o-', label="Ideale (TIMBIR)")
        plt.plot(t, self.theta_interlaced_real, 'o-', label="Reale (Taxi)")
        plt.title("Angoli: ideale vs taxi-corretti")
        plt.xlabel("Indice acquisizione")
        plt.ylabel("Angolo (°)")
        plt.grid(); plt.legend()

        plt.figure(figsize=(14, 6))
        plt.plot(t, err_deg, 'o-')
        plt.title("Errore angolare (real - ideal)")
        plt.xlabel("Indice"); plt.ylabel("Errore (°)")
        plt.grid()

        plt.figure(figsize=(14, 6))
        plt.plot(t, err_counts, 'o-')
        plt.title("Errore impulsi (real - ideal)")
        plt.xlabel("Indice"); plt.ylabel("Counts")
        plt.grid()

        plt.show()

        print("\n✔ Grafici diagnostici generati.")

