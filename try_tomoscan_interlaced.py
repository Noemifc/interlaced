"""
theta_interlaced           = angoli TIMBIR ordinati
theta_real_interlaced      = angoli TIMBIR corretti dal taxi model
pulses_interlaced_ideal    = conversione diretta
pulses_interlaced_real     = conversione taxi-corretta

"""
import numpy as np
import struct
import math
import matplotlib.pyplot as plt


# ============================================================================
#               CLASSE InterlacedScan 
# ============================================================================

class InterlacedScan:

    """
    Questa classe implementa una pipeline completa per una scansione TIMBIR:

    1) Genera angoli TIMBIR usando bit-reversal.
    2) Ordina gli angoli in senso crescente (richiesto dalla FPGA).
    3) Simula il moto reale del piatto (modello TAXI):
         - accelerazione
         - moto a velocità costante
         - decelerazione
    4) Calcola l'angolo effettivamente raggiunto nel tempo (θ_real).
    5) Converte angoli ideali e reali in impulsi encoder assoluti.
    6) Scrive il file pulses.bin pronto da caricare sulla FPGA.
    7) Produce grafici diagnostici per verificare eventuali differenze.

    Tutto è commentato e indipendente da Tomoscan/EPICS.
    """

    # ----------------------------------------------------------------------
    def __init__(self,
                 N_theta=32,                   # numero di proiezioni
                 K=4,                          # interlacciamento TIMBIR
                 PSOCountsPerRotation=20000,   # impulsi encoder per giro
                 accel=5,                      # accelerazione (deg/s^2)
                 decel=5,                      # decelerazione (deg/s^2)
                 omega_target=10,              # velocità target (deg/s)
                 dt=1e-4):                     # risoluzione temporale per simulazione

        # Parametri della scansione TIMBIR
        self.N_theta = N_theta
        self.K = K

        # Parametri encoder
        self.PSOCountsPerRotation = PSOCountsPerRotation
        self.pulses_per_degree = PSOCountsPerRotation / 360.0

        # Parametri dinamica TAXI
        self.accel = accel
        self.decel = decel
        self.omega_target = omega_target
        self.dt = dt

    # ============================================================================
    #                    BIT–REVERSAL → SEQUENZA TIMBIR
    # ============================================================================
    def bit_reverse(self, n, bits):
        """
        Inverte i bit di n.
        Es: 3 (0011) → 12 (1100).
        """
        b = f'{n:0{bits}b}'
        return int(b[::-1], 2)

    def generate_timbir_angles(self):
        """
        Genera gli angoli TIMBIR in gradi.
        - Usa bit-reversal
        - Poi li ORDINA in ordine crescente (richiesto dalla FPGA)
        """
        bits = int(np.log2(self.N_theta))

        # Genera la sequenza TIMBIR non ordinata
        theta = np.array([
            self.bit_reverse(n, bits) * 360.0 / self.N_theta
            for n in range(self.N_theta)
        ])

        # Ordine crescente per generare una lista monotòna
        return np.sort(theta)

    # ============================================================================
    #                        SIMULAZIONE MOTORE (TAXI)
    # ============================================================================
    def simulate_taxi_motion(self):
        """
        Simula l'intero moto del piatto:
        - Accelerazione da 0 a velocità target
        - Moto a velocità costante
        - Decelerazione fino a fermarsi

        Restituisce:
        - t_vec     : vettore dei tempi
        - theta_vec : angolo reale raggiunto ad ogni istante
        """

        accel = self.accel
        decel = self.decel
        omega_target = self.omega_target
        dt = self.dt
        theta_total = 360.0

        # --- Fase di accelerazione ---
        T_acc = omega_target / accel
        t_acc = np.arange(0, T_acc, dt)
        theta_acc = 0.5 * accel * t_acc**2

        # --- Plateau a velocità costante ---
        theta_flat_len = theta_total - 2 * theta_acc[-1]  # angolo per moto rettilineo
        T_flat = theta_flat_len / omega_target
        t_flat = np.arange(0, T_flat, dt)
        theta_flat = theta_acc[-1] + omega_target * t_flat

        # --- Decelerazione ---
        T_dec = omega_target / decel
        t_dec = np.arange(0, T_dec, dt)
        theta_dec = theta_flat[-1] + omega_target*t_dec - 0.5*decel*t_dec**2

        # --- Vettori completi ---
        t_vec = np.concatenate([
            t_acc,
            t_acc[-1] + t_flat,
            t_acc[-1] + t_flat[-1] + t_dec
        ])

        theta_vec = np.concatenate([
            theta_acc,
            theta_flat,
            theta_dec
        ])

        return t_vec, theta_vec

    # ============================================================================
    #                   INVERSIONE θ(t) → t(θ)
    # ============================================================================
    def invert_theta(self, theta_vec, t_vec, theta_targets):
        """
        Trova il tempo t per cui θ(t) = θ_target.
        Usa interpolazione inversa.
        """
        return np.interp(theta_targets, theta_vec, t_vec)

    # ============================================================================
    #                   ANGOLO → IMPULSI ABSOLUTI
    # ============================================================================
    def convert_to_counts(self, theta):
        """
        Converte angolo in impulsi PSO da mandare alla FPGA.
        Impulsi assoluti (non differenze).
        """
        return np.round(theta * self.pulses_per_degree).astype(np.uint32)

    # ============================================================================
    #                         PIPELINE COMPLETA
    # ============================================================================
    def compute(self):
        """
        Esegue l'intera pipeline TIMBIR + TAXI + conversione impulsi.
        """

        # 1) Angoli TIMBIR ordinati
        self.theta_interlaced = self.generate_timbir_angles()

        # 2) Simulazione del moto TAXI
        t_vec, theta_vec = self.simulate_taxi_motion()

        # 3) Trovo il tempo in cui viene raggiunto ogni angolo TIMBIR
        t_real = self.invert_theta(theta_vec, t_vec, self.theta_interlaced)

        # 4) Calcolo l'angolo reale (corretto)
        self.theta_interlaced_real = np.interp(t_real, t_vec, theta_vec)

        # 5) Conversione in impulsi PSO
        self.pulses_interlaced_ideal = self.convert_to_counts(self.theta_interlaced)
        self.pulses_interlaced_real  = self.convert_to_counts(self.theta_interlaced_real)

        return self

    # ============================================================================
    #                           A) GENERA pulses.bin
    # ============================================================================
    def save_pulses_bin(self, filename="pulses.bin", use_real=True):
        """
        Scrive direttamente il file pulses.bin compatibile con FPGA.
        - uint32
        - little-endian
        """

        data = self.pulses_interlaced_real if use_real else self.pulses_interlaced_ideal

        with open(filename, "wb") as f:
            for val in data:
                f.write(struct.pack("<I", int(val)))

        print(f"\n✔ File '{filename}' generato ({len(data)} impulsi).")

    # ============================================================================
    #                           B) GRAFICI DIAGNOSTICI
    # ============================================================================
    def plot_diagnostics(self):
        """
        Produce tre grafici:
        1) Angolo ideale vs angolo reale
        2) Errore angolare in gradi
        3) Errore sugli impulsi
        """

        err_deg = self.theta_interlaced_real - self.theta_interlaced
        err_counts = self.pulses_interlaced_real - self.pulses_interlaced_ideal
        t = np.arange(len(self.theta_interlaced))

        # --- Grafico angoli ---
        plt.figure(figsize=(14, 6))
        plt.plot(t, self.theta_interlaced, 'o-', label="Ideale (TIMBIR)")
        plt.plot(t, self.theta_interlaced_real, 'o-', label="Reale (Taxi)")
        plt.title("Angoli: Ideale vs Reale (Taxi Corretti)")
        plt.xlabel("Indice acquisizione")
        plt.ylabel("Angolo (°)")
        plt.grid(True); plt.legend()

        # --- Errore angolare ---
        plt.figure(figsize=(14, 5))
        plt.plot(t, err_deg, 'o-')
        plt.title("Errore angolare (reale - ideale)")
        plt.xlabel("Indice"); plt.ylabel("Errore (°)")
        plt.grid(True)

        # --- Errore impulsi ---
        plt.figure(figsize=(14, 5))
        plt.plot(t, err_counts, 'o-')
        plt.title("Errore impulsi (reale - ideale)")
        plt.xlabel("Indice"); plt.ylabel("Errore (counts)")
        plt.grid(True)

        plt.show()
        print("\n✔ Grafici diagnostici generati.")
