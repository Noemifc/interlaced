"""
theta_interlaced           = angoli TIMBIR ordinati
theta_real_interlaced      = angoli TIMBIR corretti dal taxi model
pulses_interlaced_ideal    = conversione diretta
pulses_interlaced_real     = conversione taxi-corretta

"""
import numpy as np
import math

# ============================================================================
#                       TOMOSCAN INTERLACED OFFLINE (NO EPICS)
#            nomenclatura e semantica identica a TomoScanPSO
# ============================================================================

class TomoScanInterlacedOffline:

    def __init__(self,
                 rotation_start=0.0,
                 rotation_stop=360.0,
                 num_angles=32,
                 PSOCountsPerRotation=20000,
                 RotationDirection=0,
                 RotationAccelTime=0.15,
                 exposure=0.01,
                 readout=0.01,
                 readout_margin=1):

        # --- Parametri identici a TomoScanPSO ---
        self.rotation_start = rotation_start
        self.rotation_stop  = rotation_stop
        self.num_angles     = num_angles

        self.PSOCountsPerRotation = PSOCountsPerRotation
        self.RotationDirection = RotationDirection
        self.RotationAccelTime = RotationAccelTime

        self.exposure = exposure
        self.readout  = readout
        self.readout_margin = readout_margin

        # step angolare nominale (verrà aggiustato nella compute_positions_PSO)
        self.rotation_step = (rotation_stop - rotation_start) / (num_angles - 1)

        # campi riempiti più avanti
        self.theta_classic = None
        self.theta_interlaced = None


    # ------------------------------------------------------------------------
    # IDENTICO A TomoScanPSO.compute_senses()
    # ------------------------------------------------------------------------
    def compute_senses(self):
        encoder_dir = 1 if self.PSOCountsPerRotation > 0 else -1
        motor_dir   = 1 if self.RotationDirection == 0 else -1
        user_dir    = 1 if self.rotation_stop > self.rotation_start else -1
        overall = encoder_dir * motor_dir * user_dir
        return overall, user_dir
    

    # ------------------------------------------------------------------------
    def compute_frame_time(self):
        return self.exposure + self.readout


    # ------------------------------------------------------------------------
    # EQUIVALENTE A compute_positions_PSO DI TOMOSCANPSO (senza EPICS)
    # ------------------------------------------------------------------------
    def compute_positions_PSO(self):

        overall_sense, user_direction = self.compute_senses()
        encoder_multiply = self.PSOCountsPerRotation / 360.0

        # step angolare → impulsi interi
        raw_counts = self.rotation_step * encoder_multiply
        delta_counts = round(raw_counts)

        # aggiorno come TomoScanPSO
        self.rotation_step = delta_counts / encoder_multiply

        # velocità reale
        dt = self.compute_frame_time()
        self.motor_speed = abs(self.rotation_step) / dt

        # TAXI
        accel_dist = 0.5 * self.motor_speed * self.RotationAccelTime

        if overall_sense > 0:
            self.rotation_start_new = self.rotation_start
        else:
            self.rotation_start_new = self.rotation_start - (2 - self.readout_margin) * self.rotation_step

        taxi_steps = math.ceil((accel_dist / abs(self.rotation_step)) + 0.5)
        taxi_dist  = taxi_steps * abs(self.rotation_step)

        self.PSOStartTaxi = self.rotation_start_new - taxi_dist * user_direction

        self.rotation_stop_new = self.rotation_start_new + (self.num_angles - 1) * self.rotation_step
        self.PSOEndTaxi = self.rotation_stop_new + taxi_dist * user_direction

        # theta "classica" identica a TomoScan
        self.theta_classic = self.rotation_start_new + np.arange(self.num_angles) * self.rotation_step

        return self.theta_classic, self.PSOStartTaxi, self.PSOEndTaxi


# ============================================================================
#                      TIMBIR INTERLACED ANGLES
# ============================================================================

def bit_reverse(n, bits):
    return int(f"{n:0{bits}b}"[::-1], 2)

def generate_timbir_angles(N, K):
    bits = int(np.log2(N))
    return np.array([bit_reverse(n, bits) * 360.0 / N for n in range(N)])


# ============================================================================
#                      TAXI MODEL
# ============================================================================

def simulate_taxi_motion(accel, decel, omega_target, theta_total=360.0, dt=1e-4):

    T_acc = omega_target / accel
    t_acc = np.arange(0, T_acc, dt)
    theta_acc = 0.5 * accel * t_acc**2

    theta_flat_len = theta_total - 2 * theta_acc[-1]
    T_flat = theta_flat_len / omega_target
    t_flat = np.arange(0, T_flat, dt)
    theta_flat = theta_acc[-1] + omega_target * t_flat

    T_dec = omega_target / decel
    t_dec = np.arange(0, T_dec, dt)
    theta_dec = theta_flat[-1] + omega_target*t_dec - 0.5*decel*t_dec**2

    t = np.concatenate([t_acc, t_acc[-1] + t_flat, t_acc[-1] + t_flat[-1] + t_dec])
    theta = np.concatenate([theta_acc, theta_flat, theta_dec])

    return t, theta


def invert_theta(theta_real, t_vec, theta_targets):
    return np.interp(theta_targets, theta_real, t_vec)


# ============================================================================
#                IMPULSI — NOMENCLATURA TOMOSCAN STYLE
# ============================================================================

def convert_to_counts(theta, pulses_per_degree):
    return np.round(theta * pulses_per_degree).astype(int)


# ============================================================================
#                PIPELINE COMPLETA OFFLINE INTERLACED
# ============================================================================
def compute_interlaced_offline(N_theta, K,
                               RotationAccelTime=0.15,
                               PSOCountsPerRotation=20000,
                               omega_target=10):

    scan = TomoScanInterlacedOffline(
        num_angles=N_theta,
        RotationAccelTime=RotationAccelTime,
        PSOCountsPerRotation=PSOCountsPerRotation
    )

    # 1) Tomoscan-like parameters
    theta_classic, taxi_start, taxi_end = scan.compute_positions_PSO()

    pulses_per_degree = PSOCountsPerRotation / 360.0

    # 2) TIMBIR → angoli interlacciati ORDINATI
    theta_inter = generate_timbir_angles(N_theta, K)
    idx = np.argsort(theta_inter)
    theta_inter_sorted = theta_inter[idx]
    scan.theta_interlaced = theta_inter_sorted

    # 3) TAXI MODEL
    t, theta_real_curve = simulate_taxi_motion(
        accel=omega_target / RotationAccelTime,
        decel=omega_target / RotationAccelTime,
        omega_target=omega_target,
        theta_total=360.0
    )

    t_real = invert_theta(theta_real_curve, t, theta_inter_sorted)
    theta_real_corrected = np.interp(t_real, t, theta_real_curve)

    # 4) IMPULSI coerenti con TomoScanPSO
    PSOCountsIdeal = convert_to_counts(theta_inter_sorted, pulses_per_degree)
    PSOCountsTaxiCorrected = convert_to_counts(theta_real_corrected, pulses_per_degree)
    PSOCountsFinal = PSOCountsTaxiCorrected.copy()

    return {
        "theta_interlaced": theta_inter_sorted,
        "theta_real": theta_real_corrected,
        "PSOCountsIdeal": PSOCountsIdeal,
        "PSOCountsTaxiCorrected": PSOCountsTaxiCorrected,
        "PSOCountsFinal": PSOCountsFinal,
        "theta_classic": theta_classic,
        "PSOStartTaxi": taxi_start,
        "PSOEndTaxi": taxi_end
    }
