#!/usr/bin/env python3
import numpy as np
import math
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
        InterlacedRotationStart=0.0,          # r/w
        InterlacedNumberOfRotation=4,         # r/w  (K)
        InterlacedNumAnglesPerRotation=32,    # r/w  (N)
        PSOCountsPerRotation=20000,
        PSOPulsePerRotation=11840158,
        RotationDirection=0,
        RotationAccelTime=0.15,
        exposure=0.01,
        readout=0.01,
        readout_margin=1,
        SpeedDegPerSec=60.0,                  # r/w
        MinStepTarget=0.0,                    # r/w  
    ):
        # ----------------------------
        # PV-like (r/w)  
        # ----------------------------
        self.InterlacedRotationStart = float(InterlacedRotationStart)
        self.InterlacedNumberOfRotation = int(InterlacedNumberOfRotation)
        self.InterlacedNumAnglesPerRotation = int(InterlacedNumAnglesPerRotation)
        self.SpeedDegPerSec = float(SpeedDegPerSec)

        self.MinStepTarget = float(MinStepTarget)

        # ----------------------------
        # Hardware / camera
        # ----------------------------
        self.PSOCountsPerRotation = int(PSOCountsPerRotation)
        self.PSOPulsePerRotation = int(PSOPulsePerRotation)
        self.RotationDirection = int(RotationDirection)
        self.RotationAccelTime = float(RotationAccelTime)

        self.exposure = float(exposure)
        self.readout = float(readout)
        self.readout_margin = float(readout_margin)

        # ----------------------------
        # PV calcolati  
        # ----------------------------
        self.InterlacedRotationStop = None
        self.InterlacedMinStep = None
        self.InterlacedScanTime = None
        self.InterlacedEfficiency = None  # dict

        # step nominale - deve dipendere dal calcolo di dtheta
        self.InterlacedRotationStepNominal = None

        # ----------------------------
        # placeholders angoli
        # ----------------------------
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

        # initialize derived PVs  per aggiornare i PV quando vengono aggiornati
        self._update_derived_pvs()

    # ----------------------------------------------------------------------
    # Derived PVs (r\w)
    # ----------------------------------------------------------------------
    def _update_derived_pvs(self):
        """
        Aggiorna i PV derivati :
          1) InterlacedRotationStop = ultimo angolo effettivamente acquisito (theta_monotonic[-1]) oppure start_angle + numero di rotazioni * 360
          2) InterlacedRotationStepNominal = delta_theta_min() minimo dei delta theta tra angoli monotoni
        """
        # 1) STOP = ultimo angolo monotono

        # 2) STEP nominale = minimo delta_theta
        self.InterlacedRotationStepNominal = float(self.delta_theta_min())

    def stop_angle(self, metodo=""):
        """
        stop_theta = start stop_theta = start_angle + numero di rotazioni * 360
        """
        start_angle = float(self.InterlacedRotationStart)
        numero_di_rotazioni = int(self.InterlacedNumberOfRotation)

        stop_theta = start_angle + numero_di_rotazioni * 360.0

        # aggiorna il PV  
        self.InterlacedRotationStop = float(stop_theta)
        return self.InterlacedRotationStop

    def delta_theta_min(self, metodo=""):
        theta = np.asarray(self.theta_monotonic, dtype=float)
        if theta.size < 2:                                         # se hai almeno due angoli fai la differenza
            return 0.0

        dtheta = np.diff(theta)

        #  ignora eventuali zero  
        dtheta = dtheta[dtheta > 0]

        if dtheta.size == 0:
            return 0.0

        delta_theta_min = float(np.min(dtheta))
        return delta_theta_min

    # ----------------------------------------------------------------------
    # utility
    # ----------------------------------------------------------------------
    def bit_reverse(self, n, bits):
        return int(f"{n:0{bits}b}"[::-1], 2)

    def _ensure_power_of_two_K(self):
        K = int(self.InterlacedNumberOfRotation)
        assert (K & (K - 1)) == 0, "InterlacedNumberOfRotation (K) deve essere potenza di 2"

    # =========================================================
    # MODE
    # =========================================================
    """"self._update_interlaced_metrics()  aggiorna tutti i PVS, InterlacedMinStep
        InterlacedScanTime, InterlacedEfficiency
     """

    def generate_interlaced_timbir(self):
        self._update_derived_pvs()
        self._ensure_power_of_two_K()

        N = int(self.InterlacedNumAnglesPerRotation)
        K = int(self.InterlacedNumberOfRotation)
        bits = int(np.log2(K))

        theta_acq = []
        for n in range(N):
            group = (n * K // N) % K
            group_br = self.bit_reverse(group, bits)
            idx = n * K + group_br
            angle_deg = idx * 360.0 / N
            theta_acq.append(angle_deg)

        theta_acq = np.asarray(theta_acq, dtype=float)

        self.theta_interlaced = theta_acq.astype(float)
        self.theta_interlaced_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(theta_acq))).astype(float)

        # monotonic per PSO
        self.theta_monotonic = np.sort(self.theta_interlaced_unwrapped).astype(float)

        self._update_interlaced_metrics()

    def generate_interlaced_multitimbir(self):
        self._update_derived_pvs()
        self._ensure_power_of_two_K()

        N = int(self.InterlacedNumAnglesPerRotation)
        K = int(self.InterlacedNumberOfRotation)
        bits = int(np.log2(K))

        theta_acq = []
        for loop_turn in range(K):
            base_turn = 360.0 * loop_turn
            subloop = self.bit_reverse(loop_turn, bits)

            for i in range(N):
                idx = i * K + subloop
                angle_deg = idx * 360.0 / (N * K)       # [0,360)
                angle_unwrapped = angle_deg + base_turn  # unwrapped fisico
                theta_acq.append(angle_unwrapped)

        theta_acq = np.asarray(theta_acq, dtype=float)

        self.theta_interlaced = theta_acq.astype(float)
        self.theta_interlaced_unwrapped = theta_acq.copy().astype(float)  # già unwrapped
        self.theta_monotonic = np.sort(self.theta_interlaced_unwrapped).astype(float)

        self._update_interlaced_metrics()

    def generate_interlaced_goldenangle(self):
        self._update_derived_pvs()

        N = int(self.InterlacedNumAnglesPerRotation)
        K = int(self.InterlacedNumberOfRotation)
        start = float(self.InterlacedRotationStart)

        golden_angle = 360.0 * (3.0 - np.sqrt(5.0)) / 2.0
        phi_inv = (np.sqrt(5.0) - 1.0) / 2.0

        angles_all = []
        base = np.array([(start + i * golden_angle) % 360.0 for i in range(N)], dtype=float)
        base = np.sort(base)
        angles_all.append(base)

        for k in range(1, K):
            offset = (k / (N + 1.0)) * 360.0 * phi_inv
            angles_all.append(np.sort((base + offset) % 360.0))

        theta_acq = np.concatenate(angles_all).astype(float)

        self.theta_interlaced = theta_acq.astype(float)
        self.theta_interlaced_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(theta_acq))).astype(float)
        self.theta_monotonic = np.sort(self.theta_interlaced_unwrapped).astype(float)

        self._update_interlaced_metrics()
        return angles_all

    def generate_interlaced_kturns(self, delta_theta=None):
        self._update_derived_pvs()

        N = int(self.InterlacedNumAnglesPerRotation)
        K = int(self.InterlacedNumberOfRotation)
        start = float(self.InterlacedRotationStart)
        stop = float(self.InterlacedRotationStop)

        if delta_theta is None:
            delta_theta = (stop - start) / (N - 1) if N > 1 else 0.0
        delta_theta = float(delta_theta)

        self.InterlacedRotationStepNominal = delta_theta

        base = start + np.arange(N, dtype=float) * delta_theta
        angles_all = [base + 360.0 * k for k in range(K)]
        theta_unwrapped_acq = np.concatenate(angles_all).astype(float)

        self.theta_interlaced = theta_unwrapped_acq.astype(float)
        self.theta_interlaced_unwrapped = theta_unwrapped_acq.astype(float)
        self.theta_monotonic = np.sort(self.theta_interlaced_unwrapped).astype(float)

        self._update_interlaced_metrics()
        return angles_all

    def generate_interlaced_multiturns(self, delta_theta=None):
        self._update_derived_pvs()

        N = int(self.InterlacedNumAnglesPerRotation)
        K = int(self.InterlacedNumberOfRotation)
        start = float(self.InterlacedRotationStart)
        stop = float(self.InterlacedRotationStop)

        if delta_theta is None:
            delta_theta = (stop - start) / (N - 1) if N > 1 else 0.0
        delta_theta = float(delta_theta)
        self.InterlacedRotationStepNominal = delta_theta

        n = np.arange(N, dtype=float)
        angles_all = []
        for k in range(K):
            theta_n = start + (n + k / K) * delta_theta
            angles_all.append(theta_n)

        theta_unwrapped_acq = np.concatenate(angles_all).astype(float)
        self.theta_interlaced = theta_unwrapped_acq.astype(float)
        self.theta_interlaced_unwrapped = theta_unwrapped_acq.astype(float)
        self.theta_monotonic = np.sort(self.theta_interlaced_unwrapped).astype(float)

        self._update_interlaced_metrics()
        return angles_all

    def generate_interlaced_corput(self, delta_theta=None):
        self._update_derived_pvs()

        N = int(self.InterlacedNumAnglesPerRotation)
        K = int(self.InterlacedNumberOfRotation)
        start = float(self.InterlacedRotationStart)
        stop = float(self.InterlacedRotationStop)

        if delta_theta is None:
            delta_theta = (stop - start) / (N - 1) if N > 1 else 0.0
        delta_theta = float(delta_theta)
        self.InterlacedRotationStepNominal = delta_theta

        base = start + np.arange(N, dtype=float) * delta_theta

        bitsK = int(np.ceil(np.log2(K)))
        MK = 1 << bitsK
        p_corput = np.array([self.bit_reverse(i, bitsK) for i in range(MK)])
        p_corput = p_corput[p_corput < K]
        assert len(p_corput) == K

        offsets = (p_corput / K) * delta_theta

        bitsN = int(np.ceil(np.log2(N)))
        MN = 1 << bitsN
        indices = np.array([self.bit_reverse(i, bitsN) for i in range(MN)])
        indices = indices[indices < N]

        angles_all = []
        for k in range(K):
            offset = offsets[k]
            loop_angles = base[indices] + offset

            loop_angles_mod = np.mod(loop_angles - start, 360.0) + start
            loop_angles_unwrapped = loop_angles_mod + 360.0 * k
            angles_all.append(loop_angles_unwrapped)

        theta_unwrapped_unsorted = np.concatenate(angles_all)
        theta_unwrapped = np.sort(theta_unwrapped_unsorted)

        self.theta_interlaced = np.mod(theta_unwrapped_unsorted - start, 360.0) + start
        self.theta_interlaced_unwrapped = theta_unwrapped.astype(float)
        self.theta_monotonic = np.sort(self.theta_interlaced_unwrapped).astype(float)

        self._update_interlaced_metrics()
        return angles_all
                          

    # =========================================================
    # FUNZIONI (motion + PSO)
    # =========================================================
    
    def compute_senses(self):
        encoder_dir = 1 if self.PSOCountsPerRotation > 0 else -1
        motor_dir = 1 if self.RotationDirection == 0 else -1
        user_dir = 1 if self.InterlacedRotationStop > self.InterlacedRotationStart else -1
        return encoder_dir * motor_dir * user_dir, user_dir

    def compute_frame_time(self):
        return float(self.exposure + self.readout)

    def compute_positions_PSO(self):
        overall_sense, user_direction = self.compute_senses()
        encoder_multiply = self.PSOCountsPerRotation / 360.0

        rotation_step = float(self.InterlacedRotationStepNominal)
        raw_counts = rotation_step * encoder_multiply
        delta_counts = round(raw_counts)
        rotation_step = delta_counts / encoder_multiply

        self.InterlacedRotationStepNominal = rotation_step

        dt = self.compute_frame_time()
        self.motor_speed = abs(rotation_step) / dt

        accel_dist = 0.5 * self.motor_speed * self.RotationAccelTime

        if overall_sense > 0:
            rotation_start_new = float(self.InterlacedRotationStart)
        else:
            rotation_start_new = float(self.InterlacedRotationStart) - (2 - self.readout_margin) * rotation_step

        taxi_steps = math.ceil((accel_dist / abs(rotation_step)) + 0.5)
        taxi_dist = taxi_steps * abs(rotation_step)

        self.PSOStartTaxi = rotation_start_new - taxi_dist * user_direction
        self.rotation_stop_new = rotation_start_new + (self.InterlacedNumAnglesPerRotation - 1) * rotation_step
        self.PSOEndTaxi = self.rotation_stop_new + taxi_dist * user_direction

        self.theta_classic = rotation_start_new + np.arange(
            self.InterlacedNumAnglesPerRotation, dtype=float
        ) * rotation_step

    def simulate_taxi_motion(self, omega_target=10, dt=1e-4):
        theta_max = float(np.max(np.asarray(self.theta_monotonic, dtype=float)))

        accel = decel = float(omega_target) / float(self.RotationAccelTime)

        t_acc = np.arange(0, self.RotationAccelTime, dt)
        theta_acc = 0.5 * accel * t_acc ** 2
        theta_acc_end = float(theta_acc[-1]) if len(theta_acc) > 0 else 0.0

        theta_flat_len = theta_max - 2 * theta_acc_end
        if theta_flat_len < 0:
            raise ValueError("Profilo di moto non realizzabile (theta_max troppo piccolo rispetto accel/decel)")

        t_flat = np.arange(0, theta_flat_len / omega_target, dt) if omega_target > 0 else np.array([0.0])
        theta_flat = theta_acc_end + omega_target * t_flat

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

        theta_target = np.asarray(self.theta_monotonic, dtype=float)
        self.PSOCountsIdeal = np.round(theta_target * pulses_per_degree).astype(int)

        if np.any(np.diff(self.PSOCountsIdeal) <= 0):
            print("WARNING: counts non strettamente crescenti (duplicati/inversioni).")

        self.PSOCountsTaxiCorrected = np.round(
            np.asarray(self.theta_real, dtype=float) * pulses_per_degree
        ).astype(int)
        self.PSOCountsFinal = self.PSOCountsTaxiCorrected.copy()

    # =========================================================
    # FUNZIONI ADDIZIONALI
    # =========================================================
    def _compute_interlaced_min_step(self):
        if self.theta_monotonic is None or len(self.theta_monotonic) < 2:
            self.InterlacedMinStep = np.nan
            return self.InterlacedMinStep

        theta = np.asarray(self.theta_monotonic, dtype=float)
        d = np.diff(theta)
        d = d[d > 0]
        self.InterlacedMinStep = float(np.min(d)) if d.size else np.nan
        return self.InterlacedMinStep

    def _compute_interlaced_scan_time(self):
        total_deg = float(self.InterlacedNumberOfRotation) * 360.0
        speed = float(self.SpeedDegPerSec)
        if speed <= 0:
            self.InterlacedScanTime = np.nan
        else:
            self.InterlacedScanTime = float(total_deg / speed + 2.0 * max(0.0, self.RotationAccelTime))
        return self.InterlacedScanTime

    def _compute_interlaced_efficiency(self):
        step = float(self.MinStepTarget)
        if self.theta_monotonic is None or step <= 0:
            self.InterlacedEfficiency = None
            return None

        total_deg = float(self.InterlacedNumberOfRotation) * 360.0
        required_views = int(np.ceil(total_deg / step))

        theta = np.asarray(self.theta_monotonic, dtype=float)
        theta_rel = theta - float(theta[0])
        theta_rel = theta_rel[(theta_rel >= 0) & (theta_rel < total_deg)]

        bins = np.floor(theta_rel / step).astype(np.int64)
        bins = bins[(bins >= 0) & (bins < required_views)]
        covered = np.unique(bins).size

        missing = required_views - covered
        eff = covered / required_views if required_views > 0 else np.nan

        self.InterlacedEfficiency = dict(
            required_views=int(required_views),
            collected_bins=int(covered),
            missing_views=int(missing),
            efficiency=float(eff),
        )
        return self.InterlacedEfficiency

    def _update_interlaced_metrics(self):
        # sempre coerente con i PV r/w
        self._update_derived_pvs()

        # metriche
        self._compute_interlaced_min_step()
        self._compute_interlaced_scan_time()
        if self.MinStepTarget > 0:
            self._compute_interlaced_efficiency()
        else:
            self.InterlacedEfficiency = None


# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run interlaced scan simulation (PV-only names).")
    parser.add_argument("--num_angles", type=int, default=32)
    parser.add_argument("--K_interlace", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=["timbir", "multitimbir", "golden", "kturns", "multiturns", "corput"],
        default="timbir"
    )
    parser.add_argument("--PSOCountsPerRotation", type=int, default=20000)

    # PV-like
    parser.add_argument("--speed", type=float, default=90.0)
    parser.add_argument("--indexmax", type=int, default=550000)  # NON USATO
    parser.add_argument("--min_step_target", type=float, default=0.0)
    parser.add_argument("--start", type=float, default=0.0)

    args = parser.parse_args()

    scan = InterlacedScan(
        InterlacedRotationStart=args.start,
        InterlacedNumberOfRotation=args.K_interlace,
        InterlacedNumAnglesPerRotation=args.num_angles,
        PSOCountsPerRotation=args.PSOCountsPerRotation,
        SpeedDegPerSec=args.speed,
        MinStepTarget=args.min_step_target
    )

    if args.mode == "timbir":
        scan.generate_interlaced_timbir()
    elif args.mode == "multitimbir":
        scan.generate_interlaced_multitimbir()
    elif args.mode == "golden":
        scan.generate_interlaced_goldenangle()
    elif args.mode == "kturns":
        scan.generate_interlaced_kturns()
    elif args.mode == "multiturns":
        scan.generate_interlaced_multiturns()
    elif args.mode == "corput":
        scan.generate_interlaced_corput()

    # stampa PV calcolati
    print("InterlacedRotationStart:", scan.InterlacedRotationStart)
    print("InterlacedNumberOfRotation:", scan.InterlacedNumberOfRotation)
    print("InterlacedNumAnglesPerRotation:", scan.InterlacedNumAnglesPerRotation)
    print("InterlacedRotationStop:", scan.InterlacedRotationStop)
    print("InterlacedMinStep:", scan.InterlacedMinStep)
    print("InterlacedScanTime:", scan.InterlacedScanTime)
    print("InterlacedEfficiency:", scan.InterlacedEfficiency)

    # motion / counts
    scan.compute_positions_PSO()
    scan.simulate_taxi_motion()
    scan.compute_real_motion()
    scan.convert_angles_to_counts()


if __name__ == "__main__":
    main()
