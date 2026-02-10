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
        PSOPulsePerRotation=358818,           # quanti tick vede il contatore trigger *360
        RotationDirection=0,
        RotationAccelTime=0.15,
        ExposureTime=0.01,                   # r/w
        readout=0.01,
        readout_margin=1,
        SpeedDegPerSec=60.0,                 # r/w
        MinStepTarget=0.0,                   # r/w
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

        # TomoScan PV: ExposureTime
        self.ExposureTime = float(ExposureTime)

        self.readout = float(readout)
        self.readout_margin = float(readout_margin)

        # ----------------------------
        # PV calcolati
        # ----------------------------
        self.InterlacedRotationStop = None
        self.InterlacedMinStep = None
        self.InterlacedScanTime = None
        self.InterlacedEfficiency = None

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

        # altri placeholder usati in compute_positions_PSO
        self.PSOStartTaxi = None
        self.rotation_stop_new = None
        self.PSOEndTaxi = None
        self.theta_classic = None
        self.motor_speed = None

        # initialize derived PVs
        self._update_derived_pvs()

    # ----------------------------------------------------------------------
    # Derived PVs (r/w)
    # ----------------------------------------------------------------------
    
    def _update_derived_pvs(self):
        """
        Aggiorna i PV derivati:
          InterlacedRotationStop: stop teorico = start + K*360
          InterlacedRotationStepNominal: delta_theta_min() sugli angoli monotoni (se disponibili)
        """
        # stop teorico definibile
        self.stop_angle()

        # step nominale: se non hai theta_monotonic ancora, delta_theta_min torna 0
        self.InterlacedRotationStepNominal = float(self.delta_theta_min())

    def _update_interlaced_metrics(self):
        # Calcola min step e tempi/efficienza
        self._compute_interlaced_min_step()
        self._compute_interlaced_scan_time()

           # per tempo tempo totale e l'efficienza
        self._compute_interlaced_total_acq_time()
        self._compute_interlaced_efficiency()


    def stop_angle(self, metodo=""):
        """
        stop_theta = start_angle + numero di rotazioni * 360
        """
        start_angle = float(self.InterlacedRotationStart)
        numero_di_rotazioni = int(self.InterlacedNumberOfRotation)
        stop_theta = start_angle + numero_di_rotazioni * 360.0
        self.InterlacedRotationStop = float(stop_theta)
        return self.InterlacedRotationStop

    def delta_theta_min(self, metodo=""):
        if self.theta_monotonic is None:
            return 0.0

        theta = np.asarray(self.theta_monotonic, dtype=float)
        if theta.size < 2:
            return 0.0

        dtheta = np.diff(theta)
        dtheta = dtheta[dtheta > 0]

        if dtheta.size == 0:
            return 0.0

        return float(np.min(dtheta))

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
    def generate_interlaced_timbir(self):
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
                angle_deg = idx * 360.0 / (N * K)         # [0,360)
                angle_unwrapped = angle_deg + base_turn   # unwrapped fisico
                theta_acq.append(angle_unwrapped)

        theta_acq = np.asarray(theta_acq, dtype=float)

        self.theta_interlaced = theta_acq.astype(float)
        self.theta_interlaced_unwrapped = theta_acq.copy().astype(float)  # già unwrapped
        self.theta_monotonic = np.sort(self.theta_interlaced_unwrapped).astype(float)

        self._update_interlaced_metrics()

    def generate_interlaced_goldenangle(self):
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
        N = int(self.InterlacedNumAnglesPerRotation)
        K = int(self.InterlacedNumberOfRotation)
        start = float(self.InterlacedRotationStart)
        stop = float(self.InterlacedRotationStop) if self.InterlacedRotationStop is not None else self.stop_angle()

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
        N = int(self.InterlacedNumAnglesPerRotation)
        K = int(self.InterlacedNumberOfRotation)
        start = float(self.InterlacedRotationStart)
        stop = float(self.InterlacedRotationStop) if self.InterlacedRotationStop is not None else self.stop_angle()

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
        N = int(self.InterlacedNumAnglesPerRotation)
        K = int(self.InterlacedNumberOfRotation)
        start = float(self.InterlacedRotationStart)
        stop = float(self.InterlacedRotationStop) if self.InterlacedRotationStop is not None else self.stop_angle()

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
        # ExposureTime è il PV TomoScan
        return float(self.ExposureTime + self.readout)

    def compute_positions_PSO(self):
        # serve che lo stop sia definito
        if self.InterlacedRotationStop is None:
            self.stop_angle()

        overall_sense, user_direction = self.compute_senses()
        encoder_multiply = self.PSOCountsPerRotation / 360.0

        rotation_step = (
            float(self.InterlacedRotationStepNominal)
            if self.InterlacedRotationStepNominal is not None
            else 0.0
        )
        raw_counts = rotation_step * encoder_multiply
        delta_counts = round(raw_counts) if encoder_multiply != 0 else 0
        rotation_step = (delta_counts / encoder_multiply) if encoder_multiply != 0 else rotation_step

        self.InterlacedRotationStepNominal = rotation_step

        dt = self.compute_frame_time()
        self.motor_speed = abs(rotation_step) / dt if dt > 0 else 0.0

        accel_dist = 0.5 * self.motor_speed * self.RotationAccelTime

        if overall_sense > 0:
            rotation_start_new = float(self.InterlacedRotationStart)
        else:
            rotation_start_new = float(self.InterlacedRotationStart) - (2 - self.readout_margin) * rotation_step

        taxi_steps = (
            math.ceil((accel_dist / abs(rotation_step)) + 0.5)
            if abs(rotation_step) > 0
            else 0
        )
        taxi_dist = taxi_steps * abs(rotation_step)

        self.PSOStartTaxi = rotation_start_new - taxi_dist * user_direction
        self.rotation_stop_new = rotation_start_new + (self.InterlacedNumAnglesPerRotation - 1) * rotation_step
        self.PSOEndTaxi = self.rotation_stop_new + taxi_dist * user_direction

        # Angoli target per PSO: per interlaced devono essere quelli monotoni generati prima
        if self.theta_monotonic is None:
    raise ValueError("theta_monotonic non definito: chiama prima un generate_interlaced_*().")

self.theta_target = np.asarray(self.theta_monotonic, dtype=float)
pulses_per_degree = self.PSOCountsPerRotation / 360.0
counts = np.round(self.theta_target * pulses_per_degree).astype(np.int64)

# per PSO: monotoni e senza duplicati
counts = np.unique(np.sort(counts))

self.PSOCountsIdeal = counts







    def simulate_taxi_motion(self, omega_target=10, dt=1e-4):
        if self.theta_monotonic is None:
            raise ValueError("theta_monotonic non definito: chiama prima un generate_interlaced_*().")

        theta_max = float(np.max(np.asarray(self.theta_monotonic, dtype=float)))

        if self.RotationAccelTime <= 0:
            raise ValueError("RotationAccelTime deve essere > 0")

        accel = decel = float(omega_target) / float(self.RotationAccelTime)

        t_acc = np.arange(0, self.RotationAccelTime, dt)
        theta_acc = 0.5 * accel * t_acc ** 2
        theta_acc_end = float(theta_acc[-1]) if len(theta_acc) > 0 else 0.0

        theta_flat_len = theta_max - 2 * theta_acc_end
        if theta_flat_len < 0:
            raise ValueError("Profilo di moto non realizzabile (theta_max troppo piccolo vs accel/decel)")

        t_flat = np.arange(0, theta_flat_len / omega_target, dt) if omega_target > 0 else np.array([0.0])
        theta_flat = theta_acc_end + omega_target * t_flat

        t_dec = np.arange(0, self.RotationAccelTime, dt)
        last_flat = float(theta_flat[-1]) if len(theta_flat) > 0 else theta_acc_end
        theta_dec = last_flat + omega_target * t_dec - 0.5 * decel * t_dec ** 2

        self.theta_vec = np.concatenate([theta_acc, theta_flat, theta_dec]).astype(float)
        self.t_vec = np.concatenate([
            t_acc,
            (t_acc[-1] if len(t_acc) > 0 else 0.0) + t_flat,
            (t_acc[-1] if len(t_acc) > 0 else 0.0)
            + (t_flat[-1] if len(t_flat) > 0 else 0.0)
            + t_dec
        ]).astype(float)

    def compute_real_motion(self):
        if self.theta_vec is None or self.t_vec is None:
            raise ValueError("Profilo moto non definito: chiama simulate_taxi_motion() prima.")

        theta_target = np.asarray(self.theta_monotonic, dtype=float)
        self.t_real = np.interp(theta_target, self.theta_vec, self.t_vec).astype(float)
        self.theta_real = np.interp(self.t_real, self.t_vec, self.theta_vec).astype(float)

    def convert_angles_to_counts(self):
        pulses_per_degree = self.PSOCountsPerRotation / 360.0

        theta_target = np.asarray(self.theta_monotonic, dtype=float)
        self.PSOCountsIdeal = np.round(theta_target * pulses_per_degree).astype(int)

        if np.any(np.diff(self.PSOCountsIdeal) <= 0):
            print("WARNING: counts non strettamente crescenti (duplicati/inversioni).")

        if self.theta_real is None:
            raise ValueError("theta_real non definito: chiama compute_real_motion() prima.")

        self.PSOCountsTaxiCorrected = np.round(
            np.asarray(self.theta_real, dtype=float) * pulses_per_degree
        ).astype(int)

        # Counts finali (quelli che userai davvero)
        self.PSOCountsFinal = self.PSOCountsTaxiCorrected.copy()


        sep=" "

        # Stringa dei pulses corretti angoli taxi-corrected
        self.theta_monotonic_pulses_list = self.PSOCountsFinal.tolist()
        self.theta_monotonic_pulses = sep.join(map(str, self.theta_monotonic_pulses_list))

        return self.theta_monotonic_pulses

    # =========================================================
    # FUNZIONI ADDIZIONALI (metriche)
    # =========================================================
    def _compute_interlaced_min_step(self):
        if self.theta_monotonic is None or len(self.theta_monotonic) < 2:
            self.InterlacedMinStep = np.nan
            return self.InterlacedMinStep

        theta = np.asarray(self.theta_monotonic, dtype=float)
        d = np.diff(theta)
        d = d[d > 0]  # Tiene solo le differenze positive
        self.InterlacedMinStep = float(np.min(d)) if d.size else np.nan
        return self.InterlacedMinStep

    # quanto tempo ci mette la rotazione durante una scansione interlacciata:
    # del tempo della rotazione non del tempo totale dell’acquisizione
    def _compute_interlaced_scan_time(self):
        total_deg = float(self.InterlacedNumberOfRotation) * 360.0
        speed = float(self.SpeedDegPerSec)

        if speed <= 0:
            self.InterlacedScanTime = np.nan
        else:
            # tempo rotazione + accel/decel (stima semplice)
            self.InterlacedScanTime = float(total_deg / speed + 2.0 * max(0.0, self.RotationAccelTime))
        return self.InterlacedScanTime

    # stima del tempo totale dell’acquisizione in fly-scan:
    # tempo necessario a ruotare l’intero angolo richiesto
    def _compute_interlaced_total_acq_time(self):
        """
        tempo rotazione a velocità effettiva (limitata dalla camera) + accel/decel
        la velocità effettiva è min(velocità impostata, velocità massima sostenibile dalla camera)
        """
        # gradi totali da percorrere K·360°
        total_deg = float(self.InterlacedNumberOfRotation) * 360.0

        # velocità impostata dal PV
        v_cmd = float(self.SpeedDegPerSec)
        # velocita' deve essere =! 0 e positiva  ho agginto altro nuovo PV:  InterlacedTotalAcqTime
        if v_cmd <= 0 or total_deg <= 0:
            self.InterlacedTotalAcqTime = np.nan
            return self.InterlacedTotalAcqTime

        # vincolo camera: periodo frame minimo per singola vista
        #   frame_period = exposure + readout * margin
        frame_period = float(self.self.ExposureTime) + float(self.readout) * float(self.readout_margin)

        # se per qualche motivo è <= 0 uso la velocita' impostata dal PV , stima meccanica
        if frame_period <= 0:
            v_eff = v_cmd
        else:
            # passo angolare richiesto per frame: quanto ruoto per ogni frame
            step = float(getattr(self, "InterlacedMinStep", np.nan))  # quanti gradi tra una vista e la successiva
            if not np.isfinite(step) or step <= 0:
                # fallback ragionevole: passo nominale (360/N) per rotazione
                N = float(self.InterlacedNumAnglesPerRotation)
                step = 360.0 / N if N > 0 else np.nan

            if not np.isfinite(step) or step <= 0:
                self.InterlacedTotalAcqTime = np.nan
                return self.InterlacedTotalAcqTime

            # velocità max sostenibile dalla camera senza saltare frame
            v_cam_max = step / frame_period

            # velocità effettiva
            v_eff = min(v_cmd, v_cam_max)

        # tempo rotazione + accel/decel
        accel = 2.0 * max(0.0, float(self.RotationAccelTime))
        self.InterlacedTotalAcqTime = float(total_deg / v_eff + accel)
        return self.InterlacedTotalAcqTime

    # provo a campionare usando come passo base Δθ_min, quali degli angoli che mi interessano cadono nell'intervallo?
    # Un angolo è preso solo se è entro una tolleranza su un multiplo di dmin
    def _compute_interlaced_efficiency(self, tol=None, tol_frac=0.1):
        """
        Efficienza = percentuale di angoli che cadono su una intervallo a passo dmin.
        tol: tolleranza assoluta in gradi
        tol_frac: frazione di dmin usata come tolleranza se tol è None
        """
        if self.theta_monotonic is None:
            self.InterlacedEfficiency = None
            return None

        theta = np.asarray(self.theta_monotonic, dtype=float)
        if theta.size == 0:
            self.InterlacedEfficiency = None
            return None

        theta_rel = theta - float(theta[0])

        dmin = float(self.delta_theta_min())
        if dmin <= 0:
            self.InterlacedEfficiency = None
            return None

        # tolleranza per floating
        if tol is None:
            tol = tol_frac * dmin

        # indice del multiplo più vicino
        m = np.rint(theta_rel / dmin).astype(np.int64)

        # distanza dall'esatto multiplo
        theta_grid = m * dmin
        err = np.abs(theta_rel - theta_grid)

        taken_mask = err <= tol
        taken = int(np.count_nonzero(taken_mask))
        total = int(theta_rel.size)
        missed = total - taken

        eff = 100.0 * taken / total if total > 0 else np.nan
        missed_pct = 100.0 * missed / total if total > 0 else np.nan

        self.InterlacedEfficiency = dict(
            delta_theta_min_deg=float(dmin),
            tol_deg=float(tol),
            total_views=total,
            taken_views=taken,          # QUELLE CHE "CADONO" SU multipli di dmin
            missed_views=missed,        # QUELLE OFF-GRID => "non prese"
            efficiency_percent=float(eff),
            missed_percent=float(missed_pct),
            max_abs_error_deg=float(np.max(err)) if err.size else np.nan,
        )
        return self.InterlacedEfficiency

        # genero indici di trigger nel treno di impulsi dell’encoder


def theta_monotonic_pulses_to_trigger_positions(self, n_rotations, pulses_per_rotation, sep=" "):
    """
    metodo interlaced decide quali angoli-> impulsi da prendere -> convento in index assoluti
    PSO in rotazione continua, quindi indici in ordine temporale 0-based e crescenti

    Usa direttamente theta_monotonic_pulses e produce positions su N rotazioni con contatore assoluto
    """
    P = int(self.PSOPulsePerRotation)           # quanti tick vede il contatore per 360
    N = int(self.InterlacedNumberOfRotation)    # quante rotazioni fisiche sono ho posto k loops valuta se concettualmente corretto
    if P <= 0 or N <= 0:
        raise ValueError("pulses_per_rotation e n_rotations devono essere > 0")

    max_idx = N * P - 1     # per N giri con P tick/giro -> indice max N·P−1

    if not hasattr(self, "theta_monotonic_pulses") or self.theta_monotonic_pulses is None:
        raise ValueError("theta_monotonic_pulses non definito")

    s = str(self.theta_monotonic_pulses)
    pulses = np.asarray([int(x) for x in s.split() if x.strip()], dtype=np.int64)

    # contatore assoluto
    pulses = np.clip(pulses, 0, max_idx)

    # increasing e senza duplicati
    positions = np.unique(np.sort(pulses)).astype(np.int64)

    self.trigger_positions = positions.tolist()
    self.trigger_positions_str = sep.join(map(str, self.trigger_positions))
    return self.trigger_positions


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
    parser.add_argument("--min_step_target", type=float, default=0.0)
    parser.add_argument("--start", type=float, default=0.0)

    # TomoScan PV
    parser.add_argument("--ExposureTime", type=float, default=0.01)
    parser.add_argument("--readout", type=float, default=0.01)

    args = parser.parse_args()

    scan = InterlacedScan(
        InterlacedRotationStart=args.start,
        InterlacedNumberOfRotation=args.K_interlace,
        InterlacedNumAnglesPerRotation=args.num_angles,
        PSOCountsPerRotation=args.PSOCountsPerRotation,
        SpeedDegPerSec=args.speed,
        MinStepTarget=args.min_step_target,
        ExposureTime=args.ExposureTime,
        readout=args.readout,
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

    print("PSOCountsIdeal (first 10):", scan.PSOCountsIdeal[:10] if scan.PSOCountsIdeal is not None else None)
    print("PSOCountsFinal (first 10):", scan.PSOCountsFinal[:10] if scan.PSOCountsFinal is not None else None)


if __name__ == "__main__":
    main()
