#!/usr/bin/env python3
import numpy as np
import argparse

from Tomoscan_PSOtoFPGA import InterlacedScan


class Dose:
    """
    Dose calculator for TomoScan/Interlaced fly scans.

    It computes:
      - Relative dose (dimensionless, proportional units)
      - Absolute dose (Gy) if you provide a dose rate calibration (Gy/s) or (Gy per monitor-unit)

    Key idea:
      Dose ~ ∫ I(t) dt  (energy deposition is proportional to fluence for fixed beam/setup)
      If intensity is constant: Dose_rel ~ sum(exposure_time_per_frame)
      If intensity varies:      Dose_rel ~ sum(I_i * exposure_time_per_frame)

    The class tries to get the time-ordered theta list from InterlacedScan (if available),
    but it can also work without it.
    """

    # ----------------------------------------------------------------------
    # init and parameters
    # ----------------------------------------------------------------------
    def __init__(
        self,
        InterlacedRotationStart=0.0,
        InterlacedNumberOfRotation=4,          # K
        InterlacedNumAnglesPerRotation=32,     # N
        PSOCountsPerRotation=20000,
        PSOPulsePerRotation=358818,
        RotationDirection=0,
        RotationAccelTime=0.15,
        ExposureTime=0.01,
        readout=0.01,
        readout_margin=1.0,
        SpeedDegPerSec=60.0,
        MinStepTarget=0.0,
        # Dose/monitor options
        frame_time_mode="exposure",            # "exposure" or "exposure+readout"
    ):
        # ----------------------------
        # PV-like (r/w)
        # ----------------------------
        self.InterlacedRotationStart = float(InterlacedRotationStart)
        self.InterlacedNumberOfRotation = int(InterlacedNumberOfRotation)
        self.InterlacedNumAnglesPerRotation = int(InterlacedNumAnglesPerRotation)
        self.PSOCountsPerRotation = int(PSOCountsPerRotation)
        self.PSOPulsePerRotation = int(PSOPulsePerRotation)

        self.RotationDirection = int(RotationDirection)
        self.RotationAccelTime = float(RotationAccelTime)

        self.ExposureTime = float(ExposureTime)
        self.readout = float(readout)
        self.readout_margin = float(readout_margin)
        self.SpeedDegPerSec = float(SpeedDegPerSec)
        self.MinStepTarget = float(MinStepTarget)
        self.frame_time_mode = str(frame_time_mode).strip().lower()

        # Under the hood: instantiate interlaced scan generator
        self.scan = InterlacedScan(
            InterlacedRotationStart=self.InterlacedRotationStart,
            InterlacedNumberOfRotation=self.InterlacedNumberOfRotation,
            InterlacedNumAnglesPerRotation=self.InterlacedNumAnglesPerRotation,
            PSOCountsPerRotation=self.PSOCountsPerRotation,
            PSOPulsePerRotation=self.PSOPulsePerRotation,
            RotationDirection=self.RotationDirection,
            RotationAccelTime=self.RotationAccelTime,
            ExposureTime=self.ExposureTime,
            readout=self.readout,
            readout_margin=self.readout_margin,
            SpeedDegPerSec=self.SpeedDegPerSec,
            MinStepTarget=self.MinStepTarget,
        )

        # Precompute time-ordered theta sequence if available
        self.theta = self._get_theta_sequence()

        # Precompute delta-theta mins if available
        self.theta_min = self._get_theta_min()

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _get_theta_sequence(self):
        """
        Try to fetch time-ordered thetas from InterlacedScan.
        Works with a few possible APIs (attribute or method).
        """
        # 1) if already computed as attribute
        for attr in ("theta_monotonic", "theta", "theta_list", "theta_sequence"):
            if hasattr(self.scan, attr):
                val = getattr(self.scan, attr)
                if val is not None and np.size(val) > 0:
                    return np.asarray(val, dtype=float)

        # 2) common method names
        for m in ("compute_theta_monotonic", "get_theta_monotonic", "theta_monotonic_list"):
            if hasattr(self.scan, m) and callable(getattr(self.scan, m)):
                val = getattr(self.scan, m)()
                if val is not None and np.size(val) > 0:
                    return np.asarray(val, dtype=float)

        # 3) fallback: uniform angles for K rotations
        # This is only a fallback so the dose math still runs.
        K = self.InterlacedNumberOfRotation
        N = self.InterlacedNumAnglesPerRotation
        total = K * N
        # Uniform over 360 (single-rotation sampling), repeated across K rotations:
        return np.linspace(self.InterlacedRotationStart, self.InterlacedRotationStart + 360.0, total, endpoint=False)

    def _get_theta_min(self):
        """
        Try to read a delta-theta-min list from InterlacedScan if it exists.
        Otherwise estimate it from theta sequence.
        """
        for attr in ("delta_theta_min_list", "theta_min_list", "delta_theta_min"):
            if hasattr(self.scan, attr):
                val = getattr(self.scan, attr)
                if val is not None and np.size(val) > 0:
                    return np.asarray(val, dtype=float)

        # If scan has a method delta_theta_min(), call it if it returns something usable.
        if hasattr(self.scan, "delta_theta_min") and callable(self.scan.delta_theta_min):
            try:
                val = self.scan.delta_theta_min()
                if val is not None and np.size(val) > 0:
                    return np.asarray(val, dtype=float)
            except Exception:
                pass

        # fallback: use consecutive diffs (absolute, ignoring wrap)
        th = np.asarray(self.theta, dtype=float)
        if th.size < 2:
            return np.array([], dtype=float)
        diffs = np.abs(np.diff(th))
        # protect against 360-wrap if theta jumps:
        diffs = np.minimum(diffs, np.abs(diffs - 360.0))
        return diffs

    def frame_time(self):
        """
        Time that contributes to dose per projection.
        - If shutter open only during exposure -> use ExposureTime
        - If beam stays on during readout -> include readout * margin
        """
        if self.frame_time_mode in ("exposure+readout", "total", "full"):
            return float(self.ExposureTime + self.readout_margin * self.readout)
        return float(self.ExposureTime)

    # ----------------------------------------------------------------------
    # Core: Relative dose
    # ----------------------------------------------------------------------
    def relative_dose(self, intensity=None):
        """
        Compute relative dose (arbitrary units).
        intensity:
          - None: assume constant intensity -> Dose_rel = sum(frame_time)
          - scalar: constant intensity -> Dose_rel = I * sum(frame_time)
          - array length Nframes: per-frame intensity samples -> Dose_rel = sum(I_i * frame_time_i)

        Returns dict with:
          total_rel
          rel_cumulative (array over frames)
          time_axis (s)
        """
        n = int(np.size(self.theta))
        if n <= 0:
            raise ValueError("No theta frames available (theta list empty).")

        dt = self.frame_time()
        t = dt * np.arange(n, dtype=float)  # time at frame start (simple model)

        if intensity is None:
            I = np.ones(n, dtype=float)
        else:
            intensity = np.asarray(intensity, dtype=float)
            if intensity.size == 1:
                I = np.full(n, float(intensity), dtype=float)
            elif intensity.size == n:
                I = intensity
            else:
                raise ValueError(f"intensity must be None, scalar, or length {n}. Got length {intensity.size}.")

        # Dose per frame proportional to I * dt
        d = I * dt
        cum = np.cumsum(d)
        total = float(cum[-1])

        return {
            "total_rel": total,
            "rel_cumulative": cum,
            "time_axis_s": t,
            "frame_time_s": dt,
            "n_frames": n,
        }

    def relative_dose_early(self, T=None, m=None, intensity=None):
        """
        Relative dose within:
          - first T seconds, OR
          - first m frames

        Provide exactly one between T and m.
        """
        res = self.relative_dose(intensity=intensity)
        cum = res["rel_cumulative"]
        dt = res["frame_time_s"]
        n = res["n_frames"]

        if (T is None) == (m is None):
            raise ValueError("Provide exactly one: T (seconds) or m (frames).")

        if T is not None:
            T = float(T)
            if T < 0:
                return 0.0
            idx = int(np.floor(T / dt))
            idx = max(0, min(idx, n - 1))
            return float(cum[idx])

        m = int(m)
        if m <= 0:
            return 0.0
        idx = min(m - 1, n - 1)
        return float(cum[idx])

    def relative_dose_normalized_curve(self, intensity=None):
        """
        Returns normalized cumulative curve: cum / cum[-1], goes 0->1
        Useful to compare different interlacing patterns ("how early you delivered dose").
        """
        res = self.relative_dose(intensity=intensity)
        cum = res["rel_cumulative"]
        total = res["total_rel"]
        if total <= 0:
            return np.zeros_like(cum)
        return cum / total

    # ----------------------------------------------------------------------
    # Optional: simulate shutter open/closed (simple model)
    # ----------------------------------------------------------------------
    def simulate_shutter(self, shutter_mode="open_after_accel"):
        """
        Returns a boolean mask (length Nframes) indicating whether shutter is open for that frame.

        shutter_mode:
          - "always_open": shutter open for all frames (typical continuous beam)
          - "open_after_accel": shutter opens after RotationAccelTime (simplified)
        """
        n = int(np.size(self.theta))
        dt = self.frame_time()
        t_mid = (np.arange(n, dtype=float) + 0.5) * dt

        shutter_mode = str(shutter_mode).strip().lower()
        if shutter_mode == "always_open":
            return np.ones(n, dtype=bool)

        if shutter_mode == "open_after_accel":
            return t_mid >= float(self.RotationAccelTime)

        raise ValueError(f"Unknown shutter_mode={shutter_mode}")

    # ----------------------------------------------------------------------
    # Absolute dose (Gy): needs calibration
    # ----------------------------------------------------------------------
    def absolute_dose(self, dose_rate_Gy_per_s=None, intensity=None, shutter_mask=None):
        """
        Compute absolute dose (Gy) with a calibration for dose rate.

        dose_rate_Gy_per_s:
          - scalar Gy/s for unit intensity (or absolute Gy/s if intensity=None)
          - If intensity is provided in "monitor units", interpret dose_rate_Gy_per_s as Gy/s per monitor-unit.

        intensity: same as relative_dose() (None/scalar/array).
        shutter_mask:
          - None: assume beam on for all frames
          - bool array length Nframes: False means no dose that frame

        Returns dict with:
          total_Gy
          Gy_cumulative
          time_axis_s
        """
        if dose_rate_Gy_per_s is None:
            raise ValueError("To compute absolute dose you must provide dose_rate_Gy_per_s (calibration).")

        rel = self.relative_dose(intensity=intensity)
        d_rel = np.diff(np.concatenate(([0.0], rel["rel_cumulative"])))  # per-frame "I*dt" units
        n = rel["n_frames"]

        if shutter_mask is None:
            mask = np.ones(n, dtype=bool)
        else:
            mask = np.asarray(shutter_mask, dtype=bool)
            if mask.size != n:
                raise ValueError(f"shutter_mask must be length {n}, got {mask.size}")

        # Convert relative units to Gy:
        # If intensity is "unitless", this is Gy per (unit*second) effectively.
        # You decide the meaning by how you feed intensity + calibration.
        k = float(dose_rate_Gy_per_s)
        d_Gy = np.where(mask, k * d_rel, 0.0)

        cum_Gy = np.cumsum(d_Gy)
        total_Gy = float(cum_Gy[-1])
        return {
            "total_Gy": total_Gy,
            "Gy_cumulative": cum_Gy,
            "time_axis_s": rel["time_axis_s"],
            "frame_time_s": rel["frame_time_s"],
            "n_frames": n,
        }


def main():
    p = argparse.ArgumentParser(description="Relative/absolute dose calculator for InterlacedScan (TomoScan fly scan).")
    p.add_argument("--K", type=int, default=4, help="InterlacedNumberOfRotation")
    p.add_argument("--N", type=int, default=32, help="InterlacedNumAnglesPerRotation")
    p.add_argument("--exposure", type=float, default=0.01, help="ExposureTime (s)")
    p.add_argument("--readout", type=float, default=0.01, help="Readout time (s)")
    p.add_argument("--readout-margin", type=float, default=1.0, help="Multiplier on readout contributing to dose")
    p.add_argument("--frame-time-mode", choices=["exposure", "exposure+readout"], default="exposure",
                   help="What time counts for dose per frame")
    p.add_argument("--accel", type=float, default=0.15, help="RotationAccelTime (s)")
    p.add_argument("--speed", type=float, default=60.0, help="SpeedDegPerSec")

    p.add_argument("--intensity", type=float, default=None,
                   help="Constant intensity (monitor units). If omitted -> assume constant 1.")
    p.add_argument("--dose-rate", type=float, default=None,
                   help="Calibration for absolute dose: Gy/s (or Gy/s per intensity unit if intensity provided).")
    p.add_argument("--shutter-mode", choices=["always_open", "open_after_accel"], default="always_open",
                   help="Simple shutter model for absolute dose")

    p.add_argument("--early-T", type=float, default=None, help="Compute dose within first T seconds")
    p.add_argument("--early-m", type=int, default=None, help="Compute dose within first m frames")

    args = p.parse_args()

    d = Dose(
        InterlacedNumberOfRotation=args.K,
        InterlacedNumAnglesPerRotation=args.N,
        ExposureTime=args.exposure,
        readout=args.readout,
        readout_margin=args.readout_margin,
        RotationAccelTime=args.accel,
        SpeedDegPerSec=args.speed,
        frame_time_mode=args.frame_time_mode,
    )

    # Relative dose
    rel = d.relative_dose(intensity=args.intensity)
    print("\n=== RELATIVE DOSE ===")
    print(f"Frames: {rel['n_frames']}")
    print(f"Frame time used for dose (s): {rel['frame_time_s']}")
    print(f"Total relative dose (arb): {rel['total_rel']:.6g}")

    if args.early_T is not None:
        val = d.relative_dose_early(T=args.early_T, intensity=args.intensity)
        print(f"Relative dose within first {args.early_T} s: {val:.6g}")

    if args.early_m is not None:
        val = d.relative_dose_early(m=args.early_m, intensity=args.intensity)
        print(f"Relative dose within first {args.early_m} frames: {val:.6g}")

    # Absolute dose (optional)
    if args.dose_rate is not None:
        shutter = d.simulate_shutter(args.shutter_mode)
        absd = d.absolute_dose(dose_rate_Gy_per_s=args.dose_rate, intensity=args.intensity, shutter_mask=shutter)
        print("\n=== ABSOLUTE DOSE (Gy) ===")
        print(f"Shutter mode: {args.shutter_mode}")
        print(f"Total dose (Gy): {absd['total_Gy']:.6g}")

    # Tiny sanity peek
    print("\n=== THETA PREVIEW (first 10) ===")
    print(np.array2string(d.theta[:10], precision=6, separator=", "))


if __name__ == "__main__":
    main()
