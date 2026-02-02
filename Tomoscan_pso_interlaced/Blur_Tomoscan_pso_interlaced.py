import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    from Tomoscan_pso_interlaced import InterlacedScan
except Exception:
    InterlacedScan = None

# ----------------------------
# Motion blur tools
# ----------------------------
def motion_blur_px(r_px, speed_deg_s, exposure_s):

    eff_rad = np.deg2rad(speed_deg_s * exposure_s)    # spostamento angolare durante l’esposizione -> angolo in radianti
    return 2.0 * r_px * np.sin(eff_rad / 2.0)


def t_max_for_blur(b_px, r_px, speed_deg_s):

    speed = np.asarray(speed_deg_s, dtype=float)
    omega = np.deg2rad(speed)  # rad/s

    with np.errstate(divide="ignore", invalid="ignore"):
        tmax = b_px / (r_px * omega)

    tmax = np.where(omega > 0, tmax, np.inf)

    # return scalar if scalar input
    return float(tmax) if np.ndim(speed_deg_s) == 0 else tmax


def run_blur_plots(
    detector_x_size,
    motor_speeds,
    exposure_min,
    exposure_max,
    exposure_step,
    speed_min,
    speed_max,
    speed_n,
    blur_limit_px,
):
    r = detector_x_size / 2.0

    exposure_times = np.arange(exposure_min, exposure_max + 1e-12, exposure_step)
    speeds_grid = np.linspace(speed_min, speed_max, speed_n)

    # ---- plot blur vs exposure for selected speeds
    plt.figure(figsize=(7, 5))
    for s in motor_speeds:
        blur = motion_blur_px(r, s, exposure_times)
        plt.plot(exposure_times, blur, "-o", label=f"{s:.6g} deg/s")
    plt.axhline(blur_limit_px, linestyle="--", label=f"Blur limit ({blur_limit_px} px)")
    plt.xlabel("Exposure time [s]")
    plt.ylabel("Motion blur [pixels]")
    plt.title("Motion blur vs exposure time (fly-scan)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # print t_max for chosen speeds
    for s in motor_speeds:
        print(f"{s:.6g} deg/s -> t_max({blur_limit_px}px) = {t_max_for_blur(blur_limit_px, r, s):.6g} s")

    # tmax vs speed (linear)
    tmax = t_max_for_blur(blur_limit_px, r, speeds_grid)

    plt.figure(figsize=(7, 5))
    plt.plot(speeds_grid, tmax, label=f"t_max for blur={blur_limit_px}px")
    for s in motor_speeds:
        plt.axvline(s, linestyle="--")
        plt.scatter([s], [t_max_for_blur(blur_limit_px, r, s)], zorder=3)
        plt.text(s, t_max_for_blur(blur_limit_px, r, s), f"  {s:.6g} deg/s", va="bottom")
    plt.xlabel("Motor speed [deg/s]")
    plt.ylabel("Max exposure time [s]")
    plt.title("Max exposure time vs motor speed (linear)")
    plt.grid(True)
    plt.legend()
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.show()

    # tmax vs speed (log y)
    plt.figure(figsize=(7, 5))
    plt.plot(speeds_grid, tmax, label=f"t_max for blur={blur_limit_px}px")
    for s in motor_speeds:
        plt.axvline(s, linestyle="--")
        plt.scatter([s], [t_max_for_blur(blur_limit_px, r, s)], zorder=3)
        plt.text(s, t_max_for_blur(blur_limit_px, r, s), f"  {s:.6g} deg/s", va="bottom")
    plt.xlabel("Motor speed [deg/s]")
    plt.ylabel("Max exposure time [s]")
    plt.title("Max exposure time vs motor speed (log y)")
    plt.grid(True)
    plt.legend()
    plt.yscale("log")
    plt.ylim(bottom=1e-6)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Tomoscan methods (optional)
# ----------------------------
def run_one_method(scan, mode: str):
    if mode == "timbir":
        scan.generate_interlaced_timbir()
        scan.plotdelta("timbir")

    elif mode == "multitimbir":
        scan.generate_interlaced_multitimbir()
        scan.plotdelta("multitimbir")

    elif mode == "golden":
        angles_all = scan.generate_interlaced_goldenangle()
        scan.print_angles_table(angles_all)
        scan.print_cumulative_angles_table(angles_all)
        scan.plot_interlaced_circles(angles_all)
        scan.plotdelta("golden")

    elif mode == "kturns":
        angles_all = scan.generate_interlaced_kturns()
        scan.plot_equally_loops_polar_kturns()
        scan.print_cumulative_angles_table_kturns(angles_all)
        scan.print_angles_table_kturns(angles_all)
        scan.plotdelta("kturns")

    elif mode == "multiturns":
        angles_all = scan.generate_interlaced_multiturns()
        scan.plot_equally_loops_polar_multiturns()
        scan.print_cumulative_angles_table_multiturns(angles_all)
        scan.print_angles_table_multiturns(angles_all)
        scan.plotdelta("multiturns")

    elif mode == "corput":
        angles_all = scan.generate_interlaced_corput()
        scan.plot_equally_loops_polar_corput()
        scan.print_cumulative_angles_table_corput(angles_all)
        scan.print_angles_table_corput(angles_all)
        scan.plot_live_corput()
        scan.plotdelta("corput")

    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    p = argparse.ArgumentParser(description="Run interlaced scan + optional motion blur analysis.")

    # tomoscan params
    p.add_argument("--mode", choices=["timbir", "multitimbir", "golden", "kturns", "multiturns", "corput"],
                   default="timbir")
    p.add_argument("--num_angles", type=int, default=32)
    p.add_argument("--K_interlace", type=int, default=4)
    p.add_argument("--PSOCountsPerRotation", type=int, default=20000)

    # camera timing
    p.add_argument("--exposure", type=float, default=0.01)
    p.add_argument("--readout", type=float, default=0.01)

    # blur flags
    p.add_argument("--blur_only", action="store_true",
                   help="Run ONLY blur analysis, without importing/running tomoscan methods.")
    p.add_argument("--run_blur", action="store_true",
                   help="When running tomoscan, also run blur analysis at the end.")

    # blur params (single speed)
    p.add_argument("--motor_speed", type=float, default=0.2,
                   help="Motor speed in deg/s (single value).")
    p.add_argument("--compare_factor", action="store_true",
                   help="If set, also plot a second curve at speed_factor * motor_speed.")
    p.add_argument("--speed_factor", type=float, default=4.0,
                   help="Factor for second speed when --compare_factor is set (default 4).")

    # blur plot ranges
    p.add_argument("--detector_x_size", type=int, default=2048)
    p.add_argument("--blur_limit_px", type=float, default=1.0)
    p.add_argument("--exposure_min", type=float, default=0.0001)
    p.add_argument("--exposure_max", type=float, default=0.5)
    p.add_argument("--exposure_step", type=float, default=0.01)
    p.add_argument("--speed_min", type=float, default=0.05)
    p.add_argument("--speed_max", type=float, default=2.0)
    p.add_argument("--speed_n", type=int, default=200)

    args = p.parse_args()

    # build speeds list (single speed, optionally also factor*speed)
    ms = [float(args.motor_speed)]
    if args.compare_factor:
        ms.append(float(args.speed_factor) * float(args.motor_speed))

    # ---- BLUR ONLY
    if args.blur_only:
        run_blur_plots(
            args.detector_x_size, ms,
            args.exposure_min, args.exposure_max, args.exposure_step,
            args.speed_min, args.speed_max, args.speed_n,
            args.blur_limit_px
        )
        return

    # ---- TOMOSCAN + optional blur
    if InterlacedScan is None:
        raise ImportError(
            "Non riesco a importare InterlacedScan da Tomoscan_pso_interlaced.py.\n"
            "Soluzioni:\n"
            "  1) Metti questo script nella stessa cartella di Tomoscan_pso_interlaced.py\n"
            "  2) Oppure lancia con --blur_only se vuoi solo il blur."
        )

    scan = InterlacedScan(
        num_angles=args.num_angles,
        K_interlace=args.K_interlace,
        PSOCountsPerRotation=args.PSOCountsPerRotation,
        exposure=args.exposure,
        readout=args.readout,
    )

    run_one_method(scan, args.mode)

    # pipeline counts
    scan.compute_positions_PSO()
    scan.simulate_taxi_motion()
    scan.compute_real_motion()
    scan.convert_angles_to_counts()
    scan.plot_all_comparisons()
    scan.plot()

    # optional blur at end
    if args.run_blur:
        print(f"\n[Blur] Using motor speed(s) (deg/s): {ms}")
        run_blur_plots(
            args.detector_x_size, ms,
            args.exposure_min, args.exposure_max, args.exposure_step,
            args.speed_min, args.speed_max, args.speed_n,
            args.blur_limit_px
        )


if __name__ == "__main__":
    main()



# python Blur_Tomoscan_pso_interlaced.py --blur_only --motor_speed 0.2 --blur_limit_px 1.0

#  python Blur_Tomoscan_pso_interlaced.py --mode timbir --num_angles 32 --K_interlace 4 --PSOCountsPerRotation 20000 --run_blur --motor_speed 0.2 --blur_limit_px 1.0
