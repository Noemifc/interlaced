
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
