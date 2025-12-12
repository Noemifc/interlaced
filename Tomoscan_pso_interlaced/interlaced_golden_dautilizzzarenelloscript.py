import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parametri di acquisizione
PSOCountsPerRotation = 200  # numero di impulsi per rotazione completa

# -----------------------------------
# Funzione per generare gli angoli interlacciati
# -----------------------------------
def generate_interlaced_angles(rotation_start=0.0,
                               rotation_stop=360.0,
                               num_angles=32,
                               K_interlace=4):
    '''golden angle + fibo shift'''
    golden_angle = 360 * (3 - np.sqrt(5)) / 2  # Golden Angle ≈ 111.246°

    angles_all = []

    # Primo loop: Golden angle
    angles = np.zeros(num_angles)
    for i in range(num_angles):
        angles[i] = (rotation_start + i * golden_angle) % 360
    angles_all.append(np.sort(angles))

    # K-loop: shift basato su Fibonacci
    for k in range(1, K_interlace):
        fib_offset = (np.round((k / (num_angles + 1)) * 360 * (np.sqrt(5) - 1) / 2, 5)) % 360
        new_angles = (angles_all[0] + fib_offset) % 360
        angles_all.append(np.sort(new_angles))

    return angles_all

# -----------------------------------
# Funzione per visualizzare la tabella degli angoli
# -----------------------------------
def print_angles_table(angles_all):
    print(f"{'Index':>5}", end='')
    for k in range(len(angles_all)):
        print(f"{f'Loop {k + 1}':>12}", end='')
    print()

    num_angles = len(angles_all[0])
    for i in range(num_angles):
        print(f"{i:5}", end='')
        for k in range(len(angles_all)):
            print(f"{angles_all[k][i]:12.3f}", end='')
        print()

# -----------------------------------
# Funzione per visualizzare gli angoli cumulativi
# -----------------------------------
def print_cumulative_angles_table(angles_all):
    print(f"{'Index':>5}", end='')
    for k in range(len(angles_all)):
        print(f"{f'Loop {k + 1}':>15}", end='')
    print()

    num_angles = len(angles_all[0])
    cumulative_loops = [angles_all[0].copy()]

    for k in range(1, len(angles_all)):
        prev_max = cumulative_loops[k - 1].max()
        cumulative_angles = angles_all[k] + np.ceil(prev_max / 360) * 360
        cumulative_loops.append(cumulative_angles)

    for i in range(num_angles):
        print(f"{i:5}", end='')
        for k in range(len(cumulative_loops)):
            print(f"{cumulative_loops[k][i]:15.3f}", end='')
        print()

    return cumulative_loops

# -----------------------------------
# Funzione per visualizzare il grafico interlacciato in stile Timbir
# -----------------------------------
def plot_interlaced_circles(angles_all):
    K_interlace = len(angles_all)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for k, angles in enumerate(angles_all):
        radius = 1 + (K_interlace - 1 - k) * 0.3
        theta = np.deg2rad(angles)
        r = np.full_like(theta, radius)
        ax.scatter(theta, r, label=f'Loop {k + 1}', s=10)
    ax.set_rticks([])  # Non mostrare i ticks radialmente
    ax.set_theta_zero_location('N')  # Inizia la griglia a nord
    ax.set_theta_direction(-1)  # Direzione oraria
    ax.set_title("Interlaced Angles - Timbir Style")
    ax.legend(loc='upper right')
    plt.show()

# -----------------------------------
# Funzione per convertire gli angoli in pulsazioni
# -----------------------------------
def convert_angles_to_pulses(angles_all, description=""):
    pulses_per_degree = PSOCountsPerRotation / 360.0
    pulses_loops = []
    print(f"\n--- Conversione in pulsazioni: {description} ---")
    for loop_idx, angles in enumerate(angles_all):
        pulse_counts = np.round(angles * pulses_per_degree).astype(int)
        pulses_loops.append(pulse_counts)
        actual_angles = pulse_counts / pulses_per_degree
        angular_error = actual_angles - angles

        print(f"\nLoop {loop_idx + 1}:")
        print(f"{'Target [deg]':>12} | {'Pulse':>5} | {'Actual [deg]':>12} | {'Error [deg]':>12}")
        print("-" * 50)
        for a, p, act, err in zip(angles, pulse_counts, actual_angles, angular_error):
            print(f"{a:12.3f} | {p:5d} | {act:12.6f} | {err:12.6f}")
    return pulses_loops

# -----------------------------------
# Funzione per visualizzare angoli vs pulsazioni
# -----------------------------------
def plot_angles_vs_pulses(angles_all, pulses_all, title="Angles vs Pulses"):
    plt.figure(figsize=(10, 6))
    for k, (angles, pulses) in enumerate(zip(angles_all, pulses_all)):
        plt.plot(angles, pulses, 'o-', label=f'Loop {k + 1}')
    plt.xlabel("Angle [deg]")
    plt.ylabel("Pulse count")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

# -----------------------------------
# Funzione principale
# -----------------------------------
def main():
    global PSOCountsPerRotation

    parser = argparse.ArgumentParser(description="Generate and analyze interlaced golden-angle angles.")
    parser.add_argument(
        "--rotation_start",
        type=float,
        default=0.0,
        help="Start rotation angle in degrees (default: 0.0)",
    )
    parser.add_argument(
        "--rotation_stop",
        type=float,
        default=360.0,
        help="Stop rotation angle in degrees (default: 360.0)",
    )
    parser.add_argument(
        "--num_angles",
        type=int,
        default=32,
        help="Number of angles per loop (default: 32)",
    )
    parser.add_argument(
        "--K_interlace",
        type=int,
        default=4,
        help="Number of interlaced loops K (default: 4)",
    )
    parser.add_argument(
        "--PSOCountsPerRotation",
        type=int,
        default=200,
        help="PSO counts per full rotation (default: 200)",
    )

    args = parser.parse_args()

    # Override globale PSOCountsPerRotation con il valore della CLI
    PSOCountsPerRotation = args.PSOCountsPerRotation

    # Generazione angoli interlacciati
    angles_list = generate_interlaced_angles(
        rotation_start=args.rotation_start,
        rotation_stop=args.rotation_stop,
        num_angles=args.num_angles,
        K_interlace=args.K_interlace,
    )

    print("\n--- Tabella angoli originali ---")
    print_angles_table(angles_list)

    print("\n--- Tabella cumulativa ---")
    cumulative_loops = print_cumulative_angles_table(angles_list)

    # Conversione in pulsazioni
    pulses_list = convert_angles_to_pulses(angles_list, description="Original angles")
    pulses_cumulative = convert_angles_to_pulses(cumulative_loops, description="Cumulative angles")

    # Plot
    plot_interlaced_circles(angles_list)
    plot_angles_vs_pulses(angles_list, pulses_list, title="Original Angles vs Pulses")
    plot_angles_vs_pulses(cumulative_loops, pulses_cumulative, title="Cumulative Angles vs Pulses")

if __name__ == "__main__":
    main()
