'''
genera angoli interlacciati a path golden angle
stamap tab angoli semplici e cumulativi per loop
coverte angoli in pulses
grafici su base Timbir

fibonacci_offset= [( k loop / (numero totale di angoli da generare in ogni ciclo +1 ) ] * 360 * 0.618

l'offset basato sulla sequenza di Fibonacci viene aggiunto agli angoli del loop precedente
creando una nuova serie di angoli distribuiti uniformemente
'''

import numpy as np
import matplotlib.pyplot as plt

PSOCountsPerRotation = 200  # numero di impulsi per rotaz completa
# -----------------------------------
# angoli interlacciati
# -----------------------------------
def generate_interlaced_angles(rotation_start=0.0,
                               rotation_stop=360.0,
                               num_angles=32,
                               K_interlace=4):
    '''golden angle + fibo shift'''
    golden_angle = 360 * (3 - np.sqrt(5)) / 2  # Golden Angle ≈ 111.246°

    angles_all = []

    # 1 loop: golden angle
    angles = np.zeros(num_angles)
    for i in range(num_angles):
        angles[i] = (rotation_start + i * golden_angle) % 360
    angles_all.append(np.sort(angles))

    # K-loop : shift basato su fibo
    for k in range(1, K_interlace):
        fib_offset = (np.round((k / (num_angles + 1)) * 360 * (np.sqrt(5) - 1) / 2, 5)) % 360
        new_angles = (angles_all[0] + fib_offset) % 360   # **
        angles_all.append(np.sort(new_angles))

    return angles_all

# -----------------------------------
# tabella che mostra angoli per ogni loop
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
# angoli cumulativi per loop
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
# Plot tipo Timbir
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
    ax.set_rticks([])
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title("Interlaced Angles - Timbir Style")
    ax.legend(loc='upper right')
    plt.show()

# -----------------------------------
# conversione angoli in pulsazioni
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
# Plot angoli vs pulsazioni
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
# Plot combinato cumulativo angoli vs pulsazioni
# -----------------------------------
def plot_combined_cumulative(angles_cumulative, pulses_cumulative):
    plt.figure(figsize=(12, 6))
    for k, (angles, pulses) in enumerate(zip(angles_cumulative, pulses_cumulative)):
        plt.plot(angles, pulses, 'o-', label=f'Loop {k + 1}')
    # Linea cumulativa totale
    all_angles = np.concatenate(angles_cumulative)
    all_pulses = np.concatenate(pulses_cumulative)
    sort_idx = np.argsort(all_angles)
    plt.plot(all_angles[sort_idx], all_pulses[sort_idx], 'k-', alpha=0.5, label='Cumulative total')
    plt.xlabel("Cumulative angle [deg]")
    plt.ylabel("Cumulative pulse count")
    plt.title("Combined Cumulative Plot - All Loops")
    plt.grid(True)
    plt.legend()
    plt.show()

# -----------------------------------
# ESEMPIO DI UTILIZZO ORDINATO
# -----------------------------------
angles_list = generate_interlaced_angles(num_angles=32, K_interlace=4)

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
plot_combined_cumulative(cumulative_loops, pulses_cumulative)


