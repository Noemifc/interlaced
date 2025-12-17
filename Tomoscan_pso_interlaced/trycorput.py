import time
import matplotlib.pyplot as plt
import numpy as np

def van_der_corput(n, base=2):
    """Calcola il n-esimo numero della sequenza di Van der Corput in una data base."""
    vdc = 0
    denom = 1
    while n > 0:
        n, remainder = divmod(n, base)
        denom *= base
        vdc += remainder / denom
    return vdc

# Parametri
num_points = 50      # numero di dati simulati
base = 11             # base per Van der Corput
delay = 0.2          # intervallo tra i dati in secondi

# Preparazione grafico
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
fig.suptitle("Simulazione acquisizione dati con sequenza di Van der Corput")

# Grafico 1: valori Van der Corput 1D
ax1.set_xlim(0, num_points)
ax1.set_ylim(0, 1)
ax1.set_xlabel("Acquisizione")
ax1.set_ylabel("Valore VdC")
line1, = ax1.plot([], [], 'bo')

# Grafico 2: angoli di acquisizione lineari
ax2.set_xlim(0, num_points)
ax2.set_ylim(0, 360)
ax2.set_xlabel("Acquisizione")
ax2.set_ylabel("Angolo (gradi)")
line2, = ax2.plot([], [], 'ro')

# Grafico 3: plot circolare
ax3 = plt.subplot(313, polar=True)
scatter = ax3.scatter([], [], c='g', s=50)
ax3.set_title("Distribuzione circolare degli angoli")

# Dati
x_data = []
y_data = []
angle_data = []
theta_data = []

print("Simulazione acquisizione dati in tempo reale con plot circolare:")
for i in range(1, num_points + 1):
    # Valore Van der Corput
    value = van_der_corput(i, base)
    x_data.append(i)
    y_data.append(value)
    line1.set_data(x_data, y_data)
    ax1.relim()
    ax1.autoscale_view()

    # Angolo di acquisizione lineare
    angle = value * 360
    angle_data.append(angle)
    line2.set_data(x_data, angle_data)
    ax2.relim()
    ax2.autoscale_view()

    # Angolo in radianti per plot circolare
    theta = np.deg2rad(angle)
    theta_data.append(theta)
    ax3.cla()  # pulisce il plot circolare
    ax3.scatter(theta_data, [1]*len(theta_data), c='g', s=50)  # raggio=1
    ax3.set_ylim(0, 1.2)
    ax3.set_title("Distribuzione circolare degli angoli")

    # Aggiorna grafico e stampa
    plt.draw()
    plt.pause(0.01)
    print(f"Dato {i}: VdC={value:.4f}, Angolo={angle:.1f}°")
    time.sleep(delay)

plt.ioff()
plt.show()
