import time
import matplotlib.pyplot as plt

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
num_points = 50      # quanti punti simulare
base = 2             # base della sequenza (2 o 10)
delay = 0.2          # intervallo tra i dati in secondi

# Preparazione grafico
plt.ion()  # modalità interattiva
fig, ax = plt.subplots()
ax.set_xlim(0, num_points)
ax.set_ylim(0, 1)
ax.set_xlabel("Acquisizione")
ax.set_ylabel("Valore Van der Corput")
line, = ax.plot([], [], 'bo')

x_data = []
y_data = []

print("Simulazione acquisizione dati in tempo reale:")
for i in range(1, num_points+1):
    value = van_der_corput(i, base)
    x_data.append(i)
    y_data.append(value)
    line.set_data(x_data, y_data)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)
    print(f"Dato {i}: {value:.4f}")
    time.sleep(delay)

plt.ioff()
plt.show()
