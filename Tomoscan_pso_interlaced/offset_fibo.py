import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parametri
# ----------------------------------------------------------------------
num_angles = 32       # numero di angoli per ciclo
K_interlace = 4       # numero di loop
golden_ratio = (np.sqrt(5) - 1) / 2  # rapporto aureo = 0.618
rotation_start = 0    # angolo iniziale

# ----------------------------------------------------------------------
# Funzione per generare angoli interlacciati con offset Fibonacci
# ----------------------------------------------------------------------
def generate_interlaced_angles(rotation_start, num_angles, K_interlace):
    golden_angle = 360 * golden_ratio  # Golden Angle= 222.492° per  360*(√5-1)/2
    # golden_a = (3 - np.sqrt(5)) / 2 # ≈ 111.246° oppure questo?
    
    
    angles_all = []

    # ------------------------------------------------------------------
    # 1. Primo loop: angoli base senza offset (Golden Angle)
    # metodo: theta_i = theta0 + i * incremento angolare, modulo 360
    # ------------------------------------------------------------------
    angles = np.zeros(num_angles)    
    for i in range(num_angles):
        angles[i] = (rotation_start + i * golden_angle) % 360     
    angles_all.append(np.sort(angles))

    # ------------------------------------------------------------------
    # 2. Loop successivi: aggiungo offset Fibonacci
    # ------------------------------------------------------------------
    for k in range(1, K_interlace):
        # calcolo offset Fibonacci proporzionale al loop
        fib_offset = (np.round((k / (num_angles + 1)) * 360 * golden_ratio, 5)) % 360
        # aggiungo offset agli angoli del primo loop
        new_angles = (angles_all[0] + fib_offset) % 360  
        angles_all.append(np.sort(new_angles))
        
    return angles_all

# ----------------------------------------------------------------------
# Generazione angoli interlacciati
# ----------------------------------------------------------------------
angles_all = generate_interlaced_angles(rotation_start, num_angles, K_interlace)

# ----------------------------------------------------------------------
# Stampa tabella angoli per loop
# ----------------------------------------------------------------------
print("\n--- Tabella Angoli per Loop ---")
for i in range(num_angles):
    # ogni k nella stessa riga
    print(f"{i + 1:3} ", end="")  # numero dell'angolo
    for k in range(K_interlace):
        print(f"{angles_all[k][i]:12.3f}", end="  ")
    print()  # a capo per il prossimo angolo

# ----------------------------------------------------------------------
# Plot grafico angoli interlacciati
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10(np.linspace(0, 1, K_interlace))

for k, angles in enumerate(angles_all):
    plt.scatter(angles, np.ones_like(angles) * k, label=f'Loop {k + 1}', color=colors[k], s=50)

plt.title("Interlaced Angles with Fibonacci Offset")
plt.xlabel("Angle [deg]")
plt.ylabel("Loop Number")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

