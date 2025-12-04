import numpy as np
import matplotlib.pyplot as plt

def round_robin_interlaced(N_theta=32, K=4, r_outer=1.0, r_step=0.15):
    angles = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
    loops = np.arange(N_theta) % K
    radii = r_outer - loops * r_step

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, polar=True)
    ax.set_title(f"Round-Robin Interlacing (N={N_theta}, K={K})")

    # Traccia i punti interlacciati
    ax.plot(angles, radii, '-o', lw=1.2)

    # Etichette dei cicli
    for i in range(N_theta):
        ax.text(angles[i], radii[i] + 0.03, str(loops[i]+1), ha='center')

    # Cerchio di riferimento al raggio massimo
    circle_angles = np.linspace(0, 2*np.pi, 500)
    circle_radii = np.full_like(circle_angles, r_outer)
    ax.plot(circle_angles, circle_radii, '--', color='gray', lw=1, label='r_outer')

    ax.set_rticks([])
    ax.legend()
    plt.show()
