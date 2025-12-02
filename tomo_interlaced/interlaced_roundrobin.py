"""
Ogni loop prende un punto, poi passa al successivo, e così via
"""

import numpy as np
import matplotlib.pyplot as plt

def round_robin_interlaced(N_theta=32, K=4, r_outer=1.0, r_step=0.15):
    angles = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
    loops = np.arange(N_theta) % K
    radii = r_outer - loops * r_step

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, polar=True)
    ax.set_title(f"Round-Robin Interlacing (N={N_theta}, K={K})")

    ax.plot(angles, radii, '-o', lw=1.2)

    for i in range(N_theta):
        ax.text(angles[i], radii[i] + 0.03, str(loops[i]+1), ha='center')

    ax.set_rticks([])
    plt.show()
