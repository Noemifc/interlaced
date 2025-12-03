# prendi un angolo e poi salta di J passi

import numpy as np
import matplotlib.pyplot as plt

def jump_interlaced(N_theta=32, J=5, K=4, r_outer=1.0, r_step=0.15):
    visited = []
    idx = 0
    for _ in range(N_theta):
        visited.append(idx)
        idx = (idx + J) % N_theta

    visited = np.array(visited)
    angles = visited * 2*np.pi / N_theta
    loops = visited % K
    radii = r_outer - loops * r_step

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, polar=True)
    ax.set_title(f"Jump Interlacing J={J} (N={N_theta}, K={K})")

    ax.plot(angles, radii, '-o', lw=1.2)

    for i in range(N_theta):
        ax.text(angles[i], radii[i] + 0.03, str(loops[i]+1), ha='center')

    ax.set_rticks([])
    plt.show()
