import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import rotate

# ============================================
# Funzioni di base
# ============================================
def generate_object(size=200):
    """Genera una semplice croce luminosa al centro"""
    img = np.zeros((size, size))
    c = size // 2
    img[c-2:c+2, :] = 1
    img[:, c-2:c+2] = 1
    return img

def simulate_blur(img, exposure, angular_velocity, n_samples=50):
    """Simula il blur durante la rotazione"""
    total_angle = angular_velocity * exposure
    angles = np.linspace(-total_angle/2, total_angle/2, n_samples)
    blurred = np.zeros_like(img)
    for a in angles:
        blurred += rotate(img, a, reshape=False, order=1)
    blurred /= n_samples
    return blurred

def radon_projection(img, angles):
    """Calcola proiezioni semplici (sinogramma)"""
    sino = []
    for a in angles:
        rotated = rotate(img, a, reshape=False, order=1)
        projection = np.sum(rotated, axis=0)
        sino.append(projection)
    return np.array(sino)

def simple_backprojection(sino, angles):
    """Retroproiezione grezza per visualizzazione"""
    size = sino.shape[1]
    recon = np.zeros((size, size))
    center = size // 2
    x = np.arange(size) - center
    X, Y = np.meshgrid(x, x)
    for i, angle in enumerate(angles):
        t = X * np.cos(np.deg2rad(angle)) + Y * np.sin(np.deg2rad(angle))
        t_index = (t + center).astype(int)
        mask = (t_index >= 0) & (t_index < size)
        recon[mask] += sino[i, t_index[mask]]
    recon /= len(angles)
    return recon

def interlaced_angles(angles):
    """Genera un ordine interlacciato delle proiezioni usando bit-reversal"""
    n = len(angles)
    bits = int(np.ceil(np.log2(n)))
    order = []
    for i in range(n):
        rev = 0
        for b in range(bits):
            rev = (rev << 1) | ((i >> b) & 1)
        if rev < n:
            order.append(rev)
    return angles[order]

# ============================================
# Setup iniziale
# ============================================
obj = generate_object()
exposure_init = 0.05
velocity_init = 90
step_init = 2.0  # passo angolare in gradi
use_interlaced = False  # True per attivare proiezioni interlacciate

angles_raw = np.arange(0, 180, step_init)
angles = interlaced_angles(angles_raw) if use_interlaced else angles_raw

blurred = simulate_blur(obj, exposure_init, velocity_init)
sino = radon_projection(blurred, angles)
recon = simple_backprojection(sino, angles)

# ============================================
# Visualizzazione
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(bottom=0.3)

im1 = axes[0].imshow(blurred, cmap='gray')
axes[0].set_title('Proiezione sfocata')

im2 = axes[1].imshow(sino, cmap='gray', aspect='auto')
axes[1].set_title('Sinogramma')

im3 = axes[2].imshow(recon, cmap='gray')
axes[2].set_title('Ricostruzione')

# Slider
ax_expo = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_speed = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_step = plt.axes([0.25, 0.1, 0.65, 0.03])

slider_expo = Slider(ax_expo, 'Exposure Time [s]', 0.01, 0.2, valinit=exposure_init)
slider_speed = Slider(ax_speed, 'Angular Velocity [°/s]', 10, 360, valinit=velocity_init)
slider_step = Slider(ax_step, 'Angular Step [°]', 0.5, 10.0, valinit=step_init)

# Aggiornamento interattivo
def update(val):
    exp = slider_expo.val
    vel = slider_speed.val
    step = slider_step.val

    angles_raw = np.arange(0, 180, step)
    angles = interlaced_angles(angles_raw) if use_interlaced else angles_raw

    blurred_new = simulate_blur(obj, exp, vel)
    sino_new = radon_projection(blurred_new, angles)
    recon_new = simple_backprojection(sino_new, angles)

    im1.set_data(blurred_new)
    im2.set_data(sino_new)
    im3.set_data(recon_new)
    fig.canvas.draw_idle()

slider_expo.on_changed(update)
slider_speed.on_changed(update)
slider_step.on_changed(update)

plt.show()

