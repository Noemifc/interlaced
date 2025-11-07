import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from scipy.ndimage import rotate

# ============================================
# Funzioni di base
# ============================================
def generate_object(size=200, mode='cross'):
    """
    Genera un'immagine 2D per la simulazione CT.
    mode:
        'cross' -> croce
        'cube' -> cubo 3D proiettato in 2D
    """
    img = np.zeros((size, size))
    c = size // 2

    if mode == 'cross':
        img[c-2:c+2, :] = 1
        img[:, c-2:c+2] = 1
    elif mode == 'cube':
        # cubo 3D: proiezione ortogonale lungo Z
        cube_size = size // 4
        start = c - cube_size // 2
        end = c + cube_size // 2
        img[start:end, start:end] = 1
        # aggiungiamo "ombreggiatura" per simulare profondità
        img[start:end, start:end] += 0.5  # faccia superiore leggermente più chiara
        img = np.clip(img, 0, 1)

    return img

def simulate_blur(img, exposure, angular_velocity, n_samples=50):
    total_angle = angular_velocity * exposure
    angles = np.linspace(-total_angle/2, total_angle/2, n_samples)
    blurred = np.zeros_like(img)
    for a in angles:
        blurred += rotate(img, a, reshape=False, order=1)
    blurred /= n_samples
    return blurred

def radon_projection(img, angles):
    sino = []
    for a in angles:
        rotated = rotate(img, a, reshape=False, order=1)
        projection = np.sum(rotated, axis=0)
        sino.append(projection)
    return np.array(sino)

def simple_backprojection(sino, angles):
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

def golden_interlaced_angles(theta_start, num_proj, golden_a=180*(3 - np.sqrt(5))/2):
    golden_angles_tomo = np.mod(
        theta_start[:, None] + np.arange(num_proj) * golden_a,
        180
    ).flatten()
    return golden_angles_tomo

# ============================================
# Setup iniziale
# ============================================
# Se non hai immagine esterna, genera un cubo 3D
obj = generate_object(mode='cube')

exposure_init = 0.05
velocity_init = 90
num_proj_init = 36

theta_start = np.array([
    13.76941013, 30., 40.0310562, 56.26164608, 82.52329215,
    98.75388203, 108.78493823, 125.0155281, 151.27717418, 167.50776405
])
use_golden = True

def compute_angles(num_proj, use_golden):
    if use_golden:
        return golden_interlaced_angles(theta_start, num_proj)
    else:
        return np.linspace(0, 180, num_proj, endpoint=False)

angles = compute_angles(num_proj_init, use_golden)

blurred = simulate_blur(obj, exposure_init, velocity_init)
sino = radon_projection(blurred, angles)
recon = simple_backprojection(sino, angles)

# ============================================
# Visualizzazione
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(15,5))
plt.subplots_adjust(bottom=0.35)

im1 = axes[0].imshow(blurred, cmap='gray')
axes[0].set_title('Proiezione sfocata')

im2 = axes[1].imshow(sino, cmap='gray', aspect='auto')
axes[1].set_title('Sinogramma')

im3 = axes[2].imshow(recon, cmap='gray')
axes[2].set_title('Ricostruzione')

# Slider
ax_expo = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_speed = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_numproj = plt.axes([0.25, 0.15, 0.65, 0.03])

slider_expo = Slider(ax_expo, 'Exposure Time [s]', 0.01, 0.2, valinit=exposure_init)
slider_speed = Slider(ax_speed, 'Angular Velocity [°/s]', 10, 360, valinit=velocity_init)
slider_numproj = Slider(ax_numproj, 'Num Proj per Serie', 5, 100, valinit=num_proj_init, valstep=1)

# Check button
ax_check = plt.axes([0.025, 0.5, 0.15, 0.15])
check = CheckButtons(ax_check, ['Golden-angle interlacing'], [use_golden])

# Aggiornamento
def update(val):
    exp = slider_expo.val
    vel = slider_speed.val
    num_proj = int(slider_numproj.val)
    global use_golden

    angles = compute_angles(num_proj, use_golden)

    blurred_new = simulate_blur(obj, exp, vel)
    sino_new = radon_projection(blurred_new, angles)
    recon_new = simple_backprojection(sino_new, angles)

    im1.set_data(blurred_new)
    im2.set_data(sino_new)
    im3.set_data(recon_new)
    fig.canvas.draw_idle()

slider_expo.on_changed(update)
slider_speed.on_changed(update)
slider_numproj.on_changed(update)

# Aggiornamento check button
def toggle_golden(label):
    global use_golden
    use_golden = not use_golden
    update(None)

check.on_clicked(toggle_golden)

plt.show()
