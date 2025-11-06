import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Detector / geometry
# ------------------------
detector_x_size = 2048  # horizontal detector size in pixels
r = detector_x_size / 2  # radius in pixels

# ------------------------
# Exposure times to evaluate (s)
# ------------------------
exposure_times = np.arange(0.0001, 0.5, 0.01)

# ------------------------
# Motor speeds (deg/s)
# ------------------------
motor_speed_original = 0.2   # original speed
motor_speeds = np.array([motor_speed_original, 4*motor_speed_original])  # original vs 4x speed

labels = ['Original speed', '4x speed']

# ------------------------
# Compute effective blur
# ------------------------
effective_blur_px = []
for speed in motor_speeds:
    effective_blur_rad = np.radians(speed * exposure_times)        # angular displacement during exposure
    blur_px = 2 * r * np.sin(effective_blur_rad / 2)              # projected blur in pixels
    effective_blur_px.append(blur_px)

# ------------------------
# Nyquist limit (1 pixel)
# ------------------------
nyquist_limit = 1.0  # pixels

# ------------------------
# Plot
# ------------------------
plt.figure(figsize=(7,5))
for blur, label in zip(effective_blur_px, labels):
    plt.plot(exposure_times, blur, '-o', label=label)
plt.axhline(nyquist_limit, color='red', linestyle='--', label='Nyquist limit (1 px)')
plt.xlabel('Exposure time [s]')
plt.ylabel('Motion blur [pixels]')
plt.title('Motion blur vs exposure time for fly-scan tomography')
plt.grid(True)
plt.legend()
plt.show()
