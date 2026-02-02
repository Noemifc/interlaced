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

#Funzioni aggiuntive 

#dato il limite di blur -> dammi exopur massimo entro i lim

def t_max_for_blur(b_px, r_px, speed_deg_s):
    omega = np.deg2rad(speed_deg_s)  # rad/s
    return b_px / (r_px * omega)

for speed in [0.2, 0.8]:                                                         #tes con due velocita'
    print(speed, "deg/s -> t_max(1px) =", t_max_for_blur(1.0, r, speed), "s")
    print(speed, "deg/s -> t_max(0.5px) =", t_max_for_blur(0.5, r, speed), "s")
    
# plot tmax di esposizione vs velocità del motore per un dato limite di blur
speeds = np.linspace(0.05, 2.0, 200)
#soglie di blur
tmax_1px  = t_max_for_blur(1.0, r, speeds)
tmax_05px = t_max_for_blur(0.5, r, speeds)

plt.figure(figsize=(7,5))
plt.plot(speeds, tmax_1px,  label="t_max per blur = 1 px")
plt.plot(speeds, tmax_05px, label="t_max per blur = 0.5 px")

for s in [0.2, 0.8]:
    plt.axvline(s, linestyle="--")
    plt.scatter([s], [t_max_for_blur(1.0, r, s)], zorder=3)
    plt.text(s, t_max_for_blur(1.0, r, s), f"  {s} deg/s", va="bottom")

plt.xlabel("Motor speed [deg/s]")
plt.ylabel("Max exposure time [s]")
plt.title("Max exposure time vs motor speed (dato un limite di blur)")
plt.grid(True)
plt.legend()
plt.ylim(bottom=1e-4)    
plt.yscale("log")         # scala log 
plt.show()



































