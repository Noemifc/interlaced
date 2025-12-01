


# ----------------------------------------------------
# PROFILO MOTORE E CALCOLO TRIGGER
# ----------------------------------------------------
omega_target = 5.0         # velocità angolare target [deg/s]
T_acc = 0.5                # tempo accelerazione [s]
T_dec = 0.5                # tempo decelerazione [s]

# Calcolo tratto piatto (utile)
T_flat = 180.0 / omega_target - T_acc - T_dec

# Funzione inversa t_real(θ) per trigger reali
def t_real(theta_target, T_acc, T_flat, T_dec, omega_target):
    alpha = omega_target / T_acc
    t_out = np.zeros_like(theta_target, dtype=float)
    theta_acc = 0.5 * alpha * T_acc**2
    theta_flat = theta_acc + omega_target * T_flat
    theta_total = theta_flat + 0.5*alpha*T_dec**2

    for i, th in enumerate(theta_target):
        if th <= theta_acc:
            t_out[i] = np.sqrt(2*th/alpha)
        elif th <= theta_flat:
            t_out[i] = T_acc + (th - theta_acc)/omega_target
        elif th <= theta_total:
            a = 0.5*alpha
            b = -omega_target
            c = theta_flat - th
            dt = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
            t_out[i] = T_acc + T_flat + dt
        else:
            t_out[i] = T_acc + T_flat + T_dec
    return t_out

# Calcolo tempo trigger per angoli corretti
t_triggers = t_real(theta_corrected, T_acc, T_flat, T_dec, omega_target)

# Velocità istantanea
def omega_inst(t, T_acc, T_flat, T_dec, omega_target):
    alpha = omega_target / T_acc
    omega = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < T_acc:
            omega[i] = alpha * ti
        elif ti < T_acc + T_flat:
            omega[i] = omega_target
        elif ti <= T_acc + T_flat + T_dec:
            dt = ti - T_acc - T_flat
            omega[i] = omega_target - alpha*dt
        else:
            omega[i] = 0.0
    return omega

omega_values = omega_inst(t_triggers, T_acc, T_flat, T_dec, omega_target)

# Impulsi reali PSO
pulses_real = theta_corrected * (counts_per_rev / 360.0)

# ----------------------------------------------------
# STAMPA RISULTATI
# ----------------------------------------------------
for i in range(len(theta_corrected)):
    print(f"Angle {theta_corrected[i]:6.2f} deg -> Loop {loop_indices[i]} -> Pulse {pulses_real[i]:.0f} -> t_trigger {t_triggers[i]:.4f} s -> omega {omega_values[i]:.2f} deg/s")

print(f"Fine taxi: {theta_end_corrected:.3f} deg -> Pulse {pulses_end_corrected}")
