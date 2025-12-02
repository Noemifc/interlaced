import numpy as np

def compute_real_timeline(theta_corrected, omega_target, accel, counts_per_rev):
    """
    Restituisce gli impulsi reali simulando la rampa del PSO.
    θ is in degrees.
    """
    theta_corrected = np.array(theta_corrected)

    # --- 1) Calcola tempo necessario per accelerare ---
    t_acc = omega_target / accel
    theta_acc = 0.5 * accel * t_acc**2

    pulses_per_deg = counts_per_rev / 360.0

    real_pulses = []

    for theta in theta_corrected:

        # Caso 1: angolo cade dentro la rampa di accelerazione
        if theta <= theta_acc:
            # θ = ½ a t^2  →  t = sqrt(2θ/a)
            t = np.sqrt(2 * theta / accel)

        # Caso 2: angolo cade in regime costante
        else:
            t_const = (theta - theta_acc) / omega_target
            t = t_acc + t_const

        # Il PSO genera impulsi a frequenza istantanea
        # In regime costante: pulses = θ * pulses_per_deg
        # In rampa: devi convertirlo come θ(t)
        pulses = theta * pulses_per_deg
        real_pulses.append(int(np.round(pulses)))

    return np.array(real_pulses)
