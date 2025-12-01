"""
2bmb:TomoScan:PSOStartTaxi = start_taxi = -0.749939 = Posizione di inizio taxi; distanza necessaria alla rotazione per accelerare prima della prima proiezione utile
= end_taxi = 180.735 = Posizione di fine taxi; distanza necessaria per la decelerazione dopo l'ultima proiezione utile
2bmb:TomoScan:PSOEndTaxi = taxi_correct = dati gli angoli voluti ( impulsi voluti ) , restituisce gli angoli effettivamente ottenuti ( e impulsi corrispondenti ) poiche' considera la correzione di taxi
Taxi correction = correggere gli angoli + convertirli in impulsi considerando:

il tratto di accelerazione (start taxi)

il tratto di decelerazione (end taxi)

la velocità di rotazione reale (omega)

eventuali differenze dovute alla dinamica del motore

[start_taxi] —— (accelerazione) —— [0°] —— (angoli utili) —— [180°] —— (decelerazione) —— [end_taxi_position]

  conversione impulsi/angoli è consistente con un encoder da 11.840.200 impulsi/giro

"""
from epics import PV

# PV corretti
pv_start_taxi = PV("2bmb:TomoScan:PSOStartTaxi")
pv_end_taxi   = PV("2bmb:TomoScan:PSOEndTaxi")
pv_counts     = PV("2bmb:TomoScan:PSOCountsPerRotation")

# lettura valori dai PV
start_taxi     = pv_start_taxi.get()      # es: -0.749939
end_taxi       = pv_end_taxi.get()        # es: 180.735 -> ma tu usi solo l'extra (0.735)
counts_per_rev = pv_counts.get()          # es: 1.18402e7 (vero per 2-BM-B)

# array angoli ideali  o altro
angles_deg = [0, 0.1, 0.2]

def taxi_correct(angles_deg, start_taxi, end_taxi, counts_per_rev):
    # conversione angoli-impulsi
    pulse_per_deg = counts_per_rev / 360.0

    theta_corrected = []
    pulses_corrected = []

    # correzione start taxi: shift angolare
    for theta in angles_deg:
        theta_corr = theta + abs(start_taxi)
        theta_corrected.append(theta_corr)
        pulses_corrected.append(theta_corr * pulse_per_deg)

    # correzione fine taxi
    theta_end_corrected = 180.0 + end_taxi
    pulses_end_corrected = theta_end_corrected * pulse_per_deg

    return pulses_corrected, pulses_end_corrected, theta_corrected, theta_end_corrected


# ===========================

pulses_corrected, pulses_end_corrected, theta_corrected, theta_end_corrected = taxi_correct(
    angles_deg, start_taxi, end_taxi, counts_per_rev
)

print("theta_corrected:", theta_corrected)
print("pulses_corrected:", pulses_corrected)
print("theta_end_corrected:", theta_end_corrected)
print("pulses_end_corrected:", pulses_end_corrected)
