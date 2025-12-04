import numpy as np
from epics import PV

'''
tipi di interlacciamento 
# Equally spaced
angles_eq = generate_angles(N_theta=32, method='equally_spaced')

# Golden
angles_golden = generate_angles(N_theta=32, method='golden')



'''






# --------------------------------------------------------------------
# Interlaced  TIMBIR
# --------------------------------------------------------------------

def generate_timbir_angles():
    """
    Genera gli angoli secondo schema TIMBIR con bit-reversal,
    leggendo i PV di tomoscan:
        - 2bmb:TomoScan:NTheta   (numero totale proiezioni)
        - 2bmb:TomoScan:KLoops   (numero loop interlacciati)
    Ritorna:
        angles  -> array numpy float, angoli interlacciati in radianti
        angles_deg -> in gradi
        loops   -> array degli indici di loop
    """

    # ----------------------------------------------------
    # Lettura PV
    # ----------------------------------------------------
    pv_N_theta = PV("2bmb:TomoScan:NTheta")
    pv_K       = PV("2bmb:TomoScan:KLoops")

    N_theta = int(pv_N_theta.get() or 32)
    K       = int(pv_K.get() or 4)

    # ----------------------------------------------------
    # Funzione interna: bit reverse
    # ----------------------------------------------------
    def bit_reverse(x, bits):
        b = f'{x:0{bits}b}'
        return int(b[::-1], 2)

    # ----------------------------------------------------
    # Generazione schema TIMBIR
    # ----------------------------------------------------
    bits = int(np.log2(K))
    angles = []
    loops = []

    for n in range(N_theta):
        base = n * K
        loop = (base // N_theta) % K
        rev = bit_reverse(loop, bits)
        val = base + rev

        theta = (val * 360.0 / N_theta) % 180.0
        angles.append(theta)
        loops.append(loop)

    angles = np.array(angles)
    loops = np.array(loops)

    # Conversione opzionale in radianti
    angles_rad = np.deg2rad(angles)

    return angles_rad, angles, loops


# --------------------------------------------------------------------
# ESEMPIO USO
# --------------------------------------------------------------------
if __name__ == "__main__":
    ang_rad, ang_deg, loops = generate_timbir_angles()
    print("TIMBIR in gradi:")
    print(np.round(ang_deg, 4))
