import numpy as np

# parametri
N_theta = 32
K = 4

# ------------------------
# Universal bit-reversal permutation for ANY K
# ------------------------
def bit_reverse_generalized(K):
    bits = int(np.ceil(np.log2(K)))
    raw_vals = []

    for x in range(K):
        b = f'{x:0{bits}b}'
        rev = int(b[::-1], 2)
        raw_vals.append(rev)

    order = np.argsort(raw_vals)
    ranks = np.zeros(K, dtype=int)

    for r, idx in enumerate(order):
        ranks[idx] = r

    return ranks


# stampa la lista
values = bit_reverse_generalized(K)
print("Bit-reversal generalized:", values.tolist())











