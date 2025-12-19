import matplotlib.pyplot as np 
import numpy 

#parametri 
N-theta = 32
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

    # rank to ensure a valid permutation 0..K-1
    order = np.argsort(raw_vals)
    ranks = np.zeros(K, dtype=int)
    for r, idx in enumerate(order):
        ranks[idx] = r

    return ranks


    def bit_reverse(self, n, bits):
        return int(f"{n:0{bits}b}"[::-1], 2)
      












