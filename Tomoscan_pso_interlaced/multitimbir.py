 
    for i in range(self.num_angles):
        for g in range(self.K_interlace):
            loop = self.bit_reverse(g, bits)  # ordine loop TIMBIR (0..self.K_interlace-1 permutato)
            idx = i * self.K_interlace + loop  # 0..self.num_angles*self.K_interlace-1 (tutti unici)
            angle_deg = idx * 360.0 / (self.num_angles * self.K_interlace)  # angoli unici in [0, 360)
            theta.append(angle_deg)
            group_indices.append(loop)

    theta = np.array(theta, dtype=float)
    group_indices = np.array(group_indices, dtype=int)

     
