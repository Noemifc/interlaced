

encoder_multiply =   pv_counts     = PV("2bmb:TomoScan:PSOCountsPerRotation") # Numero di impulsi per giro del PSO
raw_delta_encoder_counts = pulse_per_deg = counts_per_rev / 360.0  



delta_encoder_counts =   pulse_timbir

passo in gradi × counts_per_degree = passo in encoder counts
'''

   # Compute the actual delta to keep each interval an integer number of encoder counts
encoder_multiply = float(self.epics_pvs['PSOCountsPerRotation'].get()) / 360.       #single count           # pulse_per_deg  quanti impulsi corrispondono a un grado , cioè quanti encoder counts corrispondono a un grado di rotazione
raw_delta_encoder_counts = self.rotation_step * encoder_multiply                              # quanti impulsi dovrebbe fare l’encoder per quel passo (rotation_step =rotation_step = passo angolare in gradi)
delta_encoder_counts = round(raw_delta_encoder_counts)          #rounding       pulse_timbir                    # arrotonda all’impulso intero più vicino


# che cosa si intende per rotation step perche arrotonda 
# sara' una nuova routine o si andra' a riconnettee a questa
# ----------------------------------------------------
self.epics_pvs['PSOEncoderCountsPerStep'].put(delta_encoder_counts)  # ogni step di rotazione vale X counts 
# Change the rotation step Python variable and PV
self.rotation_step = delta_encoder_counts / encoder_multiply #trasformo counts in angolo reale per vedere a quanti gradi reali corrispondono nel reale 
self.epics_pvs['RotationStep'].put(self.rotation_step)  # passo in gradi mandato ad episcs che viene utilizzato 

#---------- Se ho capito -------------------------------------------------------------------------------------------------------------

encoder_multiply = float(self.epics_pvs['PSOCountsPerRotation'].get()) / 360.   #prende single count
raw_delta_encoder_counts = self.angles_timbir * encoder_multiply                # converti in impulsi relate all angoloideale fornito  
delta_encoder_counts = round(raw_delta_encoder_counts)                          # arrotonda rispetto all' impulso piu' vicino 

# riporta ad Epics 

self.epics_pvs['PSOEncoderCountsPerStep'].put(delta_encoder_counts)  # ogni step di rotazione vale X counts 
self.angles_timbir = delta_encoder_counts / encoder_multiply         # perche riprende qui di nuovo rotation steps aka angles_timbir


# FPGA import self.epics_pvs['RotationStep'].put(self.angles_timbir)  # passo in gradi mandato ad episcs che viene utilizzato 





# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TOMOSCAN FUNCTION : compute_positions_PSO
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

''' questo metodo prende la configurazione desiderata dello scan (passo angolare, numero di angoli, accelerazione) 
e calcola tutto ciò che serve per il motore PSO: angoli, velocità, distanze di accelerazione (“taxi”), start e stop reali.'''
 def compute_positions_PSO(self):
        '''Computes several parameters describing the fly scan motion.
        Computes the spacing between points, ensuring it is an integer number
        of encoder counts.
        Uses this spacing to recalculate the end of the scan, if necessary.
        Computes the taxi distance at the beginning and end of scan to allow
        the stage to accelerate to speed.
        Assign the fly scan angular position to theta[]
        '''
        overall_sense, user_direction = self._compute_senses()                                        # rotazione che puo' essere in 2 sensi scelto by user
        
        # Compute the actual delta to keep each interval an integer number of encoder counts
        encoder_multiply = float(self.epics_pvs['PSOCountsPerRotation'].get()) / 360.                 # quanti impulsi corrispondono a 1 e PSOCountsPerRotation = numero totale di impulsi encoder per 360
        raw_delta_encoder_counts = self.rotation_step * encoder_multiply                              # passo desiderato in encoder counts, non ancora intero
        delta_encoder_counts = round(raw_delta_encoder_counts)                                        # passo arrotondato a intero (l’encoder deve lavorare con numeri interi)
        if abs(raw_delta_encoder_counts - delta_encoder_counts) > 1e-4:
            log.warning('  *** *** *** Requested scan would have used a non-integer number of encoder counts.')       
            log.warning('  *** *** *** Calculated # of encoder counts per step = {0:9.4f}'.format(raw_delta_encoder_counts))
            log.warning('  *** *** *** Instead, using {0:d}'.format(delta_encoder_counts))            # Se l’arrotondamento introduce una differenza significativa (maggiore di 0.0001)
''' Salva il passo corretto nei PV EPICS (controllo remoto del motore) Aggiorna self.rotation_step in gradi tenendo conto dell’arrotondamento'''
     
        self.epics_pvs['PSOEncoderCountsPerStep'].put(delta_encoder_counts)
        # Change the rotation step Python variable and PV
        self.rotation_step = delta_encoder_counts / encoder_multiply
        self.epics_pvs['RotationStep'].put(self.rotation_step)

''' Calcola il tempo per frame (compute_frame_time()) e la velocità del motore:
motor_speed = gradi/sec necessari per coprire rotation_step in time_per_angle.  '''
                
        # Compute the time for each frame
        time_per_angle = self.compute_frame_time()
        self.motor_speed = np.abs(self.rotation_step) / time_per_angle


'''  Legge il tempo di accelerazione impostato RotationAccelTime e calcola la distanza di accelerazione (accel_dist) usando  s=1/2 vt 
serve a sapere quanto spazio serve per arrivare alla velocità costante '''
        # Get the distance needed for acceleration = 1/2 a t^2 = 1/2 * v * t
        motor_accl_time = float(self.epics_pvs['RotationAccelTime'].get()) # Acceleration time in s    as pv ACC
        accel_dist = motor_accl_time / 2.0 * float(self.motor_speed) 

''' Determina il nuovo punto di partenza della scansione (rotation_start_new):
Se senso positivo → punto di partenza normale.
Altrimenti → regola lo start per compensare la lettura (“readout margin”).'''
         
     
#------------------------------



# Make taxi distance an integer number of measurement deltas >= accel distance
        # Add 1/2 of a delta to ensure that we are really up to speed.
        if overall_sense>0:
            self.rotation_start_new = self.rotation_start       #---------> rotation star e' taxi start ? conteggi da inviare FPGA
        else:
            self.rotation_start_new  = self.rotation_start-(2-self.readout_margin)*self.rotation_step

''' Calcola la distanza “taxi”, cioè quanto deve muoversi il motore prima di iniziare la scansione reale:
Arrotonda alla quantità intera di passi.
Salva il risultato nei PV EPICS (PSOStartTaxi).'''
                if self.rotation_step > 0:
            taxi_dist = math.ceil(accel_dist / self.rotation_step + 0.5) * self.rotation_step 
        else:
            taxi_dist = math.floor(accel_dist / self.rotation_step - 0.5) * self.rotation_step 
        self.epics_pvs['PSOStartTaxi'].put(self.rotation_start_new - taxi_dist * user_direction)


''' Calcola il punto finale della scansione (rotation_stop):
Ultimo angolo = start + numero di angoli × passo.
Aggiunge la distanza “taxi” finale per decelerare.
Salva anche questo nei PV EPICS (PSOEndTaxi).
   '''     
        #Where will the last point actually be?
        self.rotation_stop = (self.rotation_start_new
                                + (self.num_angles - 1) * self.rotation_step)
        self.epics_pvs['PSOEndTaxi'].put(self.rotation_stop + taxi_dist * user_direction)

''' Assegna tutti gli angoli della scansione in un array theta[]:
parrtendo da rotation_start
Incrementando di rotation_step
Per num_angles punti.
  '''      
        # Assign the fly scan angular position to theta[]
        self.theta = self.rotation_start + np.arange(self.num_angles) * self.rotation_step







# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TOMOSCAN FUNCTION : compute_senses
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''Definisce una funzione membro (self) che calcola il verso del movimento.
Serve a capire due cose:
La direzione rispetto al contatore dell’encoder (incrementa o decrementa).
La direzione rispetto all’utente (cioè da dove parte a dove arriva il movimento).'''


    def _compute_senses(self):
        '''Computes whether this motion will be increasing or decreasing encoder counts.
        
        user direction, overall sense.
        '''
        # Encoder direction compared to dial coordinates  serve a capire come l’encoder incrementa rispetto alla rotazione fisica.
        encoder_dir = 1 if self.epics_pvs['PSOCountsPerRotation'].get() > 0 else -1  #Legge il PV (Process Variable) di EPICS PSOCountsPerRotation e il numero di conteggi per rotazione è positivo → encoder_dir = +1
       
        
        # Get motor direction (dial vs. user); convert (0,1) = (pos, neg) to (1, -1)
        motor_dir = 1 if self.epics_pvs['RotationDirection'].get() == 0 else -1   #La logica interna della macchina usa 0 per positivo, 1 per negativo. Qui la convertiamo in +1 o -1 per poter moltiplicare facilmente dopo.
      ''' Controlla la direzione voluta dall’utente:

Se rotation_stop > rotation_start, l’utente vuole un movimento positivo.
Altrimenti, negativo.
Serve per distinguere il movimento reale da quello relativo all’encoder.
      '''
    
        # Figure out whether motion is in positive or negative direction in user coordinates
        user_direction = 1 if self.rotation_stop > self.rotation_start else -1
''' Stampa sul log per debug i valori calcolati: (encoder_dir, motor_dir, user_direction).
Utile per capire se i tre segnali sono coerenti e se la moltiplicazione finale darà la direzione corretta.
'''
        
        # Figure out overall sense: +1 if motion in + encoder direction, -1 otherwise
        log.debug((encoder_dir, motor_dir, user_direction))

'''Calcola il verso complessivo del movimento come prodotto:

encoder_dir * motor_dir * user_direction

Risultato +1 → il movimento aumenta i conteggi dell’encoder.

Risultato -1 → il movimento diminuisce i conteggi.

Ritorna anche user_direction separatamente, utile se vuoi sapere solo la direzione "logica" dell’utente.

'''
        return user_direction * motor_dir * encoder_dir, user_direction
        
















#if __name__ == "__main__":
    
