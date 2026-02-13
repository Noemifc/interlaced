#!/usr/bin/env python3
import numpy as np
import math
import argparse
# import pymcsl as mcs

from Tomoscan_PSOtoFPGA import InterlacedScan 

class Dose:
    # ----------------------------------------------------------------------
    # init and parameters
    # ----------------------------------------------------------------------
    def __init__(
        self,
        InterlacedRotationStart=0.0,          # r/w
        InterlacedNumberOfRotation=4,         # r/w  (K)
        InterlacedNumAnglesPerRotation=32,    # r/w  (N)
        PSOCountsPerRotation=20000,
        PSOPulsePerRotation=358818,           # how many ticks the trigger counter sees per *360
        RotationDirection=0,
        RotationAccelTime=0.15,
        ExposureTime=0.01,                    # r/w
        readout=0.01,
        readout_margin=1,
        SpeedDegPerSec=60.0,                  # r/w
        MinStepTarget=0.0,                    # r/w
    ):
        # ----------------------------
        # PV-like (r/w)
        # ----------------------------
         self




      
        self.delta_theta_min()



        # ----------------------------
        # Dose (Gy)
        # ----------------------------
      
     


      

      def relative_dose(self):


      def absolute_dose(self):

      def simulate_shutter(self, delta_theta_min)





        
  
