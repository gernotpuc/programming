# Model Parameters
# Partly inspired by Munz et al. (2009).
# For further explanations and possible options, please see section 3 of the Report.

START = 'mcs' # Mandatory; Options: 'animation' | 'mcs'
P = 100        # Mandatory
Z = 50         # Mandatory
Pv = 1          # Mandatory
T = 300 # Mandatory
MCS_RUNS = 100 # Mandatory

MOVEMENT = 'random' # Mandatory; Options: 'random' | 'hunt'
IMMUNITY = True  # Mandatory; Options: True | False
ZOMBIE_TYPE = 'dawn' # Optional; Options: 'dawn' | '28days'
WEAPON = 'gun' # Optional; Options: 'machete' | 'gun' | 'none'
TRANSMISSION = 'bite' # Optional; Options: 'bite'| 'spit' | 'air'
