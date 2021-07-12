from mcs_zombies import monte_carlo
from animate_zombies import plot_animate
from config import START

# Please check config.py before starting the simulation
# The START config parameter determines if the simulation
# produces a Monte Carlo Simulation or an animated plot.

def start_simulation(s):
    if s == 'mcs':
        monte_carlo()
    elif s == 'animation':
        plot_animate()
    else:
        print("Please choose either 'mcs' or 'animation' "
              "to start the simulation.")

start_simulation(START)