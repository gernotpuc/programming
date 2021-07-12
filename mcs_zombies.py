import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import matplotlib.image as mpimg
from simulation import agents_list
from config import P,Z,MCS_RUNS

def monte_carlo():
    # This function computes a Monte Carlo simulation.
    # The simulation runs for the number of times defined in MCS_RUNS
    # of config.py. Based on multiple runs of the stochastic simulation,
    # the average number of humans and zombies per step are calculated.
    # This allows to make assumptions about the influence of different
    # parameter configurations on the simulation outcome.
    print("Number of MCS runs set to ", MCS_RUNS)
    stats = pd.DataFrame(columns=['x', 'y', 'step', 'id', 'type', 'infected'])
    # The simulation is executed through calling the agents_list
    # function for the number of times in MCS_RUNS. After each run,
    # results are appended to the DataFrame stats.
    for i in range(MCS_RUNS):
        result = agents_list()
        stats = stats.append(result)
        print(i+1,' of ',MCS_RUNS,' run(s) completed')
    # The average number of agent types ‘human’ and ‘zombie’ grouped by steps is
    # calculated and stored in the DataFrame aggr_stats.
    aggr_stats = stats.groupby(['step','type'],as_index=False).count()
    aggr_stats_avg = aggr_stats['x']/MCS_RUNS
    aggr_stats = aggr_stats.merge(aggr_stats_avg, left_index=True, right_index=True)
    aggr_stats = aggr_stats.rename(columns={'x_y': 'meannr'}).drop(columns=['infected', 'x_x', 'y', 'id'])
    aggr_stats = aggr_stats[aggr_stats.type != 'shot']
    # The average survival rate of humans is calculated.
    survival = aggr_stats.loc[aggr_stats['type'] == 'human', 'meannr'].iloc[-1]
    s_rate = round((survival/P)*100)
    # The average population development rate of zombies is calculated.
    zspread = aggr_stats.loc[aggr_stats['type'] == 'zombie', 'meannr'].iloc[-1]
    z_rate = round((zspread/Z)*100)
    # Plotting a line chart. x is a list of simulation steps.
    # yh is a list of the average number of humans per step.
    # yz is a list of the average number of zombies per step.
    # Humans are plotted as blue lines, zombies as red lines.
    # The number of MCS runs, the avg. human survival rate and
    # the average zombie development are included in the plot title.
    x = list(aggr_stats['step'].unique())
    yh = list(aggr_stats.loc[aggr_stats['type'] == 'human', 'meannr'])
    yz = list(aggr_stats.loc[aggr_stats['type'] == 'zombie', 'meannr'])
    plt.figure(num=3, figsize=(8, 5))
    plt.plot(x, yh,
             color='blue',
             linewidth=1.0,
             linestyle='--', label='Humans'
             )
    plt.plot(x, yz,             color='red',
             linewidth=1.0,
             linestyle='--', label='Zombies')
    plt.legend(fontsize=10, loc='upper left')
    plt.suptitle("Average Number of Humans and Zombies "
                 "after each Simulation Step", fontsize=10)
    plt.title("MCS Runs: {}, Avg. Human Survival Rate: {}%, "
              "Avg. Zombie Development: {}%".format(MCS_RUNS,s_rate, z_rate), fontsize=10)
    plt.show()
