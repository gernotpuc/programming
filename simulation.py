import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import animation
from array import * # kann weg?????
import pandas as pd
import itertools
from pandas import DataFrame
from itertools import combinations
from scipy.spatial.distance import pdist
import matplotlib.image as mpimg
from config import P,Z,Pv,ZOMBIE_TYPE,IMMUNITY,WEAPON,TRANSMISSION,T,MOVEMENT
from scipy.spatial.distance import cdist
##################################################################
# This code was produced by Gernot Pucher as classwork for the
# class "Programming languages for Data Science" at the University
# of Dundee.
# The implementation of random agent movements was inspired by:
# https://plotly.com/python/random-walk/, 10.07.2021
# The computation of a distance matrix was inspired by:
# https://stackoverflow.com/questions/61569321/
# how-to-calculate-distance-between-n-coordinates-in-python, 10.07.21
# https://stackoverflow.com/questions/464864/
# how-to-get-all-possible-combinations-of-a-list-s-elements, 10.07.21
##################################################################

def init_humans():
    ## Create an initial set of human agents based
    ## on the parameters P (human population).
    humans_xy = []
    h = 0
    for i in range(P):
        x = random.uniform(-100,100)
        #print(x)
        y = random.uniform(-40,40)
        s = 0 # steps made
        h = h +1 # human id
        humans_xy.append((x, y, s, h))
    return(humans_xy)

def init_zombies():
    # Create an initial set of zombie agents based
    # on the parameters Z (zombie population).
    # Based on a list of 4 options,
    # zombies are place in the upper, lower, left or right area
    # of the 2-dimensional space.
    zombies_xy = []
    z = 0
    for i in range(Z):
        rndm_list = [0,1,2,3]
        k = np.random.choice(rndm_list)
        if  (k) == 0:
            x = random.uniform(-100,100)
            y = random.uniform(50,70)
        elif (k) == 1:
            x = random.uniform(-100,100)
            y = random.uniform(-50,-70)
        elif (k) == 2:
            x = random.uniform(110,130)
            y = random.uniform(-50,50)
        elif (k) == 3:
            x = random.uniform(-110,-130)
            y = random.uniform(-50,50)
        s = 0
        z = z +1
        zombies_xy.append((x, y, s, z))
    return (zombies_xy)

def create_agents_df():
    # This function calls the initialisation functions for human
    # and zombie agents. The lists of humans and zombies get
    # assigned to DataFrames and are then merged.
    # The returned output is used by agents_list().

    humans_df = DataFrame(init_humans(), columns=['x', 'y', 'step', 'id'])
    zombies_df = DataFrame(init_zombies(), columns=['x', 'y', 'step', 'id'])
    humans_df['type'] = 'human'
    zombies_df['type'] = 'zombie'
    dfs = [humans_df, zombies_df]
    agents_df = pd.concat(dfs).reset_index(drop=True)
    return (agents_df)



def agents_list():
    # This function appends the results of the stepwise simulation of agent
    # movements, transmissions and kills in a dataframe. This dataframe is
    # then used by the aggregation and visualisation components of
    # mcs_zombies.py and animate_zombies.py

    # Create an empty dataframe to which simulation results will get appended to
    result = pd.DataFrame(columns=['x', 'y', 'step', 'id','type'])
    # An iterable representing the current simulation step
    s = 0
    # Calling the function create_agents_df() to initiate human and zombie agents
    # at step = 0 in the dataframe agents_df
    agents_df = create_agents_df()
    # Initiate agent movements by calling the function move_agents. The step and the
    # initial setup of human and zombie agents are passed to that function.
    # The result from the first step is then added to the initial setup of human
    # and zombie agents.
    agents_df_sim = pd.concat([agents_df,move_agents(agents_df,s)]).reset_index(drop=True)
    # Call the function calc_dist() to calculate a distance matrix between all agents and assign
    # the information if a human got infected or a zombie got killed in that step.
    # The dataframe of moved agents and the step number are passed to the function.
    min_dist = calc_dist(agents_df_sim, s)
    # The dataframe of agents with simulate steps is merged with the results from the
    # calculation of the distance matrix. If the attribute 'infected' got set True for
    # the respective step (can only happen to humans), the type of the agents is changed to zombie.
    # If a zombie is set to be shot, the type of the agent gets changed to 'shot'.
    agents_df_sim = agents_df_sim.merge(min_dist[['infected', 'k', 'type_k']], left_index=True, right_on='k', how='outer')
    agents_df_sim.loc[(agents_df_sim['infected'] == True), 'type'] = 'zombie'
    agents_df_sim.loc[(agents_df_sim['type_k'] == 'shot'), 'type'] = 'shot'
    agents_df_sim = agents_df_sim.drop(columns=['infected', 'k'])
    # Now that the required dataframe with simulated agents has been initialized,
    # the above procedure is repeated for the given number of T simulation steps.
    # The result of each iteration gets append to the result dataframe.
    for i in range (T):
        agents_df_sim = move_agents(agents_df_sim, s)
        min_dist = calc_dist(agents_df_sim,s)
        agents_df_sim = agents_df_sim.merge(min_dist[['infected', 'k', 'type_k']], left_index=True, right_on='k', how = 'outer')
        agents_df_sim.loc[(agents_df_sim['infected'] == True), 'type'] = 'zombie'
        agents_df_sim.loc[(agents_df_sim['type_k'] == 'shot'), 'type'] = 'shot'
        agents_df_sim = agents_df_sim.drop(columns=['k', 'type_k'])
        s = s+1
        result = result.append(agents_df_sim, ignore_index = True)
    return(result)

def move_agents(agents_df_sim,s):
    # This function handles the stepwise movement of human and zombie agents.
    # It is called by agents_list() for every step of the simulation.
    # It takes the list of agents in the simulation and their current
    # x,y coordinates and step numbers as input.
    # The for loop iterates over all agents of the DataFrame agents_df
    # filtered by the current step number s. This is done first for the agent
    # type 'human', then for the agent type 'zombie.
    # The movement for each step is defined as a random choice between [-1 1]
    # for x and y plus the configured speed of humans and zombies. The new
    # coordinates are calculated by adding the computed step to the existing
    # x,y coordinates of agents_df.
    # The new coordinates for x and y as well as the incremented
    # step number are saved and appended to lists for human and zombie agents.
    # These lists are then merged into a common DataFrame which is returned
    # by the function.
    human_step_xy = []
    zombie_step_xy = []
    agents_df=agents_df_sim
    for i, row in agents_df.loc[(agents_df['step'] == s) & (agents_df['type'] =='human')].iterrows():
        # The step movement for x and y of each agent is calculated as the
        # random choice between [-1 1] plus the defined agent speed.
        # To avoid agents moving constantly into a positive direction,
        # the calculated step movement is multiplied with the randn
        # function, which computes random floats from an univariate
        # normal distribution of mean 0 and variance 1.
        x_step = np.random.choice([-1, 1]) + Pv * np.random.randn()
        y_step = np.random.choice([-1, 1]) + Pv * np.random.randn()
        x = row['x'] + np.cumsum(x_step)
        y = row['y'] + np.cumsum(y_step)
        # Avoid agents going out of bounds of the scatterplot.
        if x > 100:
            x = x-1
        elif x < -100:
            x = x+1
        if y > 100:
            y = y-1
        elif y < -100:
            y = y+1
        human_step_xy.append((x, y, s+1, row['id'], row['type']))
    human_step_xy_ = DataFrame(human_step_xy, columns=['x', 'y', 'step', 'id', 'type'])
    mean_x = human_step_xy_["x"].mean()
    mean_y = human_step_xy_["y"].mean()

    for i, row in agents_df.loc[(agents_df['step'] == s) & (agents_df['type'] =='zombie')].iterrows():
        # Same procedure as for humans, with an additional differentiation
        # of the configured ZOMBIE_TYPE and MOVEMENT.
        # ZOMBIE_TYPE causes the speed per step to be different.
        # MOVEMENT determines if zombies move randomly or have a tendency
        # to move towards the average position of humans.
        if ZOMBIE_TYPE == 'dawn':
            if MOVEMENT == 'random':
                x_step = np.random.choice([-1, 1]) + 1 * np.random.randn()
                y_step = np.random.choice([-1, 1]) + 1 * np.random.randn()
                x = row['x'] + np.cumsum(x_step)
                y = row['y'] + np.cumsum(y_step)
            elif MOVEMENT == 'hunt':
                if row['x'] < mean_x:
                    x_step = np.random.choice([-0.2, 1]) + 1 * np.random.randn()
                else:
                    x_step = np.random.choice([-1, 0.2]) + 1 * np.random.randn()
                if row['y'] < mean_y:
                    y_step = np.random.choice([-0.2, 1]) + 1 * np.random.randn()
                else:
                    y_step = np.random.choice([-1, 0.2]) + 1 * np.random.randn()
                x = row['x'] + np.cumsum(x_step)
                y = row['y'] + np.cumsum(y_step)
            if x > 100:
                x = x - 1
            elif x < -100:
                x = x + 1
            if y > 100:
                y = y - 1
            elif y < -100:
                y = y + 1
        elif ZOMBIE_TYPE == '28days':
            if MOVEMENT == 'random':
                x_step = np.random.choice([-1, 1]) + 3 * np.random.randn()
                y_step = np.random.choice([-1, 1]) + 3 * np.random.randn()
                x = row['x'] + np.cumsum(x_step)
                y = row['y'] + np.cumsum(y_step)
            elif MOVEMENT == 'hunt':
                if row['x'] < mean_x:
                    x_step = np.random.choice([-0.2, 1]) + 3 * np.random.randn()
                else:
                    x_step = np.random.choice([-1, 0.2]) + 3 * np.random.randn()
                if row['y'] < mean_y:
                    y_step = np.random.choice([-0.2, 1]) + 3 * np.random.randn()
                else:
                    y_step = np.random.choice([-1, 0.2]) + 3 * np.random.randn()
                x = row['x'] + np.cumsum(x_step)
                y = row['y'] + np.cumsum(y_step)
            if x > 100:
                x = x - 3
            elif x < -100:
                x = x + 3
            if y > 100:
                y = y - 3
            elif y < -100:
                y = y + 3
        else:
            if MOVEMENT == 'random':
                x_step = np.random.choice([-1, 1]) + 2 * np.random.randn()
                y_step = np.random.choice([-1, 1]) + 2 * np.random.randn()
                x = row['x'] + np.cumsum(x_step)
                y = row['y'] + np.cumsum(y_step)
            elif MOVEMENT == 'hunt':
                if row['x'] < mean_x:
                    x_step = np.random.choice([-0.2, 1]) + 2 * np.random.randn()
                else:
                    x_step = np.random.choice([-1, 0.2]) + 2 * np.random.randn()
                if row['y'] < mean_y:
                    y_step = np.random.choice([-0.2, 1]) + 2 * np.random.randn()
                else:
                    y_step = np.random.choice([-1, 0.2]) + 2 * np.random.randn()
                x = row['x'] + np.cumsum(x_step)
                y = row['y'] + np.cumsum(y_step)
            if x > 100:
                x = x - 2
            elif x < -100:
                x = x + 2
            if y > 100:
                y = y - 2
            elif y < -100:
                y = y + 2
        zombie_step_xy.append((x, y, s+1, row['id'], row['type']))
    zombie_step_xy_ = DataFrame(zombie_step_xy, columns=['x', 'y', 'step', 'id', 'type'])
    # Join the resulting lists in a common DataFrame and return it.
    frames_ = [human_step_xy_, zombie_step_xy_]
    agents_df_step = pd.concat(frames_).reset_index(drop=True)
    agents_df_step['x'] = agents_df_step['x'].str[0]
    agents_df_step['y'] = agents_df_step['y'].str[0]
    return(agents_df_step)

def calc_dist(agents_df_sim,s):
    # This function computes the distances between all agents of
    # each simulation step.
    # It is called by the function agents_list for each simulation
    # step. Input parameters are the DataFrame agents_df_sim,
    # which includes the new coorindates of agents
    # after their stepwise movements, as well as the current
    # step number.
    ds = []
    # Filter for the step number for which new step movements
    # were created. Then create a distance matrix for all agents
    # of the same step number with the combinations function of
    # itertools. For these combinations, Euclidean distances are
    # calculated using the pdist function of SciPy.
    agents_df_sim_step = agents_df_sim.loc[agents_df_sim['step'] == s+1]
    coords = agents_df_sim_step[['x','y']].values
    combi_df = pd.DataFrame(itertools.combinations(agents_df_sim_step.index, 2), columns=['k','j'])
    combi_df['dist'] = pdist(coords, 'euclid')
    ds.append(combi_df)
    # The resulting distance matrix is merged with the agent types.
    distmatrix = pd.concat(ds).reset_index(drop=True)
    distmatrix = distmatrix.merge(agents_df_sim_step['type'], left_on='k', right_index=True)
    distmatrix = distmatrix.rename(columns={'type': 'type_k'})
    distmatrix = distmatrix.merge(agents_df_sim_step['type'], left_on='j', right_index=True)
    distmatrix = distmatrix.rename(columns={'type': 'type_j'})
    # The column infected is added and initially set to False.
    distmatrix['infected'] = False
    # The following if conditions handle different simulation configurations.
    # These include transmission types, immunity and weapons used.
    # Based on these conditions and the calculated distances between
    # agents, the attribute 'infected' might be set to True.
    # Transmission condition: The threshold of minimum distance to
    # cause infection.
    if TRANSMISSION == 'bite':
        tr = 2
    elif TRANSMISSION == 'spit':
        tr = 3
    elif TRANSMISSION == 'air':
        tr = 5
    else:
        tr = 2
    # Immunity Condition: If False, every positive distance condition
    # will cause infection. If True, infection will happen based on
    # random.choice.
    if IMMUNITY == False:
        distmatrix.loc[(distmatrix['dist'] < tr) & (distmatrix['type_k'] == 'human') & (
                distmatrix['type_j'] == 'zombie'), 'infected'] = True
        distmatrix.loc[(distmatrix['dist'] < tr) & (distmatrix['type_k'] == 'zombie') & (
            distmatrix['type_j'] == 'human'), 'infected'] = True
    if IMMUNITY == True:
        distmatrix.loc[(distmatrix['dist'] < tr) & (distmatrix['type_k'] == 'human') & (
                distmatrix['type_j'] == 'zombie'), 'infected'] = random.choice([True, False])
        distmatrix.loc[(distmatrix['dist'] < tr) & (distmatrix['type_k'] == 'zombie') & (
            distmatrix['type_j'] == 'human'), 'infected'] = random.choice([True, False])
    # Weapon Conditon: If gun or machete is configured, there is a random choice
    # that the type will turn from zombie to shot.
    sz = ['shot', 'zombie']
    if WEAPON == 'gun':
        distmatrix.loc[(distmatrix['dist'] < 3) & (distmatrix['type_k'] == 'human') & (
                distmatrix['type_j'] == 'zombie'), 'type_j'] = random.choice(sz)
    if WEAPON == 'machete':
        distmatrix.loc[(distmatrix['dist'] < 1) & (distmatrix['type_k'] == 'human') & (
                distmatrix['type_j'] == 'zombie'), 'type_j'] = random.choice(sz)
    # All 'shot' types are included in a list. This status is updated for the
    # respective agent in the distance matrix.
    deadzombies = list(distmatrix['j'].loc[distmatrix['type_j'] == 'shot'])
    distmatrix.loc[distmatrix.k.isin(deadzombies), 'type_k'] = 'shot'
    # The minimal distances are calculated for each agent in the simulation.
    min_dist = distmatrix.loc[distmatrix.groupby('k').dist.idxmin()]
    return(min_dist)
