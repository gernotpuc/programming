import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from config import T
from simulation import agents_list
import requests
import pathlib
def plot_animate():
    # This function handles the animated visualisation of the simulation.
    # It is started through the execution of start.py and the option
    # START = 'animation' in config.py.
    # Simulation results are generated by calling the agents_list function.
    # A scatterplot is created in a for loop for every step of
    # the simulation. The pause function handles the animation speed.

    #################################################################
    # Important: The IMAGE_PATH to the downloaded image 'mall.jpg' is
    # set to be in the same directory as this script. If the image file
    # is for some reason not downloaded directly to the folder of
    # this script, IMAGE_PATH must be set to the folger 'mall.jpg' is
    # stored in.
    # gets downloaded to. As a default, is it saved to the folder
    # where this python script is executed in.
    #################################################################
    # Download the background image for the scatter plot and
    # assign it to the image reader of matplotlib.
    image_url = 'https://raw.githubusercontent.com/gernotpuc/programming/main/mall.jpg'
    mall_img = requests.get(image_url).content
    IMAGE_PATH = str(pathlib.Path(__file__).parent.resolve())
    with open('mall.jpg', 'wb') as handler:
        handler.write(mall_img)
    mall_img = mpimg.imread(IMAGE_PATH+'\mall.jpg')
    # Call the function agents_list to obtain the DataFrame
    # of humans and zombies per simulation step.
    result = agents_list()
    # Setup a scatterplot with blue points representing
    # humans and red points representing zombies.
    fig, ax = plt.subplots()
    xarry, yarry = [], []
    xarryz, yarryz = [], []
    humans = ax.scatter(xarry, yarry, c='blue', label="Humans")
    zombies = ax.scatter(xarryz, yarryz, c='red', label="Zombies")
    plt.imshow(mall_img, extent=[-150, 150, -100, 100], alpha=0.5)
    plt.ylabel("y", fontsize=12)
    plt.xlabel("x", fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.ion()


    for i in range(T+1):
        # For every step of the simulation, retrieve stepwise results
        # from the DataFrame which includes all simulated
        # agents. Every step is visualized in a scatter plot.
        # The numbers of humans and zombies are counted in each step and
        # included in the title of the plot.
        rslt_df = result.loc[(result['step'] == i) & (result['type'] == 'human') & (result['infected'].notnull())]
        nrh = (rslt_df['type'].values == 'human').sum()
        xarry = rslt_df['x']
        yarry = rslt_df['y']
        rslt_df = result.loc[(result['step'] == i) & (result['type'] == 'zombie') & (result['infected'].notnull())]
        nrz = (rslt_df['type'].values == 'zombie').sum()
        xarryz = rslt_df['x']
        yarryz = rslt_df['y']
        humans.set_offsets(np.c_[xarry, yarry])
        zombies.set_offsets(np.c_[xarryz, yarryz])
        plt.title("Step: {}, Humans: {}, Zombies: {}".format(i, nrh, nrz))
        fig.canvas.draw_idle()
        if i == 0:
            plt.pause(5)
        else:
            plt.pause(0.3)
        # Uncomment in case single plots should be saved to local directory
        #plt.savefig(IMAGE_PATH+'\img'+str(i)+'.png')

    plt.waitforbuttonpress()
