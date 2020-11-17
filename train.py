import logging, os
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np

import models
from environment.maze import Maze, Render
from mazeReader.maze import readMaze

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)  # Only show messages *equal to or above* this level

class Test(Enum):
    SHOW_MAZE_ONLY = auto()
    RANDOM_MODEL = auto()
    Q_LEARNING = auto()
    Q_ELIGIBILITY = auto()
    SARSA = auto()
    SARSA_ELIGIBILITY = auto()
    DEEP_Q = auto()
    LOAD_DEEP_Q = auto()
    SPEED_TEST_1 = auto()
    SPEED_TEST_2 = auto()

test = Test.DEEP_Q  # which test to run

maze, nets = readMaze('mazeData/test_12_12_1.in')
games = [Maze(maze, *net) for net in nets]

varfile="/content/drive/My Drive/vars"
startnet=0

if os.path.isfile(varfile):
    startnet=int(open(varfile).read().strip().split()[0])
    f=open(varfile, 'w')
modelpath="/content/drive/My Drive/QReplayNetworkModel"
load = True if os.path.isfile(modelpath) else False

for netID, game in enumerate(games):
    if netID < startnet:
        continue
    # train using a neural network with experience replay (also saves the resulting model)
    if test == Test.DEEP_Q:
        # game.render(Render.TRAINING)
        model = models.QReplayNetworkModel(game, name=modelpath, load=load)
        h, w, _, _ = model.train(discount=0.80, exploration_rate=0.10, episodes=3, max_memory=maze.size * 4,
                                stop_at_convergence=True)
        startnet+=1
        open(varfile, 'w+').write(f'{startnet}')
        print(f'Net ID: {netID} completed')

# draw graphs showing development of win rate and cumulative rewards
try:
    h  # force a NameError exception if h does not exist, and thus don't try to show win rate and cumulative reward
    fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
    fig.canvas.set_window_title(model.name)
    ax1.plot(*zip(*w))
    ax1.set_xlabel("episode")
    ax1.set_ylabel("win rate")
    ax2.plot(h)
    ax2.set_xlabel("episode")
    ax2.set_ylabel("cumulative reward")
    plt.show()
except NameError:
    pass