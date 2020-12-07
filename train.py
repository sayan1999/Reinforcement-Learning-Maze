import logging, os
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np

import models
from environment.maze import Maze, Render
from mazeReader.maze import readMaze

logging.basicConfig(format="%(levelname)-8s: %(asctime)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

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


# configuration setup
mazefile='mazeData/test_12_12_1.in'
modelpath='/content/drive/My Drive/QReplayNetworkModel'
test = Test.DEEP_Q  # which test to run
episodesPerNet = 4

def canload(modelpath):
    if (True if os.path.isfile(modelpath) else False):
        print('Loading old model for retrain')
        return True
    print('Initializing new model')
    return False

load = canload(modelpath)
maze, nets = readMaze(mazefile)
games = [Maze(maze, *net) for net in nets]

# training
for netID, game in enumerate(games):

    if test == Test.DEEP_Q:
        ## uncomment to turn GUI off
        # game.render(Render.TRAINING)
        model = models.QReplayNetworkModel(game, name=modelpath, load=load)
        h, w, _, _ = model.train(discount=0.80, exploration_rate=0.10, episodes=episodesPerNet, max_memory=maze.size * 4,
                                stop_at_convergence=True)
        print(f'Net ID: {netID} completed')