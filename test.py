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
modelpath="QReplayNetworkModel"
test = Test.DEEP_Q

maze, nets = readMaze(mazefile)
games = [Maze(maze, *net) for net in nets]

for game in games:
    game.render(Render.MOVES)
    model=models.QReplayNetworkModel(game, name=modelpath)
    print(game.play(model, start_cell=(8, 8)))
    plt.show()