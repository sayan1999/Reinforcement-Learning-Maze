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

# load a previously trained model
if test == Test.LOAD_DEEP_Q:
    model = models.QReplayNetworkModel(game, name="/content/drive/My Drive/QReplayNetworkModel", load=True)

# # compare learning speed (cumulative rewards and win rate) of several models in a diagram
# if test == Test.SPEED_TEST_1:
#     rhist = list()
#     whist = list()
#     names = list()

#     models_to_run = [4]

#     for model_id in models_to_run:
#         logging.disable(logging.WARNING)
#         model = models.QReplayNetworkModel(game, name="QReplayNetworkModel")

#         r, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, exploration_decay=0.999, learning_rate=0.10,
#                         episodes=30)
#         rhist.append(r)
#         whist.append(w)
#         names.append(model.name)

#     f, (rhist_ax, whist_ax) = plt.subplots(2, len(models_to_run), sharex="row", sharey="row", tight_layout=True)

#     for i in range(len(rhist)):
#         rhist_ax[i].set_title(names[i])
#         rhist_ax[i].set_ylabel("cumulative reward")
#         rhist_ax[i].plot(rhist[i])

#     for i in range(len(whist)):
#         whist_ax[i].set_xlabel("episode")
#         whist_ax[i].set_ylabel("win rate")
#         whist_ax[i].plot(*zip(*(whist[i])))

#     plt.show()

# # run a number of training episodes and plot the training time and episodes needed in histograms (time consuming)
# if test == Test.SPEED_TEST_2:
#     runs = 10

#     epi = list()
#     nme = list()
#     sec = list()

#     models_to_run = [4]

#     for model_id in models_to_run:
#         episodes = list()
#         seconds = list()

#         logging.disable(logging.WARNING)
#         for r in range(runs):
#             if model_id == 4:
#                 model = models.QReplayNetworkModel(game, name="QReplayNetworkModel")

#             _, _, e, s = model.train(stop_at_convergence=True, discount=0.90, exploration_rate=0.10,
#                         exploration_decay=0.999, learning_rate=0.10, episodes=1000)

#             print(e, s)

#             episodes.append(e)
#             seconds.append(s.seconds)

#         logging.disable(logging.NOTSET)
#         logging.info("model: {} | trained {} times | average no of episodes: {}| average training time {}".
#                         format(model.name, runs, np.average(episodes), np.sum(seconds) / len(seconds)))

#         epi.append(episodes)
#         sec.append(seconds)
#         nme.append(model.name)

#     f, (epi_ax, sec_ax) = plt.subplots(2, len(models_to_run), sharex="row", sharey="row", tight_layout=True)

#     for i in range(len(epi)):
#         epi_ax[i].set_title(nme[i])
#         epi_ax[i].set_xlabel("training episodes")
#         epi_ax[i].hist(epi[i], edgecolor="black")

#     for i in range(len(sec)):
#         sec_ax[i].set_xlabel("seconds per episode")
#         sec_ax[i].hist(sec[i], edgecolor="black")

#     plt.show()

game.render(Render.MOVES)
# game.play(model, start_cell=(0, 0))
# game.play(model, start_cell=(2, 5))
game.play(model, start_cell=(0, 0))

plt.show()  # must be placed here else the image disappears immediately at the end of the program