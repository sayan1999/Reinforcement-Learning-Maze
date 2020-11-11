import logging
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

# maze = np.array([
#     [0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 1, 1, 0, 1, 0],
#     [0, 1, 0, 1, 0, 0, 0, 0],
#     [1, 0, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 1, 1, 1],
#     [0, 1, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 0]
# ])  # 0 = free, 1 = occupied


maze, nets = readMaze('mazeData/test_12_12_1.in')
games = [Maze(maze, *net) for net in nets]
for game in games:
# # only show the maze
# if test == Test.SHOW_MAZE_ONLY:
#     game.render(Render.MOVES)
#     game.reset()

# # play using random model
# if test == Test.RANDOM_MODEL:
#     game.render(Render.MOVES)
#     model = models.RandomModel(game)
#     game.play(model, start_cell=(0, 0))

# # train using tabular Q-learning
# if test == Test.Q_LEARNING:
#     game.render(Render.TRAINING)
#     model = models.QTableModel(game, name="QTableModel")
#     h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
#                              stop_at_convergence=True)

# # train using tabular Q-learning and an eligibility trace (aka TD-lamba)
# if test == Test.Q_ELIGIBILITY:
#     game.render(Render.TRAINING)
#     model = models.QTableTraceModel(game)
#     h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
#                              stop_at_convergence=True)

# # train using tabular SARSA learning
# if test == Test.SARSA:
#     game.render(Render.TRAINING)
#     model = models.SarsaTableModel(game)
#     h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
#                              stop_at_convergence=True)

# # train using tabular SARSA learning and an eligibility trace
# if test == Test.SARSA_ELIGIBILITY:
#     game.render(Render.TRAINING)  # shows all moves and the q table; nice but slow.
#     model = models.SarsaTableTraceModel(game)
#     h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200,
#                              stop_at_convergence=True)


    # train using a neural network with experience replay (also saves the resulting model)
    if test == Test.DEEP_Q:
        for game in [game]:
            # game.render(Render.TRAINING)
            model = models.QReplayNetworkModel(game)
            h, w, _, _ = model.train(discount=0.80, exploration_rate=0.10, episodes=maze.size * 10, max_memory=maze.size * 4,
                                    stop_at_convergence=True)

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
    model = models.QReplayNetworkModel(game, load=True)

# compare learning speed (cumulative rewards and win rate) of several models in a diagram
if test == Test.SPEED_TEST_1:
    rhist = list()
    whist = list()
    names = list()

    models_to_run = [0, 1, 2, 3, 4]

    for model_id in models_to_run:
        logging.disable(logging.WARNING)
        if model_id == 0:
            model = models.QTableModel(game, name="QTableModel")
        elif model_id == 1:
            model = models.SarsaTableModel(game, name="SarsaTableModel")
        elif model_id == 2:
            model = models.QTableTraceModel(game, name="QTableTraceModel")
        elif model_id == 3:
            model = models.SarsaTableTraceModel(game, name="SarsaTableTraceModel")
        elif model_id == 4:
            model = models.QReplayNetworkModel(game, name="QReplayNetworkModel")

        r, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, exploration_decay=0.999, learning_rate=0.10,
                        episodes=30)
        rhist.append(r)
        whist.append(w)
        names.append(model.name)

    f, (rhist_ax, whist_ax) = plt.subplots(2, len(models_to_run), sharex="row", sharey="row", tight_layout=True)

    for i in range(len(rhist)):
        rhist_ax[i].set_title(names[i])
        rhist_ax[i].set_ylabel("cumulative reward")
        rhist_ax[i].plot(rhist[i])

    for i in range(len(whist)):
        whist_ax[i].set_xlabel("episode")
        whist_ax[i].set_ylabel("win rate")
        whist_ax[i].plot(*zip(*(whist[i])))

    plt.show()

# run a number of training episodes and plot the training time and episodes needed in histograms (time consuming)
if test == Test.SPEED_TEST_2:
    runs = 10

    epi = list()
    nme = list()
    sec = list()

    models_to_run = [0, 1, 2, 3, 4]

    for model_id in models_to_run:
        episodes = list()
        seconds = list()

        logging.disable(logging.WARNING)
        for r in range(runs):
            if model_id == 0:
                model = models.QTableModel(game, name="QTableModel")
            elif model_id == 1:
                model = models.SarsaTableModel(game, name="SarsaTableModel")
            elif model_id == 2:
                model = models.QTableTraceModel(game, name="QTableTraceModel")
            elif model_id == 3:
                model = models.SarsaTableTraceModel(game, name="SarsaTableTraceModel")
            elif model_id == 4:
                model = models.QReplayNetworkModel(game, name="QReplayNetworkModel")

            _, _, e, s = model.train(stop_at_convergence=True, discount=0.90, exploration_rate=0.10,
                        exploration_decay=0.999, learning_rate=0.10, episodes=1000)

            print(e, s)

            episodes.append(e)
            seconds.append(s.seconds)

        logging.disable(logging.NOTSET)
        logging.info("model: {} | trained {} times | average no of episodes: {}| average training time {}".
                        format(model.name, runs, np.average(episodes), np.sum(seconds) / len(seconds)))

        epi.append(episodes)
        sec.append(seconds)
        nme.append(model.name)

    f, (epi_ax, sec_ax) = plt.subplots(2, len(models_to_run), sharex="row", sharey="row", tight_layout=True)

    for i in range(len(epi)):
        epi_ax[i].set_title(nme[i])
        epi_ax[i].set_xlabel("training episodes")
        epi_ax[i].hist(epi[i], edgecolor="black")

    for i in range(len(sec)):
        sec_ax[i].set_xlabel("seconds per episode")
        sec_ax[i].hist(sec[i], edgecolor="black")

    plt.show()

game.render(Render.MOVES)
# game.play(model, start_cell=(0, 0))
# game.play(model, start_cell=(2, 5))
game.play(model, start_cell=(4, 1))

plt.show()  # must be placed here else the image disappears immediately at the end of the program