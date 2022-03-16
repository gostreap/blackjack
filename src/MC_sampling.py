from collections import defaultdict

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from env_v2 import BlackjackDoubleDownEnv
from q_learning import *


def policy(observation):
    score, dealer_score, usable_ace, can_double_down = observation
    return 0 if score >= 20 else 1


def basic_strategy(observation):
    player_total, dealer_value, soft, can_double_down = observation

    if 4 <= player_total <= 8:
        return 1
    if player_total == 9:
        if dealer_value in [1, 2, 7, 8, 9, 10] or can_double_down == 0:
            return 1
        return 2
    if player_total == 10:
        if dealer_value in [1, 10] or can_double_down == 0:
            return 1
        return 2
    if player_total == 11:
        if dealer_value == 1 or can_double_down == 0:
            return 1
        return 2
    if soft:
        # we only double soft 12 because there's no splitting
        if player_total in [12, 13, 14]:
            if dealer_value in [5, 6] and can_double_down == 1:
                return 2
            return 1
        if player_total in [15, 16]:
            if dealer_value in [4, 5, 6] and can_double_down == 1:
                return 2
            return 1
        if player_total == 17:
            if dealer_value in [3, 4, 5, 6] and can_double_down == 1:
                return 2
            return 1
        if player_total == 18:
            if dealer_value in [3, 4, 5, 6] and can_double_down == 1:
                return 2
            if dealer_value in [2, 7, 8]:
                return 0
            return 1
        if player_total >= 19:
            return 0

    else:
        if player_total == 12:
            if dealer_value in [1, 2, 3, 7, 8, 9, 10]:
                return 1
            return 0
        if player_total in [13, 14, 15, 16]:
            if dealer_value in [2, 3, 4, 5, 6]:
                return 0
            return 1

        if player_total >= 17:
            return 0


def mc_prediction(policy, env, num_episodes, discount_factor=0.5):

    sum_dict = defaultdict(float)
    count_dict = defaultdict(float)

    # The final value function
    V = defaultdict(float)

    for i_episode in range(num_episodes):
        observation = env.reset()

        episodes = []
        done = False
        while not done:
            action = policy(observation)
            next_observation, reward, done, _ = env.step(action)
            episodes.append((observation, action, reward))

            observation = next_observation

        # obtain unique observation set
        observations = set([x[0] for x in episodes])
        for i, observation in enumerate(observations):

            idx = episodes.index([episode for episode in episodes if episode[0] == observation][0])

            Q = sum([episode[2] * discount_factor**i for episode in episodes[idx:]])

            obs = (observation[0], observation[1])
            sum_dict[obs] += Q
            count_dict[obs] += 1.0

            V[obs] = sum_dict[obs] / count_dict[obs]

    return V


def plot_surface(X, Y, Z, title):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel("Player Sum")
    ax.set_ylabel("Dealer Showing")
    ax.set_zlabel("Value")
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    fig.colorbar(surf)
    plt.savefig("./figures/" + title + ".png")


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = 11  # min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1])], 2, np.dstack([X, Y]))
    plot_surface(X, Y, Z_ace, title)


if __name__ == "__main__":
    env = BlackjackDoubleDownEnv()
    model = QLearning("v2", epsilon=0.2, gamma=0.14)
    model.train(1000000)

    def tablePrintResult(integer):
        if integer == 0:
            return "S"

        elif integer == 1:
            return "H"
        else:
            return "DD"

    num_row, num_column = 17, 10
    table = np.zeros((num_row, num_column))
    table_str_hard = np.empty_like(table, dtype=str)

    for i in range(num_row):
        for j in range(num_column):
            observation = (4 + i, 1 + j, 0, 1)
            table_str_hard[i, j] = tablePrintResult(model.greedy_policy(observation))

    df = pd.DataFrame(table_str_hard)
    df.to_csv("./figures/table_strategy_hard.csv")

    num_row_soft, num_column_soft = 9, 10
    table = np.zeros((num_row_soft, num_column_soft))
    table_str_hard = np.empty_like(table, dtype=str)

    for i in range(num_row_soft):
        for j in range(num_column_soft):
            observation = (i + 13, 1 + j, 1, 1)
            table_str_hard[i, j] = tablePrintResult(model.greedy_policy(observation))

    df = pd.DataFrame(table_str_hard)
    df.to_csv("./figures/table_strategy_soft.csv")

    num_row, num_column = 17, 10
    table = np.zeros((num_row, num_column))
    table_str_hard = np.empty_like(table, dtype=str)

    for i in range(num_row):
        for j in range(num_column):
            observation = (4 + i, 1 + j, 0, 1)
            table_str_hard[i, j] = tablePrintResult(basic_strategy(observation))

    df = pd.DataFrame(table_str_hard)
    df.to_csv("./figures/table_basic_strategy_hard.csv")

    num_row_soft, num_column_soft = 9, 10
    table = np.zeros((num_row_soft, num_column_soft))
    table_str_hard = np.empty_like(table, dtype=str)

    for i in range(num_row_soft):
        for j in range(num_column_soft):
            observation = (i + 13, 1 + j, 1, 1)
            table_str_hard[i, j] = tablePrintResult(basic_strategy(observation))

    df = pd.DataFrame(table_str_hard)
    df.to_csv("./figures/table_basic_strategy_soft.csv")

    # V_trained_policy = mc_prediction(model.greedy_policy, env, num_episodes = 10000)
    # V_basic_policy = mc_prediction(basic_strategy, env, num_episodes = 10000)

    # plot_value_function(V_trained_policy, title = "Trained Policy Value Function")
    # plot_value_function(V_basic_policy, title = "Basic Policy Value Function")
