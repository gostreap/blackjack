import random

import gym
import numpy as np
from env_v1 import BlackjackEnv

from env_v2 import BlackjackDoubleDownEnv
from env_v3 import BlackjackDoubleDownSplitEnv


class BaseLearning:
    def __init__(self, env, epsilon=0.1, gamma=0.5) -> None:
        if env == "v1":
            self.env = BlackjackEnv()
        elif env == "v2":
            self.env = BlackjackDoubleDownEnv()
        elif env == "v3":
            self.env = BlackjackDoubleDownSplitEnv()
        self.Q = np.zeros(tuple([dim.n for dim in self.env.observation_space] + [self.env.action_space.n]))
        self.gamma = gamma
        self.epsilon = epsilon
        self.count = self.Q.copy()
        self.plot = {"n": [], "reward": [], "win":[], "draw":[], "loss":[]}

    def epsilon_greedy_policy(self, observation):
        # exploration
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        # exploitation
        else:
            return self.greedy_policy(observation)

    def greedy_policy(self, observation):
        return np.argmax(self.Q[observation])

    def train(self, n_episode):
        pass

    def test(self, n_episode):
        win, draw, loss = 0, 0, 0
        total_reward = 0
        total_hand = n_episode
        observation = self.env.reset()

        for i in range(n_episode):
            done = False
            reward = 0
            while not done:
                action = self.greedy_policy(observation)
                next_observation, reward, done, info = self.env.step(action)
                observation = next_observation

            if len(info) == 0:
                total_reward += reward

                if reward > 0:
                    win += 1
                elif reward < 0:
                    loss += 1
                else:
                    draw += 1
            else:
                reward0, reward1 = info["final_reward0"], info["final_reward1"]
                if reward0 is not None:
                    total_reward += reward0
                    if reward0 > 0:
                        win += 1
                    elif reward0 < 0:
                        loss += 1
                    else:
                        draw += 1

                if reward1 is not None:
                    total_hand += 1
                    total_reward += reward1
                    if reward1 > 0:
                        win += 1
                    elif reward1 < 0:
                        loss += 1
                    else:
                        draw += 1

            observation = self.env.reset()

        assert win + draw + loss == total_hand, "Error in win, draw, loss count : {} + {} + {} = {} != {}".format(win, draw, loss, win + draw + loss, total_hand)

        print("win: {} | draw: {} | loss: {}".format(win/ total_hand, draw / total_hand, loss / total_hand))
        print("mean reward: {}".format(total_reward / total_hand))

        return win / total_hand, draw / total_hand, loss / total_hand,  total_reward / total_hand