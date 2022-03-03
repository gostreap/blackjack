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
        observation = self.env.reset()

        for _ in range(n_episode):
            done = False
            reward = 0
            while not done:
                action = self.greedy_policy(observation)
                next_observation, reward, done, info = self.env.step(action)
                observation = next_observation
                total_reward += reward

            if reward == 0:
                draw += 1
            elif reward > 0:
                win += 1
            else:
                loss += 1

            observation = self.env.reset()

        print("win: {} | draw: {} | loss: {}".format(win/ n_episode, draw / n_episode, loss / n_episode))
        print("mean reward: {}".format(total_reward / n_episode))