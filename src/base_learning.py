import random

import gym
import numpy as np


class BaseLearning:
    def __init__(self, epsilon=0.1, gamma=0.5) -> None:
        self.env = gym.make("Blackjack-v1")
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros(
            (
                self.env.observation_space[0].n,
                self.env.observation_space[1].n,
                self.env.observation_space[2].n,
                self.env.action_space.n,
            )
        )
        self.count = self.Q.copy()

    def get_observation_int(self, observation):
        return observation[0], observation[1], 1 if observation[2] else 0

    def epsilon_greedy_policy(self, observation):
        # exploration
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        # exploitation
        else:
            return self.greedy_policy(observation)

    def greedy_policy(self, observation):
        hit = self.Q[self.get_observation_int(observation)][1]
        stick = self.Q[self.get_observation_int(observation)][0]

        if hit > stick:
            return 1
        elif stick > hit:
            return 0
        else:
            return self.env.action_space.sample()

    def train(self, n_episode):
        pass

    def test(self, n_episode):
        win, draw, loss = 0, 0, 0
        observation = self.env.reset()
        for _ in range(n_episode):
            done = False
            reward = 0
            while not done:
                action = self.greedy_policy(observation)
                next_observation, reward, done, info = self.env.step(action)
                observation = next_observation

            if reward == 0:
                draw += 1
            elif reward > 0:
                win += 1
            else:
                loss += 1

            observation = self.env.reset()

        print("win : {} | draw : {} | loss : {}".format(win/ n_episode, draw / n_episode, loss / n_episode))