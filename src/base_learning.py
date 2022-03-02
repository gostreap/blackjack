import random

import gym
import numpy as np

from env_doubledown import BlackjackDoubleDownEnv


class BaseLearning:
    def __init__(self, env, epsilon=0.1, gamma=0.5) -> None:
        if env == "v1":
            self.env = gym.make("Blackjack-v1")
            self.Q = np.zeros(
                (
                    self.env.observation_space[0].n,
                    self.env.observation_space[1].n,
                    self.env.observation_space[2].n,
                    self.env.action_space.n,
                )
            )
        elif env == "doubledown":
            self.env = BlackjackDoubleDownEnv()
            self.Q = np.zeros(
                (
                    self.env.observation_space[0].n,
                    self.env.observation_space[1].n,
                    self.env.observation_space[2].n,
                    self.env.observation_space[3].n,
                    self.env.action_space.n,
                )
            )
        self.gamma = gamma
        self.epsilon = epsilon
        self.count = self.Q.copy()

    def get_observation_int(self, observation):
        if len(observation) == 3: # env = "v1"
            return observation[0], observation[1], 1 if observation[2] else 0
        elif len(observation) == 4:
            return observation[0], observation[1], 1 if observation[2] else 0, 1 if observation[3] else 0

    def epsilon_greedy_policy(self, observation):
        # exploration
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        # exploitation
        else:
            return self.greedy_policy(observation)

    def greedy_policy(self, observation):
        return np.argmax(self.Q[self.get_observation_int(observation)])
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