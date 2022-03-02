import gym
import numpy as np
import random


class QLearning:
    def __init__(self, epsilon=0.1, gamma=0.8, alpha=0.1) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = gym.make("Blackjack-v1")
        self.Q = np.zeros(
            (
                self.env.observation_space[0].n,
                self.env.observation_space[1].n,
                self.env.observation_space[2].n,
                self.env.action_space.n
            )
        )

    def get_observation_int(self, observation):
        return observation[0], observation[1], 1 if observation[2] else 0

    def epsilon_greedy_policy(self, observation):
        # exploration
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        # exploitation
        else:
            return self.greedy_policy(observation)

    def greedy_policy(self, observation, debug=False):
        hit = self.Q[self.get_observation_int(observation)][1]
        stick = self.Q[self.get_observation_int(observation)][0]

        if debug:
            print(hit, stick)

        if hit > stick:
            return 0
        elif stick > hit:
            return 1
        else:
            return self.env.action_space.sample()

    def train(self, n_episode):
        # Training
        observation = self.env.reset()
        for i in range(n_episode):
            if i % 10000 == 0:
                print(i)
            done = False
            while not done:
                action = self.epsilon_greedy_policy(observation)
                next_observation, reward, done, info = self.env.step(action)

                maxvalue = max(self.Q[self.get_observation_int(observation)][0], self.Q[self.get_observation_int(observation)][1])
                self.Q[self.get_observation_int(observation)][action] = (1 - self.alpha) * self.Q[self.get_observation_int(observation)][action] + self.alpha * (reward + self.gamma * maxvalue)
                
                # print(observation, action, self.Q[self.get_observation_int(observation)][action])
                # print(self.Q)
                observation = next_observation

            observation = self.env.reset()
        # print(self.Q)

    def test(self, n_episode):
        # Test
        win, draw, loss= 0, 0, 0
        observation = self.env.reset()
        for _ in range(n_episode):
            done = False
            reward = 0
            while not done:
                action = self.greedy_policy(observation, debug=True)
                # action = self.env.action_space.sample()
                next_observation, reward, done, info = self.env.step(action)
                print(next_observation, reward)

                observation = next_observation

            if reward == 0:
                draw += 1
            elif reward > 0:
                win += 1
            else:
                loss+=1
            
            observation = self.env.reset()
            
        print(win/ n_episode, draw/n_episode, loss/n_episode)

if __name__ == "__main__":
    qlearning = QLearning()
    qlearning.train(100000)
    qlearning.test(1000)
    print(qlearning.Q)