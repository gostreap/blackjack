from base_learning import BaseLearning


class QLearning(BaseLearning):
    def train(self, n_episode):
        # Training
        observation = self.env.reset()
        for i in range(n_episode):
            if i % 50000 == 0:
                print("training, episode {} out of {}".format(i, n_episode))

            done = False
            while not done:
                action = self.epsilon_greedy_policy(observation)
                next_observation, reward, done, info = self.env.step(action)

                observation_int = self.get_observation_int(observation)
                next_observation_int = self.get_observation_int(observation)

                self.count[observation_int][action] += 1
                alpha = 1 / self.count[observation_int][action]

                maxvalue = max(self.Q[next_observation_int][0], self.Q[next_observation_int][1])
                self.Q[observation_int][action] = (1 - alpha) * self.Q[observation_int][action] + alpha * (
                    reward + self.gamma * maxvalue
                )
                observation = next_observation

            observation = self.env.reset()


if __name__ == "__main__":
    qlearning = QLearning(epsilon=0.05, gamma=0.2)
    qlearning.train(500000)
    qlearning.test(20000)
