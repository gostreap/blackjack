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

                self.count[observation][action] += 1
                alpha = 1 / self.count[observation][action]

                maxvalue = max(self.Q[next_observation])
                self.Q[observation][action] = (1 - alpha) * self.Q[observation][action] + alpha * (
                    reward + self.gamma * maxvalue
                )
                observation = next_observation

            observation = self.env.reset()


if __name__ == "__main__":
    model = QLearning("v3", epsilon=0.05, gamma=0.2)
    model.train(500000)
    model.test(20000)
