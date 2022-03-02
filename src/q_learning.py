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

                maxvalue = max(self.Q[next_observation][0], self.Q[next_observation][1])
                self.Q[observation][action] = (1 - alpha) * self.Q[observation][action] + alpha * (
                    reward + self.gamma * maxvalue
                )
                observation = next_observation

            observation = self.env.reset()


if __name__ == "__main__":
    qlearning = QLearning("v2", epsilon=0.05, gamma=0.2)
    qlearning.train(500000)
    qlearning.test(20000)
