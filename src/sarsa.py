from base_learning import BaseLearning


class Sarsa(BaseLearning):
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
                next_action = self.epsilon_greedy_policy(next_observation)

                self.count[observation][action] += 1
                alpha = 1 / self.count[observation][action]

                # print(self.Q.shape, observation, next_action)
                self.Q[observation][action] = (1 - alpha) * self.Q[observation][action] + alpha * (
                    reward + self.gamma * self.Q[next_observation][next_action]
                )
                observation = next_observation

            observation = self.env.reset()


if __name__ == "__main__":
    qlearning = Sarsa("v1")
    qlearning.train(300000)
    qlearning.test(20000)