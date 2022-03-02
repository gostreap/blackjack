from base_learning import BaseLearning


class TemporalDifference(BaseLearning):

    def train(self, n_episode):
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

                self.Q[observation][action] += alpha * (reward - self.Q[observation][action])
                observation = next_observation

            observation = self.env.reset()

if __name__ == "__main__":
    qlearning = TemporalDifference("v2")
    qlearning.train(20000)
    qlearning.test(20000)