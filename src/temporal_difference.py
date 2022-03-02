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

                observation_int = self.get_observation_int(observation)

                self.count[observation_int][action] += 1
                alpha = 1 / self.count[observation_int][action]

                self.Q[observation_int][action] += alpha * (reward - self.Q[observation_int][action])
                observation = next_observation

            observation = self.env.reset()

if __name__ == "__main__":
    qlearning = TemporalDifference()
    qlearning.train(20000)
    qlearning.test(20000)