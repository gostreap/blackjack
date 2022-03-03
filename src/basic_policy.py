from base_learning import BaseLearning
from MC_sampling import basic_strategy


class BasicPolicy(BaseLearning):

    def greedy_policy(self, observation):
        return basic_strategy(observation)

    def train(self, n_episode, plot=False, plot_interval=10000, plot_test_size=20000):
        # Training
        for i in range(n_episode):
            if i % plot_interval == 0:
                print("training, episode {} out of {}".format(i, n_episode))
                if plot:
                    win, draw, loss, reward = self.test(plot_test_size)
                    self.plot["n"].append(i)
                    self.plot["win"].append(win)
                    self.plot["draw"].append(draw)
                    self.plot["loss"].append(loss)
                    self.plot["reward"].append(reward)

            