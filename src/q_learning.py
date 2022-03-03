from base_learning import BaseLearning


class QLearning(BaseLearning):
    def train(self, n_episode, plot=False, plot_interval=10000, plot_test_size=20000):
        # Training
        observation = self.env.reset()
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
    for env in ["v1", "v2", "v3"]:
        print("##### {} #####".format(env))
        model = QLearning(env, epsilon=0.05, gamma=0.2)
        model.train(100000)
        model.test(20000)

        num_state = 1
        for dim in model.env.observation_space:
            num_state *= dim.n 
        print("Number of state: {}".format(num_state))
        print("Number of action: {}".format(model.env.action_space.n))

