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
    for env in ["v1", "v2", "v3"]:
        print("##### {} #####".format(env))
        model = Sarsa(env, epsilon=0.05, gamma=0.2)
        model.train(100000)
        model.test(20000)

        num_state = 1
        for dim in model.env.observation_space:
            num_state *= dim.n 
        print("Number of state: {}".format(num_state))
        print("Number of action: {}".format(model.env.action_space.n))