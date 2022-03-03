from base_learning import BaseLearning
from MC_sampling import basic_strategy
class BasicPolicy(BaseLearning):
    def greedy_policy(self, observation):
        return basic_strategy(observation)


basicpolicy = BasicPolicy("v2")
basicpolicy.test(10000)