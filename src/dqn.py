import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import TensorDataset, DataLoader
from base_learning import BaseLearning
import matplotlib.pyplot as plt
from env_v1 import BlackjackEnv
from env_v3 import BlackjackDoubleDownSplitEnv
from tqdm import tqdm

# Creation du modèle de DL
class DQNSolver(nn.Module):
    """
    Convolutional Neural Net with 3 conv layers and two linear layers
    """
    def __init__(self, input_shape, n_actions):
        super(DQNSolver, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 30),
            nn.ReLU(),
            nn.Linear(30, n_actions)
        )
    
    def forward(self, x):
        #x = self.fc(x)#.view(x.size()[0], -1)
        return self.fc(x)

class DQNAgent:

    def __init__(self, env, max_memory_size, batch_size, gamma, lr, epsilon):
        self.plot = {"n": [], "reward": [], "win":[], "draw":[], "loss":[]}

        # Define DQN Layers
        self.env = env
        self.state_space = len(env.observation_space)
        self.action_space = env.action_space.n
        #self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # DQN network  
        self.dqn = DQNSolver(self.state_space, self.action_space).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # Create memory
        self.max_memory_size = max_memory_size
        self.STATE_MEM = torch.zeros(max_memory_size, self.state_space)
        self.ACTION_MEM = torch.zeros(max_memory_size, 1)
        self.REWARD_MEM = torch.zeros(max_memory_size, 1)
        self.STATE2_MEM = torch.zeros(max_memory_size, self.state_space)
        self.DONE_MEM = torch.zeros(max_memory_size, 1)
        self.ending_position = 0
        self.num_in_queue = 0
        
        self.memory_sample_size = batch_size
        
        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device) 
        self.epsilon = epsilon

    def remember(self, state, action, reward, state2, done):
        """Store the experiences in a buffer to use later"""
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)
    
    def batch_experiences(self):
        """Randomly sample 'batch size' experiences"""
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]      
        return STATE, ACTION, REWARD, STATE2, DONE
    
    def act(self, state, train):
        """Epsilon-greedy action"""
        if (random.random() < self.epsilon) and train: 
            return torch.tensor([[self.env.action_space.sample()]])
        else:
            return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()
    
    def experience_replay(self):
        if self.memory_sample_size > self.num_in_queue:
            return
    
        # Sample a batch of experiences
        STATE, ACTION, REWARD, STATE2, DONE = self.batch_experiences()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)
        
        self.optimizer.zero_grad()
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a) 
        target = REWARD + torch.mul((self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)
        current = self.dqn(STATE).gather(1, ACTION.long())
        
        loss = self.l1(current, target)
        loss.backward() # Compute gradients
        self.optimizer.step() # Backpropagate error

        #self.exploration_rate *= self.exploration_decay

    def train(self, n_episode, plot=False, plot_interval=10000, plot_test_size=20000):
        win = 0
        draw = 0
        loss = 0
        for i in range(n_episode):
            # print(i)
            if i % plot_interval == 0:
                print("training, episode {} out of {}".format(i, n_episode))
                if plot:
                    print("Gather ploting information")
                    win, draw, loss, reward = self.test(plot_test_size)
                    self.plot["n"].append(i)
                    self.plot["win"].append(win)
                    self.plot["draw"].append(draw)
                    self.plot["loss"].append(loss)
                    self.plot["reward"].append(reward)


            done = False
            state = list(self.env.reset())
            state = torch.Tensor([state])
            total_reward = 0
            while not done:
                action = self.act(state, True)
                
                state_next, reward, done, info = self.env.step(int(action[0]))
                total_reward += reward
                state_next = torch.Tensor([state_next])
                reward = torch.tensor([reward]).unsqueeze(0)
                
                self.remember(state, action, reward, state_next, torch.tensor([int(done)]).unsqueeze(0))
                self.experience_replay()
                
                state = state_next

    def test(self, n_episode):
        win, draw, loss = 0, 0, 0
        total_reward = 0
        total_hand = n_episode

        for i in tqdm(range(n_episode)):
            done = False
            observation = list(self.env.reset())
            observation = torch.Tensor([observation])
            while not done:
                action = self.act(observation, True)
                
                observation_next, reward, done, info = self.env.step(int(action[0]))
                observation_next = torch.Tensor([observation_next])
                observation = observation_next

            if len(info) == 0:
                total_reward += reward

                if reward > 0:
                    win += 1
                elif reward < 0:
                    loss += 1
                else:
                    draw += 1
            else:
                reward0, reward1 = info["final_reward0"], info["final_reward1"]
                if reward0 is not None:
                    total_reward += reward0
                    if reward0 > 0:
                        win += 1
                    elif reward0 < 0:
                        loss += 1
                    else:
                        draw += 1

                if reward1 is not None:
                    total_hand += 1
                    total_reward += reward1
                    if reward1 > 0:
                        win += 1
                    elif reward1 < 0:
                        loss += 1
                    else:
                        draw += 1

        assert win + draw + loss == total_hand, "Error in win, draw, loss count : {} + {} + {} = {} != {}".format(win, draw, loss, win + draw + loss, total_hand)

        print("win: {} | draw: {} | loss: {}".format(win/ total_hand, draw / total_hand, loss / total_hand))
        print("mean reward: {}".format(total_reward / total_hand))

        return win / total_hand, draw / total_hand, loss / total_hand,  total_reward / total_hand


if __name__ == "__main__":
    for env in ["v1", "v2", "v3"]:
        print("##### {} #####".format(env))
        dqn = DQNAgent(
            BlackjackEnv(),
            max_memory_size=30000,
            batch_size=128,
            gamma=0.2,
            lr=0.0025,
            epsilon=0.01
        )
        dqn.train(10000)
        dqn.test(20000)
