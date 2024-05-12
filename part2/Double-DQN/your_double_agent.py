import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ExperienceBuffer, SequentialExperienceStorage
from model import Model
from utils import count_max_lives, check_if_live, process_frame, get_initialization_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EnvironmentAgent():
    def __init__(self, action_count):
        self.action_count = action_count
        self.discount_rate = 0.99
        self.eps = 1.0
        self.eps_min = 0.01
        self.exploration_steps = 500000
        self.eps_decay = (self.eps - self.eps_min) / self.exploration_steps
        self.training_start = 100000
        self.target_update = 1000

        self.sys_memory = ExperienceBuffer()
        self.policy_network = Model(action_count)
        self.policy_network.to(device)
        
        self.optimizer = optim.Adam(params=self.policy_network.parameters(), lr=learning_rate_dqn)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size_scheduler, gamma=gamma_scheduler)

        # Below, you should create a target net and init it the same as policy net
        
        # TODO
        self.target_network = Model(action_count)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.to(device)
        

    def load_policy_network(self, path):
        self.policy_network = torch.load(path)      

    # According to the interval above, target net is updated to be policy net
    def target_to_policy(self):
        ### CODE ###
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
        
    # Use eps-greedy policy
    def select_action(self, state): 
        if np.random.rand() <= self.eps:
            a = random.randrange(self.action_count)
        else:    
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_network(state)
                a = q_values.max(1)[1].item()
        return a

    
    def p_net_training(self, current_step):
        if self.eps > self.eps_min:
            self.eps -= self.eps_decay
        # Sampling from replay 
        mini_batch = self.sys_memory.get_mini_batch(current_step)
        mini_batch = np.array(mini_batch,dtype=object).transpose() # Convert to numpy array

        experience_sequence = np.stack(mini_batch[0], axis=0)

        current_states = np.float32(experience_sequence[:, :4, :, :]) / 255.
        current_states = torch.from_numpy(current_states).cuda()

        actions = list(mini_batch[1])
        action_tensors = torch.LongTensor(actions).cuda()

        rewards = list(mini_batch[2])
        reward_tensors = torch.FloatTensor(rewards).cuda()

        next_states = np.float32(experience_sequence[:, 1:, :, :]) / 255.
        next_states = torch.from_numpy(next_states).cuda()

        episode_completion_flags = mini_batch[3]

        # This is checking if game done
        done_mask = torch.tensor(list(map(int, episode_completion_flags==False)), dtype=torch.uint8).cuda()
        
        # Below, you should copy / paste the code from the your_agent.py file and adapt it to the double Q network.
        
        # Compute q val of the current state
        q_values = self.policy_network(current_states).gather(1, action_tensors.unsqueeze(1))

        # Now, compute for next state
        next_q_values = self.target_network(next_states).detach().max(1)[0]
        
        # Now that you did that, determine max val for action at next state (you'll use the policy network for this!)
        target_q_values = reward_tensors + self.discount_rate * next_q_values * done_mask
        
        # Compute the Huber Loss
        loss = F.smooth_l1_loss(q_values, target_q_values.unsqueeze(1))
        
        # Perform optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        if current_step % self.target_update == 0:
            self.target_to_policy()
