import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        #2 layer MLP
        # input to hidden
        self.linear1 = nn.Linear(input_size, hidden_size)
        # hidden to output
        self.linear2 = nn.Linear(hidden_size, output_size)

    #feed forward: relu to linear
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

#save model and load the previous trained result
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))

#DQN
class QTrainer:
    def __init__(self, lr, gamma, input_dim, hidden_dim, output_dim):
        # Discount factor
        self.gamma = gamma
        # Hidden layer size
        self.hidden_size = hidden_dim
        # This online network is updated every training step.
        self.model = Linear_QNet(input_dim, self.hidden_size, output_dim)
        # Target Q-network
        self.target_model = Linear_QNet(input_dim, self.hidden_size, output_dim)
        # Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Loss function
        self.criterion = nn.MSELoss()
        self.copy_model()

    def copy_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def train_step(self, state, action, reward, next_state, done):
        # convert to torch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long).view(-1, 1)
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1)
        done = torch.tensor(done, dtype=torch.float).view(-1, 1)

        # ensure 2D input to model
        if state.ndim == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)

        # current Q values
        q_values = self.model(state).gather(1, action)

        # target Q values
        with torch.no_grad():
            q_next = self.target_model(next_state).max(1)[0].unsqueeze(1)
            q_target = reward + self.gamma * q_next * (1 - done)

        # Compute loss
        loss = self.criterion(q_values, q_target)
        # Gradient update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Agent:
    def __init__(self, nS, nA, max_explore=1000, gamma=0.95,
                 max_memory=5000, lr=0.0001, hidden_dim=128):
        
        # Maximum exploration steps before epsilon reaches 0, use first 1000 game to learn
        self.max_explore = max_explore
        self.memory = deque(maxlen=max_memory)
        # State and action dimensions
        self.nS = nS
        self.nA = nA
        #number of game
        self.n_game = 0
        # Create DQN trainer
        self.trainer = QTrainer(lr, gamma, self.nS, hidden_dim, self.nA)
    
    #  Store a transition into replay memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #  Long-term memory training
    def train_long_memory(self, batch_size):
        if len(self.memory) > batch_size:
            mini_sample = random.sample(self.memory, batch_size)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states = np.array(states)
        next_states = np.array(next_states)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    #  Short-term training
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state, n_game, explore=True):
        # Convert state to tensor for model input
        state = torch.tensor(state, dtype=torch.float)
        prediction = self.trainer.model(state).detach().numpy().squeeze()
        # Linear epsilon decay
        epsilon = max(0, self.max_explore - n_game)

        #Exploration
        if explore and random.randint(0, self.max_explore) < epsilon:
            prob = np.exp(prediction) / np.exp(prediction).sum()
            final_move = np.random.choice(len(prob), p=prob)
        #Exploitation
        else:
            final_move = prediction.argmax()

        return final_move