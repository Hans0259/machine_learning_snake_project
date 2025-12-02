import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# QNet
class ConvQNet(nn.Module):
    def __init__(self, in_channels, n_actions, height, width):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/2, W/2
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # H/4, W/4
        )

        # layer size 256
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            out = self.conv(dummy)
            flat_size = out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Agent:
    def __init__(self, nC, nA, H, W, memory_size=100_000):
        self.nA = nA
        self.n_game = 0

        # DQN parameter
        self.gamma = 0.92
        self.lr = 1e-4

        # epsilon-greedy
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 500  # converge after 500 games

        # check device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.memory = deque(maxlen=memory_size)

        # QNet and target initialization
        self.model = ConvQNet(nC, nA, H, W).to(self.device)
        self.target_model = ConvQNet(nC, nA, H, W).to(self.device)
        self.update_target() 

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # eps decreases linearly with number of games:
    # eps = eps_start - n_game / eps_decay
    def _epsilon(self):
        eps = self.eps_start - self.n_game / self.eps_decay
        return max(self.eps_end, eps)

    def get_action(self, state, explore=True):
        # Get epsilon value
        eps = self._epsilon() if explore else 0.0
        # Exploration step
        if explore and random.random() < eps:
            return random.randint(0, self.nA - 1)
        # Exploitation: choose best Q-value
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t)
        action = torch.argmax(q_values, dim=1).item()
        return action
    
    def train_long_memory(self, batch_size=128):
        # Not enough samples to train
        if len(self.memory) < batch_size:
            return
        # Sample a random minibatch
        mini_batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        # Convert to tensors
        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_pred = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Target Q-values using target network
        with torch.no_grad():
            q_next = self.target_model(next_states).max(1)[0]

        # Bellman target
        q_target = rewards + self.gamma * q_next * (1 - dones)
        # Compute loss and gradient update
        loss = self.criterion(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path="model.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target()
