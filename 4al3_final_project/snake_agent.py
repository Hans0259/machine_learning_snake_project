import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


# ============================================================
#  GPU / CPU 自动选择
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ============================================================
#  CNN Q-Network
# ============================================================
class ConvQNet(nn.Module):
    def __init__(self, n_channels, n_actions, H, W):
        super().__init__()
        self.nA = n_actions

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # 自动计算 CNN 输出尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, H, W)
            out = self.cnn(dummy)
            self.flat_dim = out.numel()

        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    # 保存模型
    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print("✔ Model saved →", filename)

    # 载入模型
    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=DEVICE))
        print("✔ Loaded model →", filename)


# ============================================================
#  Replay Memory
# ============================================================
class ReplayMemory:
    def __init__(self, capacity=100_000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)


# ============================================================
#  Agent（使用 CNN + DQN）
# ============================================================
class Agent:
    def __init__(self, n_channels, n_actions, H, W):
        self.nA = n_actions
        self.nC = n_channels
        self.H = H
        self.W = W

        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 2000  # 越大探索越长

        self.n_game = 0

        self.model = ConvQNet(n_channels, n_actions, H, W).to(DEVICE)
        self.target = ConvQNet(n_channels, n_actions, H, W).to(DEVICE)
        self.target.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.memory = ReplayMemory()

    # --------------------------------------------------------
    #  Epsilon-Greedy
    # --------------------------------------------------------
    def get_action(self, state, explore_ratio=None):
        self.model.eval()

        # state shape: (4, H, W)
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # 随游戏数增加，epsilon 逐渐降低
        if explore_ratio is None:
            eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1.0 * self.n_game / self.epsilon_decay)
        else:
            eps = explore_ratio  # 渲染模式用

        if random.random() < eps:
            return random.randint(0, self.nA - 1)

        with torch.no_grad():
            q = self.model(state)
            return torch.argmax(q).item()

    # --------------------------------------------------------
    #  存记忆
    # --------------------------------------------------------
    def remember(self, s, a, r, s2, done):
        self.memory.push(s, a, r, s2, done)

    # --------------------------------------------------------
    #  训练 DQN
    # --------------------------------------------------------
    def train_long_memory(self, batch_size=256):
        if len(self.memory) < 1000:
            return

        batch = self.memory.sample(batch_size)
        s_list, a_list, r_list, s2_list, d_list = zip(*batch)

        s = torch.tensor(np.array(s_list), dtype=torch.float32, device=DEVICE)
        s2 = torch.tensor(np.array(s2_list), dtype=torch.float32, device=DEVICE)
        a = torch.tensor(a_list, dtype=torch.long, device=DEVICE)
        r = torch.tensor(r_list, dtype=torch.float32, device=DEVICE)
        d = torch.tensor(d_list, dtype=torch.float32, device=DEVICE)

        # Q(s,a)
        pred = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # target: r + γ * maxQ(s2)
        with torch.no_grad():
            next_q = self.target(s2).max(1)[0]
            target = r + (1 - d) * self.gamma * next_q

        loss = nn.MSELoss()(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # --------------------------------------------------------
    #  同步 target network
    # --------------------------------------------------------
    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
