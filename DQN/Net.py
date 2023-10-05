import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Q-Network （ニューラルネット）"""
    def __init__(self, n_states, n_mid, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_mid)
        self.fc4 = nn.Linear(n_mid, n_mid)
        self.fc5 = nn.Linear(n_mid, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x) # 出力層は活性化関数なし