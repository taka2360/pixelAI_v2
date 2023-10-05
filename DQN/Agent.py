from .Net import Net
from .ReplayMemory import ReplayMemory, Transition

import random
import copy
import math
import numpy as np

import torch
import torch.nn as nn


RM_CAPACITY = 10000 # ReplayMemoryのサイズ
NUM_MIDDLE_LAYER = 48 # 中間層のユニット数
LR = 5e-4 # 学習率
BATCH_SIZE = 64 # バッチサイズ
GAMMA = 0.99 # 割引率
EPS_INI = 1.0 # epsilonの初期値
EPS_FIN = 1e-3 # epsilonの収束先
EPS_DIV = 30 # 大きいほどゆっくり減少

device = torch.device("cuda:0")
print(device)

class Agent:
    """エージェントクラス"""
    def __init__(self, num_actions, num_states=0):
        assert num_actions == 2
        self.num_actions = num_actions
        self.num_states = num_states

        # main/targetニューラルネットの構築
        self.main_net = Net(self.num_states, NUM_MIDDLE_LAYER, self.num_actions).to(device) 
        self.target_net = copy.deepcopy(self.main_net)
        print(self.main_net) # 構造を表示
        # ニューラルネット学習（パラメータ最適化）の設定
        self.optimizer = torch.optim.AdamW(self.main_net.parameters(), lr=LR, amsgrad=True)
        # リプレイメモリー
        self.memory = ReplayMemory(RM_CAPACITY)
        self.epsilon = EPS_INI

    def store_transition(self, state, action, s_next, reward):
        self.memory.push(state, action, s_next, reward)

    def select_action(self, state, episode):
        # epsilon-greedyのepsilonをepisodeと共に減らしていく
        self.epsilon = EPS_FIN + (EPS_INI - EPS_FIN) * \
            math.exp(-1. * episode / EPS_DIV)

        if np.random.uniform(0, 1) > self.epsilon: # 学習した方策
            self.main_net.eval()
            with torch.no_grad():
                a_idx = self.main_net(state).max(1)[1].view(1, 1)
        else: # ランダムに選択
            a_idx = torch.LongTensor(
            [[random.randrange(self.num_actions)]])

        return a_idx

    def update_main_net(self):
        if len(self.memory) < BATCH_SIZE:
            return # BATCH_SIZE以上になるまでは学習しない

        # メモリーからの取り出し
        transitions = self.memory.sample(BATCH_SIZE)
        # BATCH_SIZE個のTransition
        # -> 1つのTransitionで各要素（stateなど）がBATCH_SIZE個
        batch = Transition(*zip(*transitions))
        # 各要素（BATCH_SIZE個）をそれぞれPyTorchのTensorにする
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # next_stateがNone （game over）ならFalse、それ以外はTrue
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
        # Noneではない状態を取り出す
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # 実際にとった行動a_tに対応するQ値 Q(s_t, a_t)を求める
        state_action_values = self.main_net(state_batch).gather(1, action_batch)

        # max_a Q(s_{t+1}, a)を求める
        # ただしnext_stateがNoneならば0とする
        next_state_values = torch.zeros(BATCH_SIZE)
        with torch.no_grad(): # 推論のみなので
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = reward_batch + next_state_values * GAMMA

        # Qネットワークの更新
        criterion = nn.MSELoss() # 最小平均二乗誤差損失
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad() # 勾配クリア
        loss.backward() # 誤差逆伝播による勾配計算
        self.optimizer.step() # ニューラルネットのパラメータ更新

    def update_target_net(self):
        """Main NetをTarget Netへコピー"""
        self.target_net.load_state_dict(self.main_net.state_dict())
