
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from IPython.display import display

import torch

from Pygame.pixelcopter import Pixelcopter
from Pygame.ple import PLE

from DQN.Agent import Agent
from DQN.Preprocessor import Preprocessor
from makeanim import make_anim

from concurrent.futures import ThreadPoolExecutor


# --- 環境の準備 ---
FPS = 30 # 環境（ゲーム）のフレームレート
game = Pixelcopter() # ゲームPixelcopterを選択
env = PLE(game, fps=FPS, display_screen=True) # 環境準備

# 状態や行動の基本情報の取得
num_states = len(env.getGameState()) # 状態数を取得
actionset = env.getActionSet() # 行動の種類を取得
num_actions = len(actionset) # 行動数を取得
screen_dims = env.getScreenDims() # 画面サイズを取得
scr_width, scr_height = screen_dims # 画面の横幅と縦幅

prep = Preprocessor(scr_width, scr_height) # 状態の正規化用
agent = Agent(num_actions, num_states)  # エージェントの生成

# --- 学習開始 ---
NUM_EPISODES = 301 # 繰り返すエピソード数（100の倍数 + 1）
UPDATE_TARGET_STEP_PERIOD = 100 # Target Netの更新ステップ周期
# 表示用の設定（何エピソードごとに表示するか）
PRINT_EPISODE_PERIOD = 10 # ログを表示するエピソード周期
SHOWVIDEO_EPISODE_PERIOD = 300 # 動画とグラフを出力する周期

VIDEO_MAXDURATION_SEC = 60

env.init()

log_dur = [] # 10エピソードごとの継続時間の中央値を保存
last10dur = deque([], maxlen=10) # 直前10エピソードの継続時間
frames = [] # 途中のエピソードでの画像フレーム列を保存

step_count = 0 # target net更新用



def main():
    global frames
    global step_count
    for episode in range(NUM_EPISODES): # 指定エピソード数反復
        env.reset_game() # 環境のリセット

        # 可視化用: episodeの1桁目を上に丸める（89,90,91 -> 90,90,100）
        round_episode = int(np.ceil(episode / 10) * 10)
        if round_episode % SHOWVIDEO_EPISODE_PERIOD == 0:
            frames.append(env.getScreenRGB()) # 1フレーム目

        # 状態の観測と正規化
        state = prep.normalize(env.getGameState())
        # 状態の値部分（7次元）を1×7のTensorに変換
        state = torch.FloatTensor(list(state.values())).unsqueeze(0)

        t = 0
        while not env.game_over(): # エピソード内の時間ループ
            step_count += 1 # エピソードをまたぐ時間ステップカウンタ
            a_idx = agent.select_action(state, episode) # 行動決定
            action = actionset[a_idx]
            reward = env.act(action) # 行動を実行（報酬: +1, 0, -5）
            reward = torch.FloatTensor([reward]) # Tensorに変換

            # 次状態の取得（ゲームオーバーの場合はNoneにする）
            state_next = prep.normalize(env.getGameState()) # 次状態
            done = env.game_over() # ゲームオーバーか否かをチェック
            if done: # ゲームオーバーの場合
                state_next = None
            else: # ゲーム継続
                state_next = torch.FloatTensor(list(state_next.values())).unsqueeze(0) # 1×7次元へ

            # ReplayMemoryへ（状態, 行動, 次状態, 報酬）を保存
            agent.store_transition(state, a_idx, state_next, reward)
            # Main Netの更新
            agent.update_main_net()
            # Target Netの更新
            if step_count % UPDATE_TARGET_STEP_PERIOD == 0:
                agent.update_target_net()

            # 可視化用
            if round_episode % SHOWVIDEO_EPISODE_PERIOD == 0:
                frames.append(env.getScreenRGB())

            # 終了時の処理
            if done:
                last10dur.append(t) # 過去10エピソードの継続時間を保存
                break # 念のため
            else:
                state = state_next # 次状態を次の時刻の状態へ
                t += 1
        #     --- エピソード内ループ終了 ---

        # --- 途中経過の表示 ---
        if episode % PRINT_EPISODE_PERIOD == 0:
            print(f'Episode {episode:5d} duration= {t:4d} ', end='')
            print(f'(eps= {agent.epsilon:6.1e}) ', end='')
            print(f'(last 10 med(t): {np.median(last10dur):6.1f})')

        # エピソードの継続時間と10エピソード中央値を保存
        log_dur.append([t, np.median(last10dur)])
        # 継続時間とその平滑化（移動中央値）をプロット
        if episode % SHOWVIDEO_EPISODE_PERIOD == 0 and episode > 1:
            plt.plot(log_dur)
            plt.legend(['Duration', 'Median last10'])
            plt.xlabel('Episode')
            plt.ylabel('Duration')
            plt.show()
            # 100エピソードごとに最後の10エピソードを動画にする
            clip = make_anim(frames, fps=FPS)
            display(clip.ipython_display(fps=FPS,autoplay=1,loop=1, maxduration=VIDEO_MAXDURATION_SEC))
            frames = [] # 画像系列の履歴をクリア

if __name__ == "__main__":
    main()  
