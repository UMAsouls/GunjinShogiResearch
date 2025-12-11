import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm # 進捗バー表示用 (pip install tqdm 推奨)

# 必要なモジュールのインポート
import GunjinShogiCore as GSC
from src.const import BOARD_SHAPE, BOARD_SHAPE_INT, PIECE_LIMIT
from src.common import make_ndarray_board, Player
from src.GunjinShogi import Environment, CppJudgeBoard, TensorBoard
from src.Agent import RandomAgent
from src.Agent.IS_MCTS import ReplayBuffer, IsMctsLearner


# 設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPISODES = 100000        # 総対戦数
LEARN_INTERVAL = 100      # 何エピソードごとに学習するか
BATCH_SIZE = 2000         # 学習時のバッチサイズ
HISTORY_LEN = PIECE_LIMIT # TensorBoardの履歴数
MAX_STEPS = 1000          # 1ゲームの最大手数
BUF_SIZE = 100000

LR = 0.01

LOSS_DIR = "model_loss/is_mcts"
MODEL_DIR = "models/is_mcts"
NAME = "v2"

def main():
    print(f"Device: {DEVICE}")
    
    # 1. Agent & Buffer Initialization
    # 入力チャンネル数: 自分の駒(16) + 敵駒(1) + mode(1) + 履歴(HISTORY_LEN)
    in_channels = 18 + HISTORY_LEN 
    mid_channels = 20 # 任意
    
    agent = RandomAgent()
    learner = IsMctsLearner(DEVICE,in_channels, mid_channels, lr = LR)
    replay_buffer = ReplayBuffer(size=BUF_SIZE) # メモリに合わせて調整
    
    cppJudge = GSC.MakeJudgeBoard("config.json")
    judge = CppJudgeBoard(cppJudge)
        
    tensorboard = TensorBoard(BOARD_SHAPE, DEVICE, history=HISTORY_LEN)
        
    env = Environment(judge, tensorboard)
    
    # 勝率記録用
    win_counts = {Player.PLAYER1: 0, Player.PLAYER2: 0}
    
    for i in tqdm(range(N_EPISODES)):
        env.reset()
        # 2. Environment Reset (Every Game)
        # 毎回ランダムな配置で初期化
        
        # CppJudgeBoardは配置に依存するため毎回生成
        
        # 内部用ボードのセットアップ
        #board1 = make_ndarray_board(pieces1)
        #board2 = make_ndarray_board(pieces2)
        #env.set_board(board1, board2) # TensorBoardもここで初期化される
        
        # エピソードデータの準備
        # TensorBoardのシェイプを取得
        temp_trajectories = [] # (player, trajectory) のタプルを一時保存
        
        done = False
        step_count = 0
        
        while not done and step_count < MAX_STEPS:
            
            # 行動決定
            action = agent.get_action(env)
            
            # 観測データ(s_t)の保存（行動前の状態）
            obs = env.get_tensor_board_current().clone()
            
            if action == -1: # 投了または合法手なし
                # 便宜上、環境を終わらせる処理
                _, _, _ = env.step(-1)
                done = True
                winner = env.get_winner()
            else:
                # 環境更新
                _, log, frag = env.step(action)
                
                if frag != GSC.BattleEndFrag.CONTINUE and frag != GSC.BattleEndFrag.DEPLOY_END:
                    done = True
                    winner = env.get_winner()
                
                # Trajectoryの一時保存
                # rewardは後で埋めるので一旦0
                temp_trajectories.append(obs)
                step_count += 1
            
            
        # 3. Reward Calculation (Outcome)
        # ゲーム終了後、勝者に+1、敗者に0、引き分け0.5 を伝播させる
        final_reward = 0.5
        if env.get_winner() == GSC.Player.PLAYER_ONE:
            final_reward = 1.0
            win_counts[Player.PLAYER1] += 1
        elif env.get_winner() == GSC.Player.PLAYER_TWO:
            final_reward = 0 # Player1視点では-1
            win_counts[Player.PLAYER2] += 1
            
        # エピソードにデータを格納
        for s in range(step_count):
            r = final_reward if(s%2 == 0) else 1 - final_reward
            replay_buffer.add(temp_trajectories[s], r)
        
        # 4. Learning Step
        if (i + 1) % LEARN_INTERVAL == 0:
            os.makedirs(f"{LOSS_DIR}/{NAME}", exist_ok=True)
            learner.learn(replay_buffer, BATCH_SIZE, f"{LOSS_DIR}/{NAME}")
            
            # 定期的にログ出力
            if (i + 1) % 100 == 0:
                p1_wins = win_counts[Player.PLAYER1]
                p2_wins = win_counts[Player.PLAYER2]
                print(f"\nEpisode {i+1}: P1 Wins: {p1_wins}, P2 Wins: {p2_wins}")
                # モデルの保存
                os.makedirs(f"{MODEL_DIR}/{NAME}", exist_ok=True)
                torch.save(learner.network.state_dict(), f"{MODEL_DIR}/{NAME}/model_{i+1}.pth")

if __name__ == "__main__":
    main()