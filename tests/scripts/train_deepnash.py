import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm # 進捗バー表示用 (pip install tqdm 推奨)

# 必要なモジュールのインポート
import GunjinShogiCore as GSC
from src.const import BOARD_SHAPE, BOARD_SHAPE_INT, PIECE_LIMIT
from src.common import make_ndarray_board, Player
from src.GunjinShogi import Environment, CppJudgeBoard, TensorBoard
from src.Agent.DeepNash.agent import DeepNashAgent
from src.Agent.DeepNash.replay_buffer import ReplayBuffer, Episode, Trajectory

# 設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPISODES = 10000        # 総対戦数
LEARN_INTERVAL = 10       # 何エピソードごとに学習するか
BATCH_SIZE = 32           # 学習時のバッチサイズ
HISTORY_LEN = 30          # TensorBoardの履歴数
MAX_STEPS = 2000          # 1ゲームの最大手数

def get_agent_output(agent: DeepNashAgent, env: Environment, device: torch.device):
    """
    Agentからアクションだけでなく、学習に必要なPolicyなども取得するヘルパー関数
    (DeepNashAgent.get_action を拡張したような処理)
    """
    agent.network.eval()
    
    # 現在の手番の盤面取得
    obs_tensor = env.get_tensor_board_current().unsqueeze(0).to(device) # (1, C, H, W)
    
    # 合法手マスク作成
    legals = env.legal_move()
    if len(legals) == 0:
        return -1, None, None # 投了
        
    non_legal_mask = np.ones((BOARD_SHAPE_INT**2), dtype=bool)
    non_legal_mask[legals] = False
    non_legal_tensor = torch.from_numpy(non_legal_mask).to(device).unsqueeze(0) # (1, ActionSize)
    
    with torch.no_grad():
        policy_logits, _ = agent.network(obs_tensor, non_legal_tensor)
        probs = F.softmax(policy_logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        
    return action, probs.squeeze(0), non_legal_tensor.squeeze(0)

def main():
    print(f"Device: {DEVICE}")
    
    # 1. Agent & Buffer Initialization
    # 入力チャンネル数: 自分の駒(16) + 敵駒(1) + 履歴(HISTORY_LEN)
    in_channels = 17 + HISTORY_LEN 
    mid_channels = 64 # 任意
    
    agent = DeepNashAgent(in_channels, mid_channels, DEVICE)
    replay_buffer = ReplayBuffer(size=5000) # メモリに合わせて調整
    
    cppJudge = GSC.MakeJudgeBoard("config.json")
    judge = CppJudgeBoard(cppJudge)
        
    tensorboard = TensorBoard(BOARD_SHAPE, DEVICE, history=HISTORY_LEN)
        
    env = Environment(judge, tensorboard)
    
    # 勝率記録用
    win_counts = {Player.PLAYER1: 0, Player.PLAYER2: 0}
    
    for i in tqdm(range(N_EPISODES)):
        # 2. Environment Reset (Every Game)
        # 毎回ランダムな配置で初期化
        pieces1 = agent.get_first_board()
        pieces2 = agent.get_first_board()
        
        # CppJudgeBoardは配置に依存するため毎回生成
        
        # 内部用ボードのセットアップ
        board1 = make_ndarray_board(pieces1)
        board2 = make_ndarray_board(pieces2)
        env.set_board(board1, board2) # TensorBoardもここで初期化される
        
        # エピソードデータの準備
        # TensorBoardのシェイプを取得
        sample_obs = env.get_tensor_board_current()
        current_episode = Episode(DEVICE, sample_obs.shape, max_step=MAX_STEPS)
        
        temp_trajectories = [] # (player, trajectory) のタプルを一時保存
        
        done = False
        step_count = 0
        
        while not done and step_count < MAX_STEPS:
            current_player = env.get_current_player() # GSC.Player
            
            # 行動決定
            action, policy, non_legal = get_agent_output(agent, env, DEVICE)
            
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
                
                if frag != GSC.BattleEndFrag.CONTINUE:
                    done = True
                    winner = env.get_winner()
                
                # Trajectoryの一時保存
                # rewardは後で埋めるので一旦0
                trac = Trajectory(
                    board=obs,
                    action=action,
                    reward=0.0,
                    policy=policy,
                    player=current_player,
                    non_legal=non_legal
                )
                temp_trajectories.append(trac)
                
            step_count += 1
            
        # 3. Reward Calculation (Outcome)
        # ゲーム終了後、勝者に+1、敗者に-1、引き分け0 を伝播させる
        final_reward = 0.0
        if env.get_winner() == GSC.Player.PLAYER_ONE:
            final_reward = 1.0
            win_counts[Player.PLAYER1] += 1
        elif env.get_winner() == GSC.Player.PLAYER_TWO:
            final_reward = -1.0 # Player1視点では-1
            win_counts[Player.PLAYER2] += 1
            
        # エピソードにデータを格納
        for trac in temp_trajectories:
            # Player1ならそのまま、Player2なら報酬を反転
            r = final_reward if trac.player == GSC.Player.PLAYER_ONE else -final_reward
            trac.reward = r
            current_episode.add_step(trac)
            
        replay_buffer.add(current_episode)
        
        # 4. Learning Step
        if (i + 1) % LEARN_INTERVAL == 0:
            agent.learn(replay_buffer, BATCH_SIZE)
            
            # 定期的にログ出力
            if (i + 1) % 100 == 0:
                p1_wins = win_counts[Player.PLAYER1]
                p2_wins = win_counts[Player.PLAYER2]
                print(f"\nEpisode {i+1}: P1 Wins: {p1_wins}, P2 Wins: {p2_wins}")
                # モデルの保存
                torch.save(agent.network.state_dict(), f"logs/deepnash_model_{i+1}.pth")

if __name__ == "__main__":
    main()