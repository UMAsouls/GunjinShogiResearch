import os
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm # 進捗バー表示用 (pip install tqdm 推奨)

# 必要なモジュールのインポート
import GunjinShogiCore as GSC
from src.const import BOARD_SHAPE, BOARD_SHAPE_INT, PIECE_LIMIT
from src.common import make_ndarray_board, Player
from src.GunjinShogi import Environment, CppJudgeBoard, TensorBoard
from src.Agent.DeepNash import DeepNashAgent, DeepNashLearner, ReplayBuffer, Episode, Trajectory


# 設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPISODES = 100000        # 総対戦数
LEARN_INTERVAL = 10       # 何エピソードごとに学習するか
BATCH_SIZE = 36           # 学習時のバッチサイズ
FIXED_GAME_SIZE = 200
HISTORY_LEN = PIECE_LIMIT # TensorBoardの履歴数
MAX_STEPS = 1000          # 1ゲームの最大手数
BUF_SIZE = 100

MID_CHANNELS = 40

LEARNING_RATE = 0.00005

LOSS_DIR = "model_loss/deepnash"
MODEL_DIR = "models/deepnash"
NAME = "v5"

def get_agent_output(agent: DeepNashAgent, env: Environment, device: torch.device):
    """
    Agentからアクションだけでなく、学習に必要なPolicyなども取得するヘルパー関数
    (DeepNashAgent.get_action を拡張したような処理)
    """
    agent.network.eval()
    
    # 現在の手番の盤面取得
    obs_tensor = env.get_tensor_board_current().unsqueeze(0) # (1, C, H, W)
    
    # 合法手マスク作成
    legals = env.legal_move()
    if len(legals) == 0:
        return -1, None, None # 投了
        
    non_legal_mask = np.ones((BOARD_SHAPE_INT**2), dtype=bool)
    non_legal_mask[legals] = False
    non_legal_tensor = torch.from_numpy(non_legal_mask).unsqueeze(0) # (1, ActionSize)
    
    with torch.no_grad():
        policy, _, logit = agent.network(obs_tensor, non_legal_tensor)
        probs = policy
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        
    return action, probs.squeeze(0), non_legal_tensor.squeeze(0)

def main():
    print(f"Device: {DEVICE}")
    
    cppJudge = GSC.MakeJudgeBoard("config.json")
    judge = CppJudgeBoard(cppJudge)
        
    tensorboard = TensorBoard(BOARD_SHAPE, torch.device("cpu"), history=HISTORY_LEN)
    
    env = Environment(judge, tensorboard)
    
    # 1. Agent & Buffer Initialization
    # 入力チャンネル数: 自分の駒(16) + 敵駒(1) + mode(1) + 履歴(HISTORY_LEN)
    in_channels = tensorboard.total_channels 
    mid_channels = MID_CHANNELS # 任意
    
    agent = DeepNashAgent(in_channels, mid_channels, torch.device("cpu"))
    leaner = DeepNashLearner(
        in_channels, mid_channels, DEVICE, lr=LEARNING_RATE
    )
    replay_buffer = ReplayBuffer(size=BUF_SIZE, board_shape=[in_channels, BOARD_SHAPE[0], BOARD_SHAPE[1]]) # メモリに合わせて調整
    
    # 勝率記録用
    win_counts = {Player.PLAYER1: 0, Player.PLAYER2: 0}
    
    for i in tqdm(range(N_EPISODES)):
        env.reset()
        # 2. Environment Reset (Every Game)
        # 毎回ランダムな配置で初期化
        pieces1 = agent.get_first_board()
        pieces2 = agent.get_first_board()
        
        # CppJudgeBoardは配置に依存するため毎回生成
        
        # 内部用ボードのセットアップ
        #board1 = make_ndarray_board(pieces1)
        #board2 = make_ndarray_board(pieces2)
        #env.set_board(board1, board2) # TensorBoardもここで初期化される
        
        # エピソードデータの準備
        # TensorBoardのシェイプを取得
        sample_obs = env.get_tensor_board_current()
        current_episode = Episode(sample_obs.shape, max_step=MAX_STEPS)
        
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
                
                if frag != GSC.BattleEndFrag.CONTINUE and frag != GSC.BattleEndFrag.DEPLOY_END:
                    done = True
                    winner = env.get_winner()
                
                # Trajectoryの一時保存
                # rewardは後で埋めるので一旦0
                trac = Trajectory(
                    board=obs.cpu(),
                    action=action,
                    reward=torch.zeros(2, dtype=torch.float32),
                    policy=policy.detach().cpu(),
                    player=current_player,
                    non_legal=non_legal.detach().cpu()
                )
                current_episode.add_step(trac)
                
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
        current_episode.set_reward(final_reward)
        current_episode.episode_end()
        replay_buffer.add(current_episode)
        
        # 4. Learning Step
        if (i + 1) % LEARN_INTERVAL == 0:
            leaner.learn(replay_buffer, BATCH_SIZE, FIXED_GAME_SIZE, f"{LOSS_DIR}/{NAME}")
            
            state_dict = leaner.get_current_network_state_dict()
            cpu_weights = {}
            for k, v in state_dict.items():
                # "_orig_mod." がついていたら削除する
                new_key = k.replace("_orig_mod.", "")
                cpu_weights[new_key] = v.cpu()
            
            agent.load_state_dict(cpu_weights)
            
            # 定期的にログ出力
            if (i + 1) % 100 == 0:
                p1_wins = win_counts[Player.PLAYER1]
                p2_wins = win_counts[Player.PLAYER2]
                print(f"\nEpisode {i+1}: P1 Wins: {p1_wins}, P2 Wins: {p2_wins}")
                # モデルの保存
                os.makedirs(f"{MODEL_DIR}/{NAME}", exist_ok=True)
                torch.save(agent.network.state_dict(), f"{MODEL_DIR}/{NAME}/model_{i+1}.pth")

if __name__ == "__main__":
    main()