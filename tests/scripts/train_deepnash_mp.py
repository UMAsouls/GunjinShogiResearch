import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import numpy as np
from tqdm import tqdm
import multiprocessing

# 必要なモジュールのインポート
import GunjinShogiCore as GSC
from src.const import BOARD_SHAPE, BOARD_SHAPE_INT, PIECE_LIMIT
from src.common import Player # win_countsで使う
from src.GunjinShogi import Environment, CppJudgeBoard, TensorBoard
from src.Agent.DeepNash import DeepNashAgent, DeepNashLearner, ReplayBuffer, Episode, Trajectory

# --- 設定 ---
# メインプロセス(学習)用デバイス
MAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ワーカープロセス(自己対戦)用デバイス。GPUメモリを節約するため "cpu" を推奨
WORKER_DEVICE_STR = "cpu" 

N_PROCESSES = 10          # 並列実行するプロセス数
TOTAL_CYCLES = 1000       # 総学習サイクル数 (総エピソード数 = N_PROCESSES * TOTAL_CYCLES)
BATCH_SIZE = 32           # 学習時のバッチサイズ
HISTORY_LEN = PIECE_LIMIT # TensorBoardの履歴数
MAX_STEPS = 1000          # 1ゲームの最大手数
BUF_SIZE = 1000           # ReplayBufferのサイズ (N_PROCESSES * 数サイクル分は最低限必要)

LOSS_DIR = "model_loss/deepnash_mp"
LOSS_NAME = "v2"

# --- ヘルパー関数 (自己対戦プロセス内で使用) ---

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
        policy, _, _ = agent.network(obs_tensor, non_legal_tensor)
        probs = policy
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        
    return action, probs.squeeze(0), non_legal_tensor.squeeze(0)

# --- 自己対戦ワーカー関数 ---

def run_self_play_episode(
    process_id: int,
    agent_state_dict: dict,
    in_channels: int,
    mid_channels: int,
    history_len: int,
    max_steps: int,
    device_str: str
):
    """
    1エピソード分の自己対戦を実行し、Episodeオブジェクトと勝者を返す
    """
    device = torch.device(device_str)

    # 1. Agent & Environment Initialization (プロセスごとに独立して生成)
    agent = DeepNashAgent(in_channels, mid_channels, device)
    agent.load_state_dict(agent_state_dict)

    cppJudge = GSC.MakeJudgeBoard("config.json")
    judge = CppJudgeBoard(cppJudge)
    tensorboard = TensorBoard(BOARD_SHAPE, device, history=history_len)
    env = Environment(judge, tensorboard)

    # 2. Self-play Episode
    env.reset()
    
    sample_obs = env.get_tensor_board_current()
    current_episode = Episode(device, sample_obs.shape, max_step=max_steps)
    temp_trajectories = []
    
    done = False
    step_count = 0
    
    while not done and step_count < max_steps:
        current_player = env.get_current_player()
        
        action, policy, non_legal = get_agent_output(agent, env, device)
        
        obs = env.get_tensor_board_current().clone()
        
        if action == -1:
            _, _, _ = env.step(-1)
            done = True
        else:
            _, _, frag = env.step(action)
            
            if frag != GSC.BattleEndFrag.CONTINUE and frag != GSC.BattleEndFrag.DEPLOY_END:
                done = True
            
            trac = Trajectory(
                board=obs, # CPUに送るのはadd_step内で行われる
                action=action,
                reward=0.0, # 後で設定
                policy=policy.detach(),
                player=current_player,
                non_legal=non_legal.detach()
            )
            temp_trajectories.append(trac)
            
        step_count += 1
        
    # 3. Reward Calculation
    winner = env.get_winner()
    final_reward = 0.5
    if winner == GSC.Player.PLAYER_ONE:
        final_reward = 1.0
    elif winner == GSC.Player.PLAYER_TWO:
        final_reward = 0.0 # Player1視点

    for trac in temp_trajectories:
        r = final_reward if trac.player == GSC.Player.PLAYER_ONE else 1 - final_reward
        trac.reward = r
        current_episode.add_step(trac)
    
    current_episode.episode_end()
    
    return current_episode, winner

# --- メイン学習プロセス ---

def main():
    print(f"Main Device: {MAIN_DEVICE}")
    print(f"Worker Device: {WORKER_DEVICE_STR}")
    print(f"Num Processes: {N_PROCESSES}")

    # 1. Agent, Learner, Buffer Initialization
    in_channels = 18 + HISTORY_LEN
    mid_channels = 20
    
    agent = DeepNashAgent(in_channels, mid_channels, MAIN_DEVICE)
    learner = DeepNashLearner(in_channels, mid_channels, MAIN_DEVICE)
    replay_buffer = ReplayBuffer(size=BUF_SIZE)
    
    win_counts = {Player.PLAYER1: 0, Player.PLAYER2: 0, "DRAW": 0}
    
    total_episodes = 0

    for i in tqdm(range(TOTAL_CYCLES), desc="Training Cycles"):
        # 最新のモデルパラメータをCPUにコピーしてワーカーに渡す
        current_state_dict = agent.network.state_dict()
        cpu_state_dict = {k: v.cpu() for k, v in current_state_dict.items()}

        # 自己対戦を並列実行するための引数リストを作成
        args_list = [
            (
                pid,
                cpu_state_dict,
                in_channels,
                mid_channels,
                HISTORY_LEN,
                MAX_STEPS,
                WORKER_DEVICE_STR
            ) for pid in range(N_PROCESSES)
        ]

        # 2. Self-play Generation (in parallel)
        with multiprocessing.Pool(processes=N_PROCESSES) as pool:
            results = pool.starmap(run_self_play_episode, args_list)

        # 3. Collect results and fill replay buffer
        for episode, winner in results:
            if len(episode.boards) > 0: # 空のエピソードは追加しない
                replay_buffer.add(episode)
            
            if winner == GSC.Player.PLAYER_ONE:
                win_counts[Player.PLAYER1] += 1
            elif winner == GSC.Player.PLAYER_TWO:
                win_counts[Player.PLAYER2] += 1
            else:
                win_counts["DRAW"] += 1
        
        total_episodes += N_PROCESSES

        # 4. Learning Step
        if len(replay_buffer) >= BATCH_SIZE:
            loss_path = f"{LOSS_DIR}/{LOSS_NAME}"
            os.makedirs(loss_path, exist_ok=True)
            learner.learn(replay_buffer, BATCH_SIZE, loss_path)
            
            # 学習後のモデルパラメータをagentに反映
            agent.load_state_dict(learner.get_current_network_state_dict())
        
        # 5. Logging and Saving
        # 約100エピソードごとにログ出力
        if (i + 1) % (100 // N_PROCESSES or 1) == 0:
            p1_wins = win_counts[Player.PLAYER1]
            p2_wins = win_counts[Player.PLAYER2]
            draws = win_counts["DRAW"]
            print(f"\nTotal Episodes: {total_episodes}: P1 Wins: {p1_wins}, P2 Wins: {p2_wins}, Draws: {draws}")
            
            # モデルの保存
            save_path = f"logs/deepnash_mp_model_{total_episodes}.pth"
            os.makedirs("logs", exist_ok=True)
            torch.save(agent.network.state_dict(), save_path)
            print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # 'spawn' を使うことで、CUDA利用時のfork関連のエラーを回避します
    multiprocessing.set_start_method('spawn', force=True)
    main()