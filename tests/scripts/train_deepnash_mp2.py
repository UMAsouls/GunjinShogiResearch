import os

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# 必要なモジュールのインポート
import GunjinShogiCore as GSC
from src.common import Player, get_action, make_action, Config # win_countsで使う
from src.GunjinShogi import Environment, CppJudgeBoard, JUDGE_TABLE
from src.Agent.DeepNash import DeepNashAgent, DeepNashCnnAgent, DeepNashLearner, DeepNashCnnLearner, \
    ReplayBuffer, Episode, Trajectory, TensorBoard, SimpleTensorBoard

# --- 設定 ---
# メインプロセス(学習)用デバイス
MAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ワーカープロセス(自己対戦)用デバイス。GPUメモリを節約するため "cpu" を推奨
WORKER_DEVICE_STR = "cpu" 

N_PROCESSES = 8          # 並列実行するプロセス数
TOTAL_CYCLES = 100000       # 総学習サイクル数 (総エピソード数 = N_PROCESSES * TOTAL_CYCLES)
BATCH_SIZE = 320           # 学習時のバッチサイズ
ACCUMRATION = 2
FIXED_GAME_SIZE = 100
HISTORY_LEN = 20 # TensorBoardの履歴数
MAX_STEPS = 400          # 1ゲームの最大手数
BUF_SIZE = 7200 # ReplayBufferのサイズ (N_PROCESSES * 数サイクル分は最低限必要)
REG_UPDATE_INTERVAL = 4000
ETA = 0.0002

NON_ATTACK_DRAW = 100

DRAW_PENALTY = [0,0,0]
PENALTY_CHANGE = [5000, 30000, 100000]

LEARNING_RATE = 0.0005

LEARN_INTERVAL = 1
BATTLE_ITERATION = 128

MODEL_SAVE_INTERVAL = 50

LOSS_DIR = "model_loss/deepnash_mp"
MODEL_DIR = "models/deepnash_mp"

MODEL_NAME = "mini_cnn_v10"

CONFIG_PATH = "mini_board_config2.json"

T_BOARD = SimpleTensorBoard
C_AGENT = DeepNashCnnAgent
C_LEARNER = DeepNashCnnLearner

# --- グローバル変数 (ワーカープロセス内でのみ有効) ---
global_agent = None
global_lock = None
global_buffer = {} # バッファの各要素を辞書で持つか、個別の変数にする
global_env: Environment = None
global_battles = 0
global_win_counts = {}

# --- ワーカー初期化関数 ---
def init_worker(
    in_channels, mid_channels, device_str, 
    # 以下、共有オブジェクトを追加
    lock, 
    head, length, 
    boards, actions, rewards, policies, non_legals, players, mask, t_effective,
    shared_battle_counter,
    win_counts
):
    global global_agent, global_lock, global_buffer, global_env, global_battles, global_win_counts
    Config.load(CONFIG_PATH, JUDGE_TABLE)
    
    #1プロセスあたりのスレッド数を1に制限 (CPUの奪い合いを防ぐ)
    torch.set_num_threads(1)
    
    # 1. Agent生成 (前回と同じ)
    device = torch.device(device_str)
    tensorboard = T_BOARD(Config.board_shape, device, history=HISTORY_LEN)
    tensorboard.set_max_step(MAX_STEPS, NON_ATTACK_DRAW)
    global_agent = C_AGENT(in_channels, mid_channels, device, tensorboard)
    global_agent.network.eval() 
    
    #環境もここで生成
    cppJudge = GSC.MakeJudgeBoard(CONFIG_PATH)
    judge = CppJudgeBoard(cppJudge)
    global_env = Environment(judge, max_step=MAX_STEPS, max_non_attack=NON_ATTACK_DRAW)
    
    # 2. 共有オブジェクトをグローバル変数に保持 (これが重要！)
    global_lock = lock
    
    # まとめて辞書に入れておくと扱いやすいです
    global_buffer = {
        "head": head,
        "length": length,
        "boards": boards,
        "actions": actions,
        "rewards": rewards,
        "policies": policies,
        "non_legals": non_legals,
        "players": players,
        "mask": mask,
        "t_effective": t_effective
    }
    
    global_battles = shared_battle_counter
    global_win_counts = win_counts

# --- ヘルパー関数 (自己対戦プロセス内で使用) ---

def get_agent_output(agent: C_AGENT, env: Environment, device: torch.device, non_legal_action:int = -1):
    """
    Agentからアクションだけでなく、学習に必要なPolicyなども取得するヘルパー関数
    (DeepNashAgent.get_action を拡張したような処理)
    """
    agent.network.eval()
    
    # 現在の手番の盤面取得
    obs_tensor = agent.get_obs(env.get_current_player()).to(device).unsqueeze(0) # (1, C, H, W)
    
    # 合法手マスク作成
    legals = env.legal_move()
    if len(legals) == 0:
        return -1, None, None # 投了
        
    non_legal_mask = np.ones((Config.board_shape_int**2), dtype=bool)
    non_legal_mask[legals] = False
    if(not non_legal_action == -1): 
        non_legal_mask[non_legal_action] = True

    if np.all(non_legal_mask):
        return -1, None, None

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
    max_steps: int,
    iter_length: int,
    draw_penalty: int
):
    global global_agent, global_lock, global_buffer, global_env, global_battles
    
    """
    1エピソード分の自己対戦を実行し、Episodeオブジェクトと勝者を返す
    """
    device = global_agent.device

    # 1. Agent & Environment Initialization (プロセスごとに独立して生成)
    global_agent.load_state_dict(agent_state_dict)
    
    while global_battles.value <  iter_length:
        # 2. Self-play Episode
        global_env.reset()
        global_agent.reset()
    
        sample_obs = global_agent.get_obs(global_env.get_current_player())
        current_episode = Episode(sample_obs.shape, max_step=max_steps)
    
        obs = global_agent.get_obs(global_env.get_current_player())
    
        done = False
        step_count = 0

        non_legal_actions = {GSC.Player.PLAYER_ONE: -1, GSC.Player.PLAYER_TWO: -1}
    
        while not done and step_count < max_steps:
            current_player = global_env.get_current_player()
        
            action, policy, non_legal = get_agent_output(global_agent, global_env, device, non_legal_actions[current_player])
        
            if action == -1:
                _, log, frag = global_env.step(-1)
                done = True
            else:
                _, log, frag = global_env.step(action)
                global_agent.step(log,frag)
            
                if frag != GSC.BattleEndFrag.CONTINUE and frag != GSC.BattleEndFrag.DEPLOY_END:
                    done = True
                    
                bef,aft = get_action(log.action)

                #if(not global_env.is_deploy()):
                #    non_legal_actions[current_player] = make_action(aft,bef)
            
                trac = Trajectory(
                    board=obs, # CPUに送るのはadd_step内で行われる
                    action=action,
                    reward=0.0, # 後で設定
                    policy=policy.detach(),
                    player=current_player,
                    non_legal=non_legal.detach()
                )
                current_episode.add_step(trac)
                obs = global_agent.get_obs(global_env.get_current_player())
            
            step_count += 1

        winner = global_env.get_winner()
        # 3. Reward Calculation
        p1_reward = 0.0
        p2_reward = 0.0
        if winner == GSC.Player.PLAYER_ONE:
            p1_reward = 1.0
            p2_reward = -1.0
        elif winner == GSC.Player.PLAYER_TWO:
            p1_reward = -1.0
            p2_reward = 1.0
        else:
            winner = "DRAW"
            p1_reward = draw_penalty
            p2_reward = draw_penalty

        current_episode.set_reward_all(p1_reward, p2_reward)

        with global_lock:
            h = global_buffer["head"].value
            l = global_buffer["length"].value
            t = current_episode.t_effective

            global_buffer["boards"][h,:t] = current_episode.boards[:t]
            global_buffer["actions"][h,:t] = current_episode.actions[:t]
            global_buffer["rewards"][h,:t] = current_episode.rewards[:t]
            global_buffer["policies"][h,:t] = current_episode.policies[:t]
            global_buffer["non_legals"][h,:t] = current_episode.non_legals[:t]
            global_buffer["players"][h, :t] = current_episode.players[:t]
            global_buffer["mask"][h,:t] = True
            global_buffer["t_effective"][h] = t

            global_buffer["head"].value = (h + 1) % BUF_SIZE
            global_buffer["length"].value = min(l + 1, BUF_SIZE)

            global_battles.value += 1

            global_win_counts[winner].value += 1

        #current_episode.episode_end()

# --- メイン学習プロセス ---

def main():
    Config.load(CONFIG_PATH, JUDGE_TABLE)
    
    print(f"Main Device: {MAIN_DEVICE}")
    print(f"Worker Device: {WORKER_DEVICE_STR}")
    print(f"Num Processes: {N_PROCESSES}")

    # 1. Agent, Learner, Buffer Initialization
    in_channels = T_BOARD.get_tensor_channels(HISTORY_LEN)
    mid_channels = in_channels*3//2
    
    learner = C_LEARNER(in_channels, mid_channels, MAIN_DEVICE, lr=LEARNING_RATE, reg_update_interval=REG_UPDATE_INTERVAL, eta=ETA)
    replay_buffer = ReplayBuffer(size=BUF_SIZE, max_step=MAX_STEPS, board_shape=[in_channels, Config.board_shape[0], Config.board_shape[1]])
    replay_buffer.mp_set()
    
    total_episodes = 0

    ctx = mp.get_context('spawn')
    with ctx.Manager() as manager:
        lock = manager.Lock()
        battle_counter = manager.Value("i",0)
        win_counts = {GSC.Player.PLAYER_ONE: manager.Value("i",0), GSC.Player.PLAYER_TWO: manager.Value("i",0), "DRAW": manager.Value("i",0)}
    
        with ctx.Pool(
                processes=N_PROCESSES, 
                initializer=init_worker, 
                initargs=(
                    in_channels, mid_channels, WORKER_DEVICE_STR,
                    lock,
                    replay_buffer.head, 
                    replay_buffer.length,
                    replay_buffer.boards,
                    replay_buffer.actions,
                    replay_buffer.rewards,
                    replay_buffer.policies,
                    replay_buffer.non_legals,
                    replay_buffer.players,
                    replay_buffer.mask,
                    replay_buffer.t_effective,
                    battle_counter,
                    win_counts
                )
            ) as pool:
            for i in tqdm(range(TOTAL_CYCLES), desc="Training Cycles"):
                
                # 最新のモデルパラメータをCPUにコピーしてワーカーに渡す
                current_state_dict = learner.get_current_network_state_dict()
                cpu_state_dict = {}
                for k, v in current_state_dict.items():
                # "_orig_mod." がついていたら削除する
                    new_key = k.replace("_orig_mod.", "")
                    cpu_tensor = v.cpu()
                    cpu_tensor.share_memory_()
                    cpu_state_dict[new_key] = cpu_tensor

                draw_penalty = 0
                for v,j in enumerate(PENALTY_CHANGE):
                    if(i <= j):
                        draw_penalty = DRAW_PENALTY[v]
                        break


                # 自己対戦を並列実行するための引数リストを作成
                battle_counter.value = 0
                args_list = [
                    (
                        pid,
                        cpu_state_dict,
                        MAX_STEPS,
                        BATTLE_ITERATION,
                        draw_penalty
                    ) for pid in range(N_PROCESSES)
                ]

                # 2. Self-play Generation (in parallel)
                pool.starmap(run_self_play_episode, args_list)
        
                total_episodes = win_counts[GSC.Player.PLAYER_ONE].value + win_counts[GSC.Player.PLAYER_TWO].value + win_counts["DRAW"].value

                # 4. Learning Step
                if len(replay_buffer) >= BATCH_SIZE*ACCUMRATION and (i+1)%LEARN_INTERVAL == 0:
                    loss_path = f"{LOSS_DIR}/{MODEL_NAME}"
                    os.makedirs(loss_path, exist_ok=True)
                    learner.learn(replay_buffer, BATCH_SIZE, FIXED_GAME_SIZE, ACCUMRATION, loss_path)

                    # 学習後のモデルパラメータをagentに反映
        
                # 5. Logging and Saving
                # 約100エピソードごとにログ出力
                if (i+1)%MODEL_SAVE_INTERVAL == 0:
                    p1_wins = win_counts[GSC.Player.PLAYER_ONE].value
                    p2_wins = win_counts[GSC.Player.PLAYER_TWO].value
                    draws = win_counts["DRAW"].value
                    print(f"\nTotal Episodes: {total_episodes}: P1 Wins: {p1_wins}, P2 Wins: {p2_wins}, Draws: {draws}")

                    # モデルの保存
                    save_path = f"{MODEL_DIR}/{MODEL_NAME}/model_{i+1}.pth"
                    os.makedirs(f"{MODEL_DIR}/{MODEL_NAME}", exist_ok=True)
                    torch.save(learner.target_network.state_dict(), save_path)
                    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # 'spawn' を使うことで、CUDA利用時のfork関連のエラーを回避します
    #mp.set_start_method('spawn', force=True)
    main()