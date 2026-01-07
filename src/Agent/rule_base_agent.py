from src.Interfaces import IAgent, IEnv
from src.const import BOARD_SHAPE, GOAL_POS, ENTRY_HEIGHT, Piece, PIECE_KINDS, GOAL_HEIGHT, GOAL_PIECES, PIECE_LIMIT
from src.common import LogData, change_pos_int_to_tuple, get_action, Config

from src.GunjinShogi.const import JUDGE_TABLE
import GunjinShogiCore as GSC


import numpy as np
import torch
import random

# 駒の重要度（価値）を定義
# Piece Enumのint値(1~16)をキーとしています
PIECE_VALUES = {
    Piece.General: 1200,           # 大将 (1)
    Piece.LieutenantGeneral: 1000, # 中将 (2)
    Piece.MajorGeneral: 800,       # 少将 (3)
    Piece.Colonel: 600,            # 大佐 (4)
    Piece.LieutenantColonel: 500,  # 中佐 (5)
    Piece.Major: 400,              # 少佐 (6)
    Piece.Captain: 300,            # 大尉 (7)
    Piece.FirstLieunant: 200,      # 中尉 (8)
    Piece.SecondLieunant: 100,     # 少尉 (9)
    Piece.Plane: 500,              # ヒコーキ (10) - 機動力
    Piece.Tank: 1300,              # タンク (11) - 戦闘力高
    Piece.Cavalry: 50,             # 騎兵 (12)
    Piece.Engineer: 150,           # 工兵 (13) - タンク・地雷対策
    Piece.Spy: 1500,               # スパイ (14) - 大将キラー
    Piece.LandMine: 200,           # 地雷 (15)
    Piece.Frag: 10000,             # 軍旗 (16) - 絶対守る
}

FIRST_DICT = {
    Piece.General: [0],
    Piece.LieutenantGeneral: [1],
    Piece.MajorGeneral: [2],
    Piece.Colonel: [3],
    Piece.LieutenantColonel: [4],
    Piece.Major: [5],
    Piece.Captain: [6,7],
    Piece.FirstLieunant: [8,9],
    Piece.SecondLieunant: [10,11],
    Piece.Plane: [12,13],
    Piece.Tank: [14,15],
    Piece.Cavalry: [16],
    Piece.Engineer: [17,18],
    Piece.Spy: [19],
    Piece.LandMine: [20,21],
    Piece.Frag: [22]
}

class RuleBaseAgent(IAgent):
    def __init__(self):
        self.head = 0
        self.pieces = self.chg_init_form(self.get_random_formation())
        
    def chg_init_form(self, form):
        heads = [0 for _ in FIRST_DICT]
        pieces = []
        for p in form:
            pieces.append(FIRST_DICT[p][heads[p-1]])
            heads[p-1] += 1
            
        return pieces
        
    def get_random_formation(self):
        """いくつかの定石からランダムに初期配置を選択する"""
        formations = [
            self.formation_fortress_right(), # 右下要塞型
            self.formation_blitz(),          # 速攻型
            self.formation_decoy_left()      # 左下本陣（ブラフ）型
        ]
        return random.choice(formations)

    def formation_fortress_right(self):
        """
        【右下要塞型】
        軍旗を右下(22)に置き、地雷で囲う基本形。
        前線には機動力のあるタンクやヒコーキ、中将を配置。
        大将は中段で遊撃する。
        """
        # 6x4 = 24マス, 23駒。
        # 0-5: 最前列, 6-11: 2列目, 12-17: 3列目, 18-22: 最後列(右下)
        return [
            # --- 最前列 (0-5) ---
            Piece.Tank, Piece.LieutenantGeneral, Piece.MajorGeneral, Piece.Plane, Piece.Cavalry, Piece.Captain,
            # --- 2列目 (6-11) ---
            Piece.Spy, Piece.General, Piece.Engineer, Piece.Major, Piece.Tank, Piece.Plane,
            # --- 3列目 (12-17) ---
            Piece.FirstLieunant, Piece.Colonel, Piece.Captain, Piece.Engineer, Piece.LandMine, Piece.Frag, 
            # --- 最後列 (18-22) ---
            Piece.SecondLieunant, Piece.FirstLieunant, Piece.SecondLieunant, Piece.LandMine, Piece.LieutenantColonel
        ]

    def formation_blitz(self):
        """
        【速攻型】
        大将・中将・飛行機・タンクを最前列に並べる。
        守りは弱いが、序盤の制圧力を最大化する。
        軍旗は中央後方に隠す。
        """
        return [
            # --- 最前列 (0-5) 強力な駒を集中 ---
            Piece.General, Piece.Tank, Piece.Plane, Piece.LieutenantGeneral, Piece.Tank, Piece.Plane,
            # --- 2列目 (6-11) 中堅とスパイ ---
            Piece.Spy, Piece.MajorGeneral, Piece.Colonel, Piece.LieutenantColonel, Piece.Major, Piece.Engineer,
            # --- 3列目 (12-17) 下級士官と地雷 ---
            Piece.Captain, Piece.LandMine, Piece.Captain, Piece.FirstLieunant, Piece.LandMine, Piece.FirstLieunant,
            # --- 最後列 (18-22) ---
            Piece.Engineer, Piece.Cavalry, Piece.Frag, Piece.SecondLieunant, Piece.SecondLieunant
        ]

    def formation_decoy_left(self):
        """
        【左下本陣（ブラフ）型】
        軍旗を左下(18)に配置し、地雷で守る。
        右側をわざと手薄に見せかけたり、逆に右側に主力を集めて
        相手の攻撃を右に誘導しつつ、本陣（左）を守る。
        """
        return [
            # --- 最前列 (0-5) ---
            Piece.MajorGeneral, Piece.Tank, Piece.Colonel, Piece.Plane, Piece.LieutenantGeneral, Piece.Spy,
            # --- 2列目 (6-11) ---
            Piece.Captain,  Piece.Frag, Piece.Tank, Piece.Plane, Piece.Engineer, Piece.General,
            # --- 3列目 (12-17) ---
            Piece.LandMine, Piece.FirstLieunant, Piece.SecondLieunant, Piece.LieutenantColonel, Piece.Captain, Piece.FirstLieunant,
            # --- 最後列 (18-22) 左端(18)に軍旗 ---
            Piece.Major, Piece.LandMine, Piece.Engineer, Piece.Cavalry, Piece.SecondLieunant
        ]
    
    def deploy_action(self):
        act = self.pieces[self.head]
        self.head += 1
        return act
    
    def get_action(self, env: IEnv):
        tensor = env.get_tensor_board_current()
        
        legals = env.legal_move()
        if(env.is_deploy()):
            return self.deploy_action()
        
        self.head = 0
        if(len(legals) == 0): return -1
        best_action = np.random.choice(legals)
        best_score = -float('inf')
        
        # --- Tensor情報の解析用定数 ---
        # 0~15: 自分の駒 (Piece.General(1) -> ch 0, ... Piece.Frag(16) -> ch 15)
        MY_CH_START = 0
        # 16~31: 敵の駒の確率 (Piece.General(1) -> ch 16, ... )
        ENEMY_CH_START = 16 
        
        for action in legals:
            score = 0
            
            # アクションを座標に変換
            bef_idx, aft_idx = get_action(action)
            bef_pos = change_pos_int_to_tuple(bef_idx) # (x, y)
            aft_pos = change_pos_int_to_tuple(aft_idx) # (x, y)
            
            # 1. 自分の駒の種類を特定する
            # Tensorのチャネル0~15を調べて、1が立っている場所を探す
            my_piece_val = -1
            for i in range(PIECE_KINDS):
                if tensor[MY_CH_START + i, bef_pos[0], bef_pos[1]] > 0.5: # float考慮
                    my_piece_val = i + 1 # Enum値は1始まり
                    break
            
            if my_piece_val == -1:
                continue # エラー回避（通常ありえない）

            my_piece_enum = Piece(my_piece_val)

            # 2. 移動先に敵がいるか確認する
            # 敵情報チャネル(16~31)の合計が0より大きければ敵がいる（確率的に存在）
            enemy_probs = tensor[ENEMY_CH_START : ENEMY_CH_START + PIECE_KINDS, aft_pos[0], aft_pos[1]]
            enemy_exist_prob = torch.sum(enemy_probs).item()
            
            # --- 評価ロジック ---

            # A. ゴール判定
            # (GOAL_HEIGHT等はconstから取得。敵陣奥深くにゴールがあると仮定)
            # ここではY座標がEntry Heightと同じで、かつGoal Posに含まれるならゴール
            if (aft_pos[1] == GOAL_HEIGHT) and (aft_pos[0] in GOAL_POS) and my_piece_enum in GOAL_PIECES:
                return action # 即決

            # B. 攻撃（敵がいる確率が高い）の場合
            if enemy_exist_prob > 0.01: 
                expected_gain = 0
                
                # 相手の種類の確率分布に基づいて期待値を計算
                for i in range(PIECE_KINDS):
                    prob = enemy_probs[i].item()
                    if prob <= 0: continue
                    
                    enemy_piece_val = i + 1
                    
                    # JUDGE_TABLE を使って勝敗判定
                    # JUDGE_TABLE[my][enemy] -> 1:Win, 0:Draw, -1:Lose
                    result = JUDGE_TABLE[my_piece_val][enemy_piece_val]
                    
                    my_v = PIECE_VALUES.get(my_piece_val, 100)
                    enemy_v = PIECE_VALUES.get(enemy_piece_val, 100)
                    
                    if result == 1:   # 勝ち
                        expected_gain += prob * (enemy_v + 50) # 敵撃破ボーナス
                    elif result == -1: # 負け
                        expected_gain -= prob * my_v
                    else:             # 引き分け (相打ち)
                        expected_gain += prob * (enemy_v - my_v)

                # スパイ特攻ボーナス: 相手が大将の確率が高い場合
                if my_piece_enum == Piece.Spy:
                    general_prob = enemy_probs[Piece.General - 1].item()
                    if general_prob > 0.15:
                        expected_gain += 2000 * general_prob

                score += expected_gain * 2.0 # 攻撃行動の重みを調整

            # C. 移動（空きマス）の場合
            else:
                # 1. 危険な駒は動かさない
                if my_piece_enum in [Piece.LandMine, Piece.Frag]:
                    score -= 5000 # 基本動かない
                
                # 2. 前進を推奨 (敵陣方向へのY移動をプラス評価)
                # プレイヤー視点でtensorが正規化されていない場合、
                # 敵陣がY=ENTRY_HEIGHT(大きい方)にあると仮定
                dist_gain = (aft_pos[1] - bef_pos[1]) 
                score += dist_gain * 20 

                # 3. 機動力のある駒の優遇
                if my_piece_enum in [Piece.Plane, Piece.Tank, Piece.Cavalry]:
                    score += 30

                # 4. 少しのランダム性（ループ防止）
                score += random.uniform(0, 5)

            # 最良手の更新
            if score > best_score:
                best_score = score
                best_action = action
            elif score == best_score:
                # 同点なら半々の確率で更新（探索の多様性）
                if random.random() < 0.5:
                    best_action = action
        
        return best_action
    
    def get_first_board(self):
        pieces = np.arange(PIECE_LIMIT)
        np.random.shuffle(pieces)
        return pieces
    
    def step(self, log:LogData, frag: GSC.BattleEndFrag):
        pass
    
    def reset(self):
        pass
    
class SimpleRuleBaseAgent(RuleBaseAgent):
    def __init__(self):
        super().__init__()
        self.head = 0
        self.pieces = np.arange(Config.piece_limit)
        self.pieces = np.random.permutation(self.pieces)
        
    def reset(self):
        self.head = 0
        self.pieces = np.arange(Config.piece_limit)
        self.pieces = np.random.permutation(self.pieces)
        
    def get_action(self, env):
        legals = env.legal_move()
        if(env.is_deploy()):
            return self.deploy_action()
        
        self.head = 0
        if(len(legals) == 0): return -1
        best_action = np.random.choice(legals)
        best_score = -float('inf')
        
        board = env.get_int_board(env.get_current_player())
        
        for action in legals:
            score = 0
            
            # アクションを座標に変換
            bef_idx, aft_idx = get_action(action)
            bef_pos = change_pos_int_to_tuple(bef_idx) # (x, y)
            aft_pos = change_pos_int_to_tuple(aft_idx) # (x, y)
            
            dist_to_goal = aft_pos[0] - GOAL_POS[0] + aft_pos[1] - GOAL_HEIGHT
            
            score = 5 - dist_to_goal
            
            if score > best_score:
                best_score = score
                best_action = action
            elif score == best_score:
                # 同点なら半々の確率で更新（探索の多様性）
                if random.random() < 0.5:
                    best_action = action
            
        return best_action