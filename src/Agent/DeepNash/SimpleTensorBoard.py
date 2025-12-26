from src.const import Piece

from src.common import Config

from src.GunjinShogi.Board import Board
from src.GunjinShogi.const import JUDGE_TABLE

from src.common import EraseFrag, get_action, change_pos_tuple_to_int, \
    change_pos_int_to_tuple, make_reflect_pos, make_reflect_pos_int, Config
    
from src.Agent.DeepNash.ITensorBoard import ITensorBoard

import numpy as np
import torch

import GunjinShogiCore as GSC

WALL_ENTRY_GOAL_CHANNEL = 3
ENEMY_INFO_CHANNEL = 1
DEPLOY_INFO_CHANNEL = 1
STEPS_INFO_CHANNEL = 2

class SimpleTensorBoard(Board,ITensorBoard):
    @classmethod
    def get_tensor_channels(cls, history):
        return Config.piece_kinds + ENEMY_INFO_CHANNEL + WALL_ENTRY_GOAL_CHANNEL + \
            DEPLOY_INFO_CHANNEL + STEPS_INFO_CHANNEL + history
    
    
    def __init__(self, size: tuple[int, int], device: torch.device, history = 30):
        super().__init__(size)
        
        self._device = device
        
        self._history_len = history
        
        # tensor: 自分の駒(16種) + 敵駒(1種) + 履歴(history)
        # channel 0-15: Piece 1-16 (自分の駒)
        # channel 16: Enemy (敵駒: -1)
        # channel 17 ~ 17+history-1: History
        self._base_channels = Config.piece_kinds + ENEMY_INFO_CHANNEL + WALL_ENTRY_GOAL_CHANNEL + \
            DEPLOY_INFO_CHANNEL + STEPS_INFO_CHANNEL 
        self._total_channels = self._base_channels + self._history_len
        
        self._piece_channels = Config.piece_kinds
        
        self.deploy_tensor_pos = Config.piece_kinds + ENEMY_INFO_CHANNEL + WALL_ENTRY_GOAL_CHANNEL
        
        self._step_tensor_pos = self.deploy_tensor_pos + DEPLOY_INFO_CHANNEL
        
        self.reset()
        
        self.piece_dict = np.array(Config.first_dict)
        
        self.judge_table = torch.from_numpy(Config.judge_table).clone().to(self._device)
        
        self.deploy_heads = {GSC.Player.PLAYER_ONE:0, GSC.Player.PLAYER_TWO:0}
        
        # マスクの事前計算: Config.tensor_piece_id に含まれる駒のみを考慮
        # True: 移動不可 (確率を0にする対象), False: 移動可能
        def _make_mask(allowed_pieces):
            mask = torch.ones(Config.piece_kinds, dtype=torch.bool, device=self._device)
            indices = [Config.get_tensor_id(p) for p in allowed_pieces if p in Config.tensor_piece_id]
            if indices:
                mask[torch.tensor(indices, dtype=torch.long, device=self._device)] = False
            return mask
        
        self.mask_frag = _make_mask([Piece.Frag])
        self.anti_mask_frag = ~self.mask_frag
        self.mask_landmine = _make_mask([Piece.LandMine])
        self.anti_mask_landmine = ~self.mask_landmine

        self.mask_vertical_long = _make_mask([Piece.Engineer, Piece.Plane])
        self.mask_vertical_step_2 = _make_mask([Piece.Tank, Piece.Cavalry, Piece.Plane, Piece.Engineer])
        self.mask_horizontal_long = _make_mask([Piece.Engineer])
        self.mask_jump = _make_mask([Piece.Plane])
        
        self.max_step = 1000
        self.max_non_attack = 200
        
        self.non_attack = 0
        self.steps = 0
        
        
    @property
    def total_channels(self) -> int:
        return self._total_channels
        
    def wall_set(self, tensor: torch.Tensor, board: np.ndarray):
        wall_channel = self._piece_channels + Config.piece_kinds
        entry_channel = wall_channel + 1
        goal_channel = entry_channel + 1
        
        for x in range(Config.board_shape[0]):
            if(x in Config.entry_pos): 
                tensor[entry_channel,x,Config.entry_height] = 1
                board[change_pos_tuple_to_int(x,Config.entry_height)] = Piece.Entry #Entry(-2)
            else: 
                tensor[wall_channel,x,Config.entry_height] = 1
                board[change_pos_tuple_to_int(x,Config.entry_height)] = Piece.Wall #Wall
                
            if(x in Config.goal_pos):
                tensor[goal_channel,x,0] = 1
                
    def step_value_set(self, tensor) -> None:
        tensor[self._step_tensor_pos] = self.steps/self.max_step
        tensor[self._step_tensor_pos+1] = self.non_attack/self.max_non_attack
        
    
    def deploy_set(self, piece, player:GSC.Player):
        
        i = 0 if player == GSC.Player.PLAYER_ONE else 1
        mi = i - 1
        
        tensor = self.tensors[0] if player == GSC.Player.PLAYER_ONE else self.tensors[1]
        oppose = self.tensors[1] if player == GSC.Player.PLAYER_ONE else self.tensors[0]
        head = self.deploy_heads[player]
        
        x = head % Config.board_shape[0]
        y = (head//Config.board_shape[0]) + Config.entry_height + 1
        
        if(y == Config.reflect_goal_height and x in Config.goal_pos):
            self.deploy_heads[player] += len(Config.goal_pos)-1
        
        rx,ry = make_reflect_pos((x,y))
        
        board = self._boards[0] if player == GSC.Player.PLAYER_ONE else self._boards[1]
        o_board = self._boards[1] if player == GSC.Player.PLAYER_ONE else self._boards[0]
        
        board[change_pos_tuple_to_int(x,y)] = piece
        o_board[change_pos_tuple_to_int(rx,ry)] = -1
        
        tensor[Config.get_tensor_id(piece),x,y] = 1
        oppose[self._piece_channels,rx,ry] = 1
        
        self.oppose_presition_pool[i, :, x,y] = 1/Config.piece_kinds
        self.private_presition_pool[mi, :, rx,ry] = 1/Config.piece_kinds
        
        if(player == GSC.Player.PLAYER_ONE): self.global_pool[Config.get_tensor_id(piece)] += 1
        
        self.deploy_heads[player] += 1
            
        self.steps += 1
        
    def deploy_end(self) -> None:
        self._tensor_p1[self.deploy_tensor_pos].fill_(1)
        self._tensor_p2[self.deploy_tensor_pos].fill_(1)
        
        self.deploy = False
        
    def reset(self) -> None:
        Board.reset(self)
        
        BOARD_SHAPE = Config.board_shape
        PIECE_KINDS = Config.piece_kinds

        #tensor:自分の駒の位置+敵駒+履歴(論文と同じもの×30)
        self._tensor_p1 = torch.zeros([self._total_channels,BOARD_SHAPE[0],BOARD_SHAPE[1]], dtype=torch.float32, device=self._device)
        self._tensor_p2 = torch.zeros([self._total_channels,BOARD_SHAPE[0],BOARD_SHAPE[1]], dtype=torch.float32, device=self._device)
        
        self.tensors = [self._tensor_p1, self._tensor_p2]
        
        for t,b in zip(self.tensors,self._boards):
            self.wall_set(t,b)
            
        self.first_p1: np.typing.NDArray[np.int32] = np.zeros(22)
        self.first_p2: np.typing.NDArray[np.int32] = np.zeros(22)
        
        self.deploy = True
        self.deploy_heads = {GSC.Player.PLAYER_ONE:0, GSC.Player.PLAYER_TWO:0}
        
        self.global_pool = torch.zeros(PIECE_KINDS, dtype=torch.float32, device=self._device)
        
        #自分から見た相手の情報
        #現存する駒
        self.private_presition_pool = torch.zeros((2,PIECE_KINDS,BOARD_SHAPE[0],BOARD_SHAPE[1]), dtype=torch.float32, device=self._device)
        #相手の駒
        self.private_dead_pool = torch.zeros((2,PIECE_KINDS), dtype=torch.float32, device=self._device)
        
        #相手から見た自分の情報
        #実装むずいのでまだ
        self.oppose_dead_pool = torch.zeros((2,PIECE_KINDS), dtype=torch.float32, device=self._device)
        self.oppose_presition_pool = torch.zeros((2,PIECE_KINDS,BOARD_SHAPE[0],BOARD_SHAPE[1]), dtype=torch.float32, device=self._device)
        
        self.non_attack = 0
        self.steps = 0
        
    
    def tensor_move(self, tensor:torch.Tensor, bef:tuple[int,int], aft:tuple[int,int], piece:int):
        layer:int = -1
        if(piece in Config.tensor_piece_id): layer = Config.get_tensor_id(piece)
        elif(piece == Piece.Enemy): layer = Config.piece_kinds
        else: return
        
        tensor[layer,bef[0],bef[1]] = 0
        tensor[layer,aft[0],aft[1]] = 1
        
    def tensor_erase(self, tensor:torch.Tensor, pos:tuple[int,int], piece:int):
        layer:int = -1
        if(piece in Config.tensor_piece_id): layer = Config.get_tensor_id(piece)
        elif(piece == Piece.Enemy): layer = Config.piece_kinds
        else: return
        
        tensor[layer,pos[0],pos[1]] = 0
        
    def update_history(self, tensor: torch.Tensor, bef: tuple[int, int], aft: tuple[int, int], win: int, bef_piece:Piece, aft_piece:Piece):
        """履歴チャンネルの更新 (Shift & Record)"""
        if self._history_len == 0:
            return

        # 1. 履歴を1つ古くする（チャンネルをシフト）
        # channel[18] <- channel[17], channel[19] <- channel[18] ...
        # clone()を使わないと参照コピーになりバグる可能性があるため注意
        hist_start = self._base_channels
        tensor[hist_start + 1:] = tensor[hist_start:-1].clone()
        
        # 2. 最新の履歴(channel 17)をクリアして書き込む
        tensor[hist_start].zero_()
        
        # 移動元: -1, 移動先: 1
        tensor[hist_start, bef[0], bef[1]] = -0.5
        
        if(aft_piece == Piece.Space):
           tensor[hist_start, aft[0], aft[1]] = 0.5
           return
       
        if(bef_piece == Piece.Enemy):
            ap = Config.get_tensor_id(aft_piece)
            if(win == 1):
                tensor[hist_start, aft[0], aft[1]] = (0.5+(ap+1)/Config.piece_kinds*0.5)
            else:
                tensor[hist_start, aft[0], aft[1]] = -(0.5+(ap+1)/Config.piece_kinds*0.5)
                
        else:
            bp = Config.get_tensor_id(bef_piece)
            if(win == 1):
                tensor[hist_start, aft[0], aft[1]] = (0.5+(bp+1)/Config.piece_kinds*0.5)
            else:
                tensor[hist_start, aft[0], aft[1]] = -(0.5+(bp+1)/Config.piece_kinds*0.5)
        
        
    
    #actionに合わせてtensorを変化させる
    #どこに何を動かしたかはself.boardから取得できる
    def step(self, action: int, player: GSC.Player, erase: GSC.EraseFrag):
        bef,aft = get_action(action)
        o_bef,o_aft = make_reflect_pos_int(bef), make_reflect_pos_int(aft)
        bef_t = change_pos_int_to_tuple(bef)
        aft_t = change_pos_int_to_tuple(aft)
        
        o_bef_t = make_reflect_pos(bef_t)
        o_aft_t = make_reflect_pos(aft_t)
        
        p_idx = 0 if player == GSC.Player.PLAYER_ONE else 1
        o_p_idx = 1-p_idx
        
        board = self._boards[p_idx]
        o_board = self._boards[o_p_idx]
        bef_piece = board[bef]
        aft_piece = board[aft]
        o_bef_piece = o_board[o_bef]
        o_aft_piece = o_board[o_aft]
        
        tensor = self.tensors[p_idx]
        o_tensor = self.tensors[o_p_idx]
        
        win = 0
        if(erase == GSC.EraseFrag.AFT): win = 1
        elif(erase == GSC.EraseFrag.BEF): win = -1
        o_win = -win
        
        if(erase == GSC.EraseFrag.BEF):
            self.tensor_erase(tensor, bef_t, bef_piece)
            self.tensor_erase(o_tensor, o_bef_t, o_bef_piece)
        elif(erase == GSC.EraseFrag.AFT):
            self.tensor_erase(tensor, aft_t, aft_piece)
            self.tensor_erase(o_tensor, o_aft_t, o_aft_piece)
        elif(erase == GSC.EraseFrag.BOTH):
            self.tensor_erase(tensor, bef_t, bef_piece)
            self.tensor_erase(o_tensor, o_bef_t, o_bef_piece)
            self.tensor_erase(tensor, aft_t, aft_piece)
            self.tensor_erase(o_tensor, o_aft_t, o_aft_piece)
        
        if(erase != GSC.EraseFrag.BEF):
            self.tensor_move(tensor, bef_t, aft_t, bef_piece)
            self.tensor_move(o_tensor, o_bef_t, o_aft_t, o_bef_piece)
            
        self.update_history(tensor,bef_t,aft_t,win,bef_piece,aft_piece)
        self.update_history(o_tensor,o_bef_t,o_aft_t,o_win, o_bef_piece, o_aft_piece)
        
        self.steps += 1
        if(aft_piece == Piece.Space):
            self.non_attack += 1
        else:
            self.non_attack = 0
        
        self.step_value_set(tensor)
        self.step_value_set(o_tensor)
        
        Board.step(self,action,p_idx+1,erase)
        
    
    def undo(self) -> bool:
        pass
    
    def set_board(self, board_player1: np.ndarray, board_player2: np.ndarray, deploy = False) -> None:
        Board.set_board(self,board_player1, board_player2)
        
        # Tensorを初期化
        self._tensor_p1.zero_()
        self._tensor_p2.zero_()
        
        self._set_tensor_from_board(self._tensor_p1, board_player1)
        self._set_tensor_from_board(self._tensor_p2, board_player2)
        
        self.deploy = deploy
        for t,b in zip(self.tensors,self._boards):
            self.wall_set(t,b)
        if(not self.deploy):
            self.deploy_end()

    def _set_tensor_from_board(self, tensor: torch.Tensor, board_array: np.ndarray):
        """1次元のボード配列からTensorを設定するヘルパー関数"""
        width = Config.board_shape[0]
        
        for i, piece in enumerate(board_array):
            if piece <= 0 and piece != -1: continue # Space(0), Entry(-2), Wall(-100) は無視
            
            x = i % width
            y = i // width
            
            layer = -1
            if 1 <= piece <= Config.piece_kinds:
                layer = Config.get_tensor_id(piece)
                tensor[layer, x, y] = 1
            elif piece == -1: # Enemy
                tensor[Config.piece_kinds, x, y] = 1
                
                
    def get_defined_board(self, pieces: np.ndarray, player: GSC.Player, deploy = False) -> "SimpleTensorBoard":
        defined_tensor = SimpleTensorBoard(Config.board_shape, self._device, self._history_len)
        defined_tensor.deploy_heads = self.deploy_heads.copy()
        
        player_board = self._boards[0] if player == GSC.Player.PLAYER_ONE else self._boards[1]
        oppose_board = self._boards[1] if player == GSC.Player.PLAYER_ONE else self._boards[0]
        
        player_board = player_board.copy()
        defined_board = oppose_board.copy()
        changed_pieces = self.piece_dict[pieces]
        changed_pieces = changed_pieces[:(defined_board > 0).sum()][::-1]
        
        defined_board[defined_board > 0] = changed_pieces
        
        defined_tensor.set_board(player_board, defined_board, deploy)
        
        return defined_tensor
        
    def get_board_player1(self) -> torch.Tensor:
        return self._tensor_p1
    
    def get_board_player2(self) -> torch.Tensor:
        return self._tensor_p2
    
    def get_board(self, player: GSC.Player) -> torch.Tensor:
        return self._tensor_p1 if player == GSC.Player.PLAYER_ONE else self._tensor_p2
    
    def tensor_board_assert(self) -> bool:
        for v,t in enumerate(self.tensors):
            for y in range(Config.board_shape[1]):
                for x in range(Config.board_shape[0]):
                    piece = self._boards[v][change_pos_tuple_to_int(x,y)]
                    if(piece < 1 or piece > Config.piece_kinds): continue
                    if(t[Config.get_tensor_id(piece),x,y] != 1): return False
        return True
    
    def set_max_step(self, max_step: int, max_non_attack: int):
        self.max_step = max_step
        self.max_non_attack = max_non_attack