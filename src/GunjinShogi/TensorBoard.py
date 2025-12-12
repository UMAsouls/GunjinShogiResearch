from src.const import BOARD_SHAPE, PIECE_KINDS, ENTRY_HEIGHT, ENTRY_POS, GOAL_POS, PIECE_DICT, Piece
from src.GunjinShogi.Interfaces import ITensorBoard

from src.GunjinShogi.Board import Board

from src.common import EraseFrag, get_action, change_pos_tuple_to_int, change_pos_int_to_tuple, make_reflect_pos, make_reflect_pos_int

import numpy as np
import torch

import GunjinShogiCore as GSC

ENEMY_CHANNEL = 1
WALL_ENTRY_CHANNEL = 2

#Tensorの設定おかしいです
class TensorBoard(Board,ITensorBoard):
    def __init__(self, size: tuple[int, int], device: torch.device, history = 30):
        super().__init__(size)
        
        self._device = device
        
        self._history_len = history
        
        # tensor: 自分の駒(16種) + 敵駒(1種) + 履歴(history)
        # channel 0-15: Piece 1-16 (自分の駒)
        # channel 16: Enemy (敵駒: -1)
        # channel 17 ~ 17+history-1: History
        self._base_channels = PIECE_KINDS + ENEMY_CHANNEL + WALL_ENTRY_CHANNEL
        self._total_channels = self._base_channels + self._history_len
        
        self.reset()
        
        self.piece_dict = np.array(PIECE_DICT)
        
    @property
    def total_channels(self) -> int:
        return self._total_channels
        
    def wall_set(self, tensor: torch.Tensor, board: np.ndarray):
        wall_channel = self._base_channels + ENEMY_CHANNEL
        entry_channel = wall_channel + 1
        
        for x in range(BOARD_SHAPE[0]):
            if(x in ENTRY_POS): 
                tensor[entry_channel,x,ENTRY_HEIGHT] = 1
                board[change_pos_tuple_to_int(x,ENTRY_HEIGHT)] = Piece.Entry #Entry(-2)
            else: 
                tensor[wall_channel,x,ENTRY_HEIGHT] = 1
                board[change_pos_tuple_to_int(x,ENTRY_HEIGHT)] = Piece.Wall #Wall

    
    
    def deploy_set(self, piece, player:GSC.Player):
        
        tensor = self.tensors[0] if player == GSC.Player.PLAYER_ONE else self.tensors[1]
        oppose = self.tensors[1] if player == GSC.Player.PLAYER_ONE else self.tensors[0]
        head = self.deploy_heads[player]
        
        x = head % BOARD_SHAPE[0]
        y = (head//BOARD_SHAPE[0]) + ENTRY_HEIGHT + 1
        
        if(y == BOARD_SHAPE[1] - 1 and x in GOAL_POS):
            self.deploy_heads[player] += len(GOAL_POS)-1
        
        rx,ry = make_reflect_pos((x,y))
        
        board = self._boards[0] if player == GSC.Player.PLAYER_ONE else self._boards[1]
        o_board = self._boards[1] if player == GSC.Player.PLAYER_ONE else self._boards[0]
        
        board[change_pos_tuple_to_int(x,y)] = piece
        o_board[change_pos_tuple_to_int(rx,ry)] = -1
        
        tensor[piece-1,x,y] = 1
        oppose[PIECE_KINDS,rx,ry] = 1
        
        self.deploy_heads[player] += 1
        
        if(y == BOARD_SHAPE[1] and x in GOAL_POS):
            self.deploy_heads[player] += len(GOAL_POS)-1
        
    def deploy_end(self) -> None:
        self._tensor_p1[PIECE_KINDS+1+self._history_len].fill_(1)
        self._tensor_p2[PIECE_KINDS+1+self._history_len].fill_(1)
        
        self.deploy = False
        
    def reset(self) -> None:
        Board.reset(self)
        
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
    
    def tensor_move(self, tensor:torch.Tensor, bef:tuple[int,int], aft:tuple[int,int], piece:int):
        layer:int = -1
        if(piece >= 1 and piece <= PIECE_KINDS): layer = piece-1
        elif(piece == -1): layer = PIECE_KINDS
        else: return
        
        tensor[layer,bef[0],bef[1]] = 0
        tensor[layer,aft[0],aft[1]] = 1
        
    def tensor_erase(self, tensor:torch.Tensor, pos:tuple[int,int], piece:int):
        layer:int = -1
        if(piece >= 1 and piece <= PIECE_KINDS): layer = piece-1
        elif(piece == -1): layer = PIECE_KINDS
        else: return
        
        tensor[layer,pos[0],pos[1]] = 0
        
    def update_history(self, tensor: torch.Tensor, bef: tuple[int, int], aft: tuple[int, int]):
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
        tensor[hist_start, bef[0], bef[1]] = -1.0
        tensor[hist_start, aft[0], aft[1]] = 1.0
    
    #actionに合わせてtensorを変化させる
    #どこに何を動かしたかはself.boardから取得できる
    def step(self, action: int, player: int, erase: GSC.EraseFrag):
        bef,aft = get_action(action)
        o_bef,o_aft = make_reflect_pos_int(bef), make_reflect_pos_int(aft)
        bef_t = change_pos_int_to_tuple(bef)
        aft_t = change_pos_int_to_tuple(aft)
        
        o_bef_t = make_reflect_pos(bef_t)
        o_aft_t = make_reflect_pos(aft_t)
        
        p_idx = player-1
        o_p_idx = 1-p_idx
        
        board = self._boards[p_idx]
        o_board = self._boards[o_p_idx]
        bef_piece = board[bef]
        aft_piece = board[aft]
        o_bef_piece = o_board[o_bef]
        o_aft_piece = o_board[o_aft]
        
        tensor = self.tensors[p_idx]
        o_tensor = self.tensors[o_p_idx]
        
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
            
        self.update_history(tensor,bef_t,aft_t)
        self.update_history(o_tensor,o_bef_t,o_aft_t)
        
        Board.step(self,action,player,erase)
        
    
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
        if(not self.deploy): self.deploy_end()

    def _set_tensor_from_board(self, tensor: torch.Tensor, board_array: np.ndarray):
        """1次元のボード配列からTensorを設定するヘルパー関数"""
        width = BOARD_SHAPE[0]
        
        for i, piece in enumerate(board_array):
            if piece <= 0 and piece != -1: continue # Space(0), Entry(-2), Wall(-100) は無視
            
            x = i % width
            y = i // width
            
            layer = -1
            if 1 <= piece <= PIECE_KINDS:
                layer = piece - 1
            elif piece == -1: # Enemy
                layer = PIECE_KINDS
            
            if layer != -1:
                tensor[layer, x, y] = 1
                
    def get_defined_board(self, pieces: np.ndarray, player: GSC.Player, deploy = False) -> "TensorBoard":
        defined_tensor = TensorBoard(BOARD_SHAPE, self._device, self._history_len)
        
        player_board = self._boards[0] if player == GSC.Player.PLAYER_ONE else self._boards[1]
        oppose_board = self._boards[1] if player == GSC.Player.PLAYER_ONE else self._boards[0]
        
        player_board = player_board.copy()
        defined_board = oppose_board.copy()
        changed_pieces = self.piece_dict[pieces]
        changed_pieces = changed_pieces[:(defined_board > 0).sum()]
        
        defined_board[defined_board > 0] = changed_pieces
        
        defined_tensor.set_board(player_board, defined_board, deploy)
        
        return defined_tensor
        
    def get_board_player1(self) -> torch.Tensor:
        return self._tensor_p1
    
    def get_board_player2(self) -> torch.Tensor:
        return self._tensor_p2
    
    def tensor_board_assert(self) -> bool:
        for v,t in enumerate(self.tensors):
            for y in range(BOARD_SHAPE[1]):
                for x in range(BOARD_SHAPE[0]):
                    piece = self._boards[v][change_pos_tuple_to_int(x,y)]
                    if(piece < 1 or piece > PIECE_KINDS): continue
                    if(t[piece-1,x,y] != 1): return False
        return True
    
    
        
        
        
        
    
    