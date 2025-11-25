from src.const import BOARD_SHAPE, PIECE_KINDS
from src.GunjinShogi.Interfaces import ITensorBoard

from src.GunjinShogi.Board import Board

from src.common import EraseFrag, get_action, change_pos_int_to_tuple, make_reflect_pos, make_reflect_pos_int

import numpy as np
import torch

import GunjinShogiCore as GSC

class TensorBoard(Board,ITensorBoard):
    def __init__(self, size: tuple[int, int], device: torch.device, history = 30):
        super().__init__(size)
        
        self._device = device
        
        #tensor:自分の駒の位置+敵駒+履歴(論文と同じもの×30)
        self._tensor_p1 = torch.zeros([PIECE_KINDS + 1 + history,BOARD_SHAPE[0],BOARD_SHAPE[1]])
        self._tensor_p2 = torch.zeros([PIECE_KINDS + 1 + history,BOARD_SHAPE[0],BOARD_SHAPE[1]])
        
        self.tensors = [self._tensor_p1, self._tensor_p2]

        self.first_p1: np.typing.NDArray[np.int32] = np.zeros(22)
        self.first_p2: np.typing.NDArray[np.int32] = np.zeros(22)
        
    def reset(self) -> None:
        pass
    
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
    
    #actionに合わせてtensorを変化させる
    #どこに何を動かしたかはself.boardから取得できる
    def step(self, action: int, player: int, erase: GSC.EraseFrag):
        bef,aft = get_action(action)
        o_bef,o_aft = make_reflect_pos_int(bef), make_reflect_pos_int(aft)
        bef_t = change_pos_int_to_tuple(bef)
        aft_t = change_pos_int_to_tuple(aft)
        
        o_bef_t = make_reflect_pos(bef_t)
        o_aft_t = make_reflect_pos(aft_t)
        
        board = self._boards[player]
        o_board = self._boards[3-player]
        bef_piece = board[bef]
        aft_piece = board[aft]
        o_bef_piece = o_board[o_bef]
        o_aft_piece = o_board[o_aft]
        
        tensor = self.tensors[player]
        o_tensor = self.tensors[3-player]
        
        if(erase == EraseFrag.BEFORE):
            self.tensor_erase(tensor, bef_t, bef_piece)
            self.tensor_erase(o_tensor, o_bef_t, o_bef_piece)
        elif(erase == EraseFrag.AFTER):
            self.tensor_erase(tensor, aft_t, aft_piece)
            self.tensor_erase(o_tensor, o_aft_t, o_aft_piece)
        elif(erase == EraseFrag.BOTH):
            self.tensor_erase(tensor, bef_t, bef_piece)
            self.tensor_erase(o_tensor, o_bef_t, o_bef_piece)
            self.tensor_erase(tensor, aft_t, aft_piece)
            self.tensor_erase(o_tensor, o_aft_t, o_aft_piece)
        
        if(erase != EraseFrag.BEFORE):
            self.tensor_move(tensor, bef_t, aft_t, bef_piece)
            self.tensor_move(o_tensor, o_bef_t, o_aft_t, o_bef_piece)
        
        super().step(action,player,erase)
        
    
    def undo(self) -> bool:
        pass
    
    def set_board(self, board_player1: np.ndarray, board_player2: np.ndarray) -> None:
        super().set_board(board_player1,board_player2)
        
    def get_board_player1(self) -> torch.Tensor:
        return self._tensor_p1
    
    def get_board_player2(self) -> torch.Tensor:
        return self._tensor_p2
    
        
        
        
        
    
    