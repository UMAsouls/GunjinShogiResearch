from src.const import BOARD_SHAPE, PIECE_KINDS, ENTRY_HEIGHT, ENTRY_POS, GOAL_POS, PIECE_DICT, Piece
from src.GunjinShogi.Interfaces import ITensorBoard

from src.GunjinShogi.Board import Board
from src.GunjinShogi.const import JUDGE_TABLE

from src.common import EraseFrag, get_action, change_pos_tuple_to_int, change_pos_int_to_tuple, make_reflect_pos, make_reflect_pos_int

import numpy as np
import torch

import GunjinShogiCore as GSC

WALL_ENTRY_GOAL_CHANNEL = 3

ENEMY_INFO_CHANNEL = PIECE_KINDS

#Tensorの設定おかしいです
class TensorBoard(Board,ITensorBoard):
    @classmethod
    def get_tensor_channels(cls, history):
        return PIECE_KINDS + ENEMY_INFO_CHANNEL  + WALL_ENTRY_GOAL_CHANNEL + 1 + history
    
    
    def __init__(self, size: tuple[int, int], device: torch.device, history = 30):
        super().__init__(size)
        
        self._device = device
        
        self._history_len = history
        
        # tensor: 自分の駒(16種) + 敵駒(1種) + 履歴(history)
        # channel 0-15: Piece 1-16 (自分の駒)
        # channel 16: Enemy (敵駒: -1)
        # channel 17 ~ 17+history-1: History
        self._base_channels = PIECE_KINDS + ENEMY_INFO_CHANNEL  + WALL_ENTRY_GOAL_CHANNEL + 1
        self._total_channels = self._base_channels + self._history_len
        
        self._piece_channels = PIECE_KINDS
        
        self.reset()
        
        self.piece_dict = np.array(PIECE_DICT)
        
        self.judge_table = torch.from_numpy(JUDGE_TABLE).clone().to(self._device)
        
        
    def lose_private_set(self, pos_private_presition: torch.Tensor, play_piece:int):
        presition = (self.judge_table[play_piece] < 0)[1:PIECE_KINDS+1]
        presition[PIECE_KINDS-1] = True
        
        no_presition = presition == False
        pos_private_presition[no_presition] = 0
        
        presition_set = pos_private_presition > 0
        pos_private_presition = torch.where(presition_set, 1/presition.sum(), 0)
        
        return pos_private_presition
    
    def win_private_set(self, private_dead: torch.Tensor, pos_private_presition: torch.Tensor, play_piece:int):
        presition = (self.judge_table[play_piece] > 0)[1:PIECE_KINDS+1]
        presition[pos_private_presition == 0] = False
        
        dead_rate = torch.where(presition, 1/presition.sum(), 0)
        private_dead += dead_rate
        
        return private_dead
        
    def draw_private_set(self, private_dead: torch.Tensor, pos_private_presition: torch.Tensor, play_piece:int):
        presition = (self.judge_table[play_piece] == 0)[1:PIECE_KINDS+1]
        presition[pos_private_presition == 0] = False
        
        dead_rate = torch.where(presition, 1/presition.sum(), 0)
        private_dead += dead_rate
        
        return private_dead

    def is_piece_between(self, bef:int, aft:int, board:np.ndarray):
        width = BOARD_SHAPE[0]
        if(abs(bef-aft) <= width):
            return False
        p1 = min(bef,aft)
        p2 = max(bef,aft)
        
        r = board[p1:p2:width]
        if((r[1:] == 0).all()):
            return False
        return True
        
    def info_set(
                    self, bef:tuple[int,int], aft:tuple[int,int], piece:int, win: int,
                    tensor:torch.Tensor, board:np.ndarray,
                    private_dead: torch.Tensor, private_presition: torch.Tensor
                ):
        
        private_presition[Piece.LandMine-1, bef[0], bef[1]] = 0
        private_presition[Piece.Frag-1, bef[0], bef[1]] = 0
        
        bef_x,bef_y = change_pos_int_to_tuple(bef)
        aft_x,aft_y = change_pos_int_to_tuple(aft)
        
        mask = torch.ones(private_presition.shape[0])
        mask[[Piece.Engineer-1, Piece.Plane-1]] = 0
        if(abs(aft_y - bef_y) >= 2):
            private_presition[mask] = 0
        
        mask[[Piece.Tank-1, Piece.Plane-1,Piece.Engineer-1]] = 0
        if(aft_y - bef_y == 2):
            private_presition[mask] = 0
        
        mask[[Piece.Tank-1, Piece.Plane-1,Piece.Engineer-1]] = 1
        mask[[Piece.Engineer-1]] = 0
        if(abs(aft_x - bef_x) >= 2):
            private_presition[mask] = 0
            
        mask[[Piece.Tank-1, Piece.Plane-1,Piece.Engineer-1]] = 1
        mask[[Piece.Plane-1]] = 0
        if(self.is_piece_between(bef, aft, board)):
            private_presition[mask] = 0
        
        
        if(win == -1):
            private_presition[:, aft[0], aft[1]] = self.lose_private_set(private_presition[:, aft[0], aft[1]], piece)
            private_presition[:, bef[0], bef[1]] = torch.zeros(PIECE_KINDS, dtype=torch.float32, device=self._device)
        
        elif(win == 1):
            private_dead = self.win_private_set(private_dead, private_presition[:, aft[0], aft[1]], piece)
            private_presition[:, aft[0], aft[1]] = torch.zeros(PIECE_KINDS, dtype=torch.float32, device=self._device)
            private_presition[:, bef[0], bef[1]] = torch.zeros(PIECE_KINDS, dtype=torch.float32, device=self._device)
            
        else:
            private_dead = self.draw_private_set(private_dead, private_presition[:, aft[0], aft[1]], piece)
            private_presition[:, aft[0], aft[1]] = torch.zeros(PIECE_KINDS, dtype=torch.float32, device=self._device)
            private_presition[:, bef[0], bef[1]] = torch.zeros(PIECE_KINDS, dtype=torch.float32, device=self._device)
        

        dead_count = self.global_pool - private_dead
        enemy_info = private_presition*dead_count.unsqueeze(1).unsqueeze(2)
        tensor[self._piece_channels:self._piece_channels+PIECE_KINDS] = enemy_info/(enemy_info.sum(dim = 0)+1e-8).unsqueeze(0)
        
        if(tensor.isnan().sum() > 0):
            print("nan detect") 
        
        return private_dead, private_presition, tensor
        
        
    @property
    def total_channels(self) -> int:
        return self._total_channels
        
    def wall_set(self, tensor: torch.Tensor, board: np.ndarray):
        wall_channel = self._piece_channels + ENEMY_INFO_CHANNEL
        entry_channel = wall_channel + 1
        goal_channel = entry_channel + 1
        
        for x in range(BOARD_SHAPE[0]):
            if(x in ENTRY_POS): 
                tensor[entry_channel,x,ENTRY_HEIGHT] = 1
                board[change_pos_tuple_to_int(x,ENTRY_HEIGHT)] = Piece.Entry #Entry(-2)
            else: 
                tensor[wall_channel,x,ENTRY_HEIGHT] = 1
                board[change_pos_tuple_to_int(x,ENTRY_HEIGHT)] = Piece.Wall #Wall
                
            if(x in GOAL_POS):
                tensor[goal_channel,x,0] = 1
    
    def deploy_set(self, piece, player:GSC.Player):
        
        i = 0 if player == GSC.Player.PLAYER_ONE else 1
        mi = i - 1
        
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
        tensor[self._piece_channels:self._piece_channels+ENEMY_INFO_CHANNEL, x, y] = 1/PIECE_KINDS
        oppose[self._piece_channels:self._piece_channels+ENEMY_INFO_CHANNEL,rx,ry] = 1/PIECE_KINDS
        
        self.oppose_presition_pool[i, :, x,y] = 1/PIECE_KINDS
        self.private_presition_pool[mi, :, rx,ry] = 1/PIECE_KINDS
        
        if(player == GSC.Player.PLAYER_ONE): self.global_pool[piece-1] += 1
        
        self.deploy_heads[player] += 1
        
        if(y == BOARD_SHAPE[1] and x in GOAL_POS):
            self.deploy_heads[player] += len(GOAL_POS)-1
        
    def deploy_end(self) -> None:
        self._tensor_p1[self._base_channels-1].fill_(1)
        self._tensor_p2[self._base_channels-1].fill_(1)
        
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
        
        win = 1 if erase == GSC.EraseFrag.AFT else -1
        o_win = -win
        self.info_set(bef_t, aft_t, bef_piece, win, tensor, board, self.private_dead_pool[p_idx], self.private_presition_pool[p_idx])
        self.info_set(o_bef_t, o_aft_t, o_bef_piece, o_win, o_tensor, o_board, self.private_dead_pool[o_p_idx], self.private_presition_pool[o_p_idx])
        
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
    
    
        
        
        
        
    
    