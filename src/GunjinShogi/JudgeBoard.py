from src.GunjinShogi.Interfaces import IJudgeBoard
from src.GunjinShogi.const import JudgeFrag, JUDGE_TABLE
from src.GunjinShogi.Board import Board

from src.const import Piece,PIECE_KINDS, GOAL_POS, ENTRY_HEIGHT, ENTRY_POS, BOARD_SHAPE
from src.common import EraseFrag

import torch

ENTRY_POS_INTS = torch.tensor([ENTRY_HEIGHT*BOARD_SHAPE[0] + i for i in ENTRY_POS])

class JudgeBoard(Board, IJudgeBoard):
    def __init__(self, size):
        device = torch.device("cpu")
        super().__init__(size, device)
        
        self._judge_table_p1 = JUDGE_TABLE
        self._judge_table_p2 = JUDGE_TABLE
        
        self._judge_tables = (self._judge_table_p1, self._judge_table_p2)
        
    def judge_win(self,value: int) -> bool:
        return (value == Piece.General) or \
            (value == Piece.LieutenantGeneral) or \
            (value == Piece.MajorGeneral) or \
            (value == Piece.Colonel) or \
            (value == Piece.LieutenantColonel) or \
            (value == Piece.Major)
        
    def is_win(self, player) -> bool:
        player_board, oppose_board = self.get_plyaer_opponent_board(player)
        
        for i in GOAL_POS:
            v = player_board[i]
            if(self.judge_win(v.item())): return True
            
        return False
        
    def _get_plane_movable(self, move_range: torch.Tensor) -> torch.Tensor:
        moved_mask = \
            (move_range == Piece.Space) | \
            (move_range == Piece.Enemy)
            
        movable = torch.where(moved_mask)[0]
        
        return movable
        
    #移動方向の範囲を取得後、どのインデックスに移動できるか判定する関数   
    def get_move_range_movable(self, move_range: torch.Tensor, not_stop: list[Piece]) -> torch.Tensor:
        stop_mask = torch.ones(move_range.shape, dtype=torch.bool)
        for i in not_stop:
            stop_mask = stop_mask & (move_range != i)
        
        if(stop_mask.any()):
            stop_pos = torch.where(stop_mask)[0][0]
            en_pos = torch.where(move_range[stop_pos] == Piece.Enemy)[0]
            if(en_pos.numel() > 0): stop_pos += 1
        else: 
            stop_pos = move_range.numel()
        
        
        
        moved_mask = \
            (move_range[:stop_pos] != Piece.Entry) & \
            (move_range[:stop_pos] != Piece.Wall) | \
            (move_range[:stop_pos] == Piece.Space) | \
            (move_range[:stop_pos] == Piece.Enemy)
        movable = torch.where(moved_mask)[0]
        
        return movable
    
    #特殊移動駒の左右移動判定
    def get_rl_action(self, player_board: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        right = pos+1
        left = pos-1
        
        right_movable = \
            (player_board[right] == Piece.Space) | \
            (player_board[right] == Piece.Enemy)
        left_movable = \
            (player_board[left] == Piece.Space) | \
            (player_board[left] == Piece.Enemy)
        
        width,_ = self._size
        
        right_action: torch.Tensor
        if(right_movable and pos % width != width-1): right_action = torch.tensor([pos * self._s + right])
        else: right_action = torch.as_tensor([])
        
        left_action: torch.Tensor
        if(left_movable and pos % width != 0): left_action = torch.tensor([pos * self._s + left])
        else: left_action = torch.as_tensor([])
        
        return right_action, left_action
    
    def get_plane_move(self, player_board: torch.Tensor, pos:torch.Tensor) -> torch.Tensor:
        """ヒコーキの動き取得

        Args:
            player_board (torch.Tensor): アクションを起こす側の盤面

        Returns:
            torch.Tensor: ヒコーキのアクション
        """
        width = self._size[0]
        x,y = pos%BOARD_SHAPE[0], pos//BOARD_SHAPE[1]
        
        up_list = player_board[x:pos:width].flip(dims=[0])
        down_list = player_board[pos+width::width]
        
        up_movable = self._get_plane_movable(up_list) + 1
        down_movable = self._get_plane_movable(down_list) + 1
        
        up_action = pos * self._s + (pos - up_movable*width)
        down_action = pos * self._s + (pos + down_movable*width)
        
        right_action, left_action = self.get_rl_action(player_board, pos)
        
        return torch.cat((up_action,down_action,right_action, left_action))
    
    def get_tank_cavalry_move(self, player_board: torch.Tensor, pos:torch.Tensor) -> torch.Tensor:
        """タンクと騎兵のアクション取得

        Args:
            player_board (torch.Tensor): プレイヤーの盤面

        Returns:
            torch.Tensor: タンクと騎兵アクション
        """
        width = self._size[0]
        
        x,y = pos%BOARD_SHAPE[0], pos//BOARD_SHAPE[1]
        
        up_list = player_board[x:pos:width].flip(dims=[0])
        down_list = player_board[pos+width::width]
        
        not_stop = [Piece.Space, Piece.Entry]
        up_movable = self.get_move_range_movable(up_list,not_stop) + 1
        down_movable = self.get_move_range_movable(down_list,not_stop) + 1
        
        up_action = pos * self._s + (pos - up_movable*width)
        down_action = pos * self._s + (pos + down_movable*width)
        
        up_action = up_action[:2]
        down_action = down_action[:1]
        
        right_action, left_action = self.get_rl_action(player_board, pos)
        
        return torch.cat((up_action,down_action,right_action, left_action))
    
    def get_engineer_move(self, player_board: torch.Tensor, pos:torch.Tensor) -> torch.Tensor:
        width = self._size[0]
        
        x,y = pos%width, pos//width
        
        up_list = player_board[x:pos:width].flip(dims=[0])
        down_list = player_board[pos+width::width]
        right_list = player_board[pos+1:pos+(width-x)]
        left_list = player_board[y*width:pos].flip(dims=[0])
        
        not_stop = [Piece.Space, Piece.Entry]
        up_movable = self.get_move_range_movable(up_list,not_stop) + 1
        down_movable = self.get_move_range_movable(down_list,not_stop) + 1
        right_movable = self.get_move_range_movable(right_list,not_stop) + 1
        left_movable = self.get_move_range_movable(left_list,not_stop) + 1
        
        up_action = pos * self._s + (pos - up_movable*width)
        down_action = pos * self._s + (pos + down_movable*width)
        right_action = pos * self._s + (pos + right_movable)
        left_action = pos * self._s + (pos - left_movable)
        
        return torch.cat((up_action,down_action,right_action, left_action))
    
    def make_normal_piece_action(self,piece_mask: torch.Tensor, target_mask: torch.Tensor, dir: tuple[int,int]) -> torch.Tensor:
        width,height = self._size
        
        move = dir[1]*width + dir[0]
        
        rolled = torch.roll(piece_mask, shifts = move)
        valid_moves = rolled & target_mask
        
        if not valid_moves.any(): return torch.tensor([])
        
        if(dir[0] < 0): valid_moves[width-1::width] = False # 右端のマスは左から移動してこれない(端処理)
        if(dir[0] > 0): valid_moves[::width] = False # 左端のマスは右から移動してこれない(端処理)
        if(dir[1] > 0): valid_moves[:width] = False # 上端のマスは下から移動してこれない
        if(dir[1] < 0): valid_moves[self._s-(width-1):] = False # 下端のマスは上から移動してこれない
        
        aft_indices = torch.nonzero(valid_moves).squeeze(1)
        bef_indices = aft_indices - move
        return bef_indices * self._s + aft_indices
    
    def make_entry_action(self, piece_mask: torch.Tensor) -> torch.Tensor:
        width, _ = self._size
        
        bef_pos = (ENTRY_POS_INTS + torch.tensor((width,-width))[:, torch.newaxis]).flatten()
        aft_pos = (ENTRY_POS_INTS + torch.tensor((-width,width))[:, torch.newaxis]).flatten()
        
        entry_mask_bef = torch.zeros(piece_mask.shape, dtype=torch.bool)
        entry_mask_bef[bef_pos] = True
        
        entry_mask_aft = torch.zeros(piece_mask.shape, dtype=torch.bool)
        entry_mask_aft[aft_pos] = True
        
        valid_entry_bef = piece_mask & entry_mask_bef
        valid_entry_aft = -1*(piece_mask & entry_mask_aft) + 1
        entry_piece_indices = torch.where(valid_entry_bef[bef_pos] & valid_entry_aft[aft_pos])[0]
        
        entry_actions = (bef_pos[entry_piece_indices]) * self._s + (aft_pos[entry_piece_indices])
        return entry_actions
            
        
    def legal_move(self, player: int) -> torch.Tensor:
        """実行可能アクションの取得

        Args:
            player (int): アクションを実行する側

        Returns:
            torch.Tensor: 実行可能アクション
        """
        player_board, _ = self.get_plyaer_opponent_board(player)
        
        #動かない駒と特殊な動き方の駒を除外
        piece_mask = \
            (player_board > int(Piece.Space)) & \
            (player_board != Piece.LandMine) & \
            (player_board != Piece.Frag) & \
            (player_board != Piece.Plane) & \
            (player_board != Piece.Tank) & \
            (player_board != Piece.Cavalry) & \
            (player_board != Piece.Engineer)
        
        target_mask = (player_board == Piece.Space) | (player_board == Piece.Enemy)
        
        width,height = self._size
        s = self._s
        
        all_legal_actions = []
        
        #突入口付近の合法手生成
        all_legal_actions.append(self.make_entry_action(piece_mask))
        
        # 上への移動 (aft = bef - width)
        # あるマス(aft)の下(bef)に駒があるか？
        all_legal_actions.append(self.make_normal_piece_action(piece_mask, target_mask, [0,-1]))

        # 下への移動 (aft = bef + width)
        # あるマス(aft)の上(bef)に駒があるか？
        all_legal_actions.append(self.make_normal_piece_action(piece_mask, target_mask, [0,1]))
            
        # 左への移動 (aft = bef - 1)
        # あるマス(aft)の右(bef)に駒があるか？
        all_legal_actions.append(self.make_normal_piece_action(piece_mask, target_mask, [-1,0]))

        # 右への移動 (aft = bef + 1)
        # あるマス(aft)の左(bef)に駒があるか？
        all_legal_actions.append(self.make_normal_piece_action(piece_mask, target_mask, [1,0]))
        
        # 特殊駒の移動判定   
        special_mask = \
            (player_board == Piece.Plane) |\
            (player_board == Piece.Tank) |\
            (player_board == Piece.Cavalry) |\
            (player_board == Piece.Engineer)
            
        special_pos = torch.where(special_mask)[0]
        
        for pos in special_pos:
            if(player_board[pos] == Piece.Plane):
                all_legal_actions.append(self.get_plane_move(player_board,pos))
            elif(player_board[pos] == Piece.Engineer):
                all_legal_actions.append(self.get_engineer_move(player_board,pos))
            else:
                all_legal_actions.append(self.get_tank_cavalry_move(player_board,pos))

        if not all_legal_actions:
            return torch.tensor([], dtype=torch.long, device=self._device)
            
        return torch.cat(all_legal_actions)

        
    def judge(self, action: int, player: int):
        bef,aft = self.get_action(action)
        o_bef, o_aft = self.get_opponent_action(bef, aft)
        player_board, oppose_board = self.get_plyaer_opponent_board(player)
        
        player_table = self._judge_tables[player-1]
        
        p1 = player_board[bef]
        p2 = oppose_board[o_aft]
        if(p2 == int(Piece.Space)): return EraseFrag.NO 
        
        if(player_table[p1][p2] == JudgeFrag.Win): return EraseFrag.AFTER
        elif(player_table[p1][p2] == JudgeFrag.Lose): return EraseFrag.BEFORE
        else: return EraseFrag.BOTH
        
    def get_piece_effect_by_action(self, action: int, player: int):
        bef,aft = self.get_action(action)
        o_bef, o_aft = self.get_opponent_action(bef, aft)
        player_board, oppose_board = self.get_plyaer_opponent_board(player)
        
        return (player_board[bef].item(), oppose_board[o_aft].item())
    
    def set_board(self, board_player1, board_player2):
        super().set_board(board_player1, board_player2)
        
        #軍旗の強さを決める
        width, _ = self._size
        for i, b in enumerate(self._boards):
            frag_pos = torch.where(b == Piece.Frag)[0]
            if frag_pos < self._s - width:
                back_piece_kind = b[frag_pos + width]
                self._judge_tables[i][int(Piece.Frag)] = self._judge_tables[i][back_piece_kind]
                self._judge_tables[1-i][:,int(Piece.Frag)] = self._judge_tables[i][:,back_piece_kind]
                
                
                
        