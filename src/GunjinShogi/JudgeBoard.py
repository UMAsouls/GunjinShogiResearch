from GunjinShogi.Interfaces import IJudgeBoard
from GunjinShogi.const import Piece, EraseFrag, JudgeFrag, JUDGE_TABLE
from GunjinShogi.Board import Board

import torch


class JudgeBoard(Board, IJudgeBoard):
    def __init__(self, size, device):
        device = torch.device("cpu")
        super().__init__(size, device)
        
        self._judge_table_p1 = JUDGE_TABLE
        self._judge_table_p2 = JUDGE_TABLE
        
        self._judge_tables = (self._judge_table_p1, self._judge_table_p2)
        
    def _get_plane_movable(self, move_range: torch.Tensor) -> torch.Tensor:
        moved_mask = \
            move_range[1:] != Piece.Entry & \
            move_range[1:] != Piece.Wall
        movable = torch.where(moved_mask)[0] + 1
        
        return movable
        
    #移動方向の範囲を取得後、どのインデックスに移動できるか判定する関数   
    def get_move_range_movable(self, move_range: torch.Tensor, not_stop: list[Piece]) -> torch.Tensor:
        stop_mask = torch.ones(move_range.shape, dtype=torch.bool)
        for i in not_stop:
            stop_mask = stop_mask & move_range != i
        
        stop_pos = torch.argmax(stop_mask)
        
        stop_pos[torch.where(move_range[stop_pos] == Piece.Enemy)[0]] += 1
        
        moved_mask = \
            move_range[1:stop_pos] != Piece.Entry & \
            move_range[1:stop_pos] != Piece.Wall
        movable = torch.where(moved_mask)[0] + 1
        
        return movable
    
    #特殊移動駒の左右移動判定
    def get_rl_action(self, player_board: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        right = pos+1
        left = pos-1
        
        right_movable = torch.where(player_board[right] == Piece.Space)[0]
        left_movable = torch.where(player_board[left] == Piece.Space)[0]
        
        right_action = pos[right_movable] * self._s + right[right_movable]
        left_aciton = pos[left_movable] * self._s + left[left_movable]
        
        return right_action, left_aciton
    
    def get_plane_move(self, player_board: torch.Tensor, pos:torch.Tensor) -> torch.Tensor:
        """ヒコーキの動き取得

        Args:
            player_board (torch.Tensor): アクションを起こす側の盤面

        Returns:
            torch.Tensor: ヒコーキのアクション
        """
        width = self._size[0]
        
        up_list = player_board[:pos+1:width].flip()
        down_list = player_board[pos::width]
        
        up_movable = self._get_plane_movable(up_list)
        down_movable = self._get_plane_movable(down_list)
        
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
        
        up_list = player_board[:pos+1:width].flip()
        down_list = player_board[pos::width]
        
        not_stop = [Piece.Space, Piece.Entry]
        up_movable = self.get_move_range_movable(up_list,not_stop)
        down_movable = self.get_move_range_movable(down_list,not_stop)
        
        up_action = pos * self._s + (pos - up_movable*width)
        down_action = pos * self._s + (pos + down_movable*width)
        
        up_action = up_action[:2]
        down_action = down_action[:1]
        
        right_action, left_action = self.get_rl_action(player_board, pos)
        
        return torch.cat((up_action,down_action,right_action, left_action))
    
    def get_engineer_move(self, player_board: torch.Tensor, pos:torch.Tensor) -> torch.Tensor:
        width = self._size[0]
        
        x,y = pos%width, pos//width
        
        up_list = player_board[:pos+1:width].flip()
        down_list = player_board[pos::width]
        right_list = player_board[pos:pos+(width-x)]
        left_list = player_board[y*width:pos].flip()
        
        not_stop = [Piece.Space, Piece.Entry]
        up_movable = self.get_move_range_movable(up_list,not_stop)
        down_movable = self.get_move_range_movable(down_list,not_stop)
        right_movable = self.get_move_range_movable(right_list,not_stop)
        left_movable = self.get_move_range_movable(left_list,not_stop)
        
        up_action = pos * self._s + (pos - up_movable*width)
        down_action = pos * self._s + (pos + down_movable*width)
        right_action = pos * self._s + (pos + right_movable)
        left_action = pos * self._s + (pos - left_movable)
        
        return torch.cat((up_action,down_action,right_action, left_action))
        
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
        
         # 上への移動 (aft = bef - width)
        # あるマス(aft)の下(bef)に駒があるか？
        rolled = torch.roll(piece_mask, shifts=-width)
        valid_moves = rolled & target_mask
        if valid_moves.any():
            aft_indices = torch.nonzero(valid_moves).squeeze(1)
            bef_indices = aft_indices + width
            all_legal_actions.append(bef_indices * s + aft_indices)

        # 下への移動 (aft = bef + width)
        # あるマス(aft)の上(bef)に駒があるか？
        rolled = torch.roll(piece_mask, shifts=width)
        valid_moves = rolled & target_mask
        if valid_moves.any():
            aft_indices = torch.nonzero(valid_moves).squeeze(1)
            bef_indices = aft_indices - width
            all_legal_actions.append(bef_indices * s + aft_indices)
            
        # 左への移動 (aft = bef - 1)
        # あるマス(aft)の右(bef)に駒があるか？
        rolled = torch.roll(piece_mask, shifts=-1)
        valid_moves = rolled & target_mask
        valid_moves[width-1::width] = False # 右端のマスは左から移動してこれない(端処理)
        if valid_moves.any():
            aft_indices = torch.nonzero(valid_moves).squeeze(1)
            bef_indices = aft_indices + 1
            all_legal_actions.append(bef_indices * s + aft_indices)

        # 右への移動 (aft = bef + 1)
        # あるマス(aft)の左(bef)に駒があるか？
        rolled = torch.roll(piece_mask, shifts=1)
        valid_moves = rolled & target_mask
        valid_moves[::width] = False # 左端のマスは右から移動してこれない(端処理)
        if valid_moves.any():
            aft_indices = torch.nonzero(valid_moves).squeeze(1)
            bef_indices = aft_indices - 1
            all_legal_actions.append(bef_indices * s + aft_indices)
        
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
        
                
        