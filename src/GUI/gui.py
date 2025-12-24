from sympy import cofactors
from src.GUI.Interfaces import IBoardGUI
from src.GUI.init import init

from src.GUI.assets import EndSurface
from src.GUI.const import END_SURFACE_SIZE

from src.common import make_action,get_action, EraseFrag, Player, LogData, Config
from src.Interfaces import IEnv

import GunjinShogiCore as GSC

import pygame as pg
from pygame.locals import *

#Envは現在の手番側から見た選択可能な行動を出力する
#そのため、gui側から変換が必須
def make_reflect_pos(pos:tuple[int,int]) -> tuple[int,int]:
    return (Config.board_shape[0] - pos[0]-1, Config.board_shape[1] - pos[1]-1)

def make_reflect_pos_int(pos_int:int) -> int:
    return Config.board_shape_int - (pos_int+1)

class GUI:
    def __init__(self, boardgui: IBoardGUI, env: IEnv):
        self._boardgui = boardgui
        
        #移動可能判定用リスト
        self.judge_legal_pos_list: list[list[int]] = \
            [[] for _ in range(Config.board_shape[0]*Config.board_shape[1])]
        self.judge_legal_pos: list[int] = []
        
        #board_guiに渡す用のリスト
        self.legal_pos: list[list[list[tuple[int,int]]]] = \
            [[[] for _ in range(Config.board_shape[0])] for _ in range(Config.board_shape[1])]
            
        self._selected_pos: int = -1
        
        self._env = env
        
        self.done: bool = False
        
        self.winner: Player = None
        
        self.set_legal_move()
        
    def _legal_pos_reset(self):
        self.judge_legal_pos_list: list[list[int]] = \
            [[] for _ in range(Config.board_shape_int)]
        self.judge_legal_pos: list[int] = []
        
        self.legal_pos: list[list[list[tuple[int,int]]]] = \
            [[[] for _ in range(Config.board_shape[0])] for _ in range(Config.board_shape[1])]
    
    
    def set_legal_move(self) -> None:
        legal_tensor = self._env.legal_move()
        self._legal_pos_reset()
        
        for i in legal_tensor:
            act = int(i.item())
            i_bef,i_aft = get_action(act)
            if(self._env.get_current_player() == GSC.Player.PLAYER_TWO):
                i_bef,i_aft = make_reflect_pos_int(i_bef),make_reflect_pos_int(i_aft)
            
            self.judge_legal_pos_list[i_bef].append(i_aft)

            width = Config.board_shape[0]
            
            i_bef_pos = (i_bef%width, i_bef//width)
            i_aft_pos = (i_aft%width, i_aft//width)
            self.legal_pos[i_bef_pos[1]][i_bef_pos[0]].append(i_aft_pos)
    
    def action(self, bef: int, aft: int) -> LogData:        
        aciton: int
        if(self._env.get_current_player() == GSC.Player.PLAYER_TWO):
            b,a = make_reflect_pos_int(bef),make_reflect_pos_int(aft)
            action = make_action(b,a)
        else:
            action = make_action(bef,aft)
        
        _, log, frag = self._env.step(action)
        
        if(frag != GSC.BattleEndFrag.CONTINUE): self.done = True

        width = Config.board_shape[0]
        
        bef_pos = (bef%width, bef//width)
        aft_pos = (aft%width, aft//width)
        
        if(log.erase == EraseFrag.AFTER or log.erase == EraseFrag.NO):
            self._boardgui.move(bef_pos, aft_pos)
        elif(log.erase == EraseFrag.BEFORE):
            self._boardgui.erase(bef_pos)
        else:
            self._boardgui.erase(bef_pos)
            self._boardgui.erase(aft_pos)
            
        self.set_legal_move()
        self._boardgui.chg_appear() 
        
        return log
        
    def draw(self, screen: pg.Surface) -> None:
        self._boardgui.draw(screen)
        
        if(self.done): 
            rect = pg.Rect((0,0),END_SURFACE_SIZE)
            rect.center = screen.get_rect().center
            if(self._env.get_winner() == GSC.Player.PLAYER_ONE):
                screen.blit(EndSurface.PLAYER1_WIN_SURFACE, rect)
            else:
                screen.blit(EndSurface.PLAYER2_WIN_SURFACE, rect)
                

    
        
        
    
    