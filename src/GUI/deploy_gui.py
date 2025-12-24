from src.GUI.Interfaces import IBoardGUI
from src.GUI.init import init

from src.GUI.assets import EndSurface
from src.GUI.const import END_SURFACE_SIZE

from src.common import EraseFrag, Player, change_pos_tuple_to_int, Config
from src.Interfaces import IEnv

import GunjinShogiCore as GSC

import numpy as np
import pygame as pg
from pygame.locals import *

class DeployGUI:
    def __init__(self, boardgui: IBoardGUI, first_piece: np.typing.NDArray[np.int32]):
        self.boardgui = boardgui
        
        self._legal_chg_pos = []
        self._legal_chg_pos_int = []
        self._make_legal_chg_pos()
        
        self.selected_pos = None
        
        self.first_piece = first_piece
        
        self._first_pos_dict = [[-1 for i in range(Config.board_shape[0])] for j in range(Config.board_shape[1])]
        self._make_first_pos_dict()
        
    def _make_first_pos_dict(self):
        pos = 0
        for i in range(len(self.first_piece)):
            x = pos%Config.board_shape[0]
            y = pos//Config.board_shape[0] + Config.entry_height + 1
            self._first_pos_dict[y][x] = i
            
            pos += 1
            if(y == Config.reflect_goal_height and x in Config.goal_pos):
                pos += len(Config.goal_pos)-1
        
    def _make_legal_chg_pos(self):
        for y in range(Config.entry_height+1, Config.board_shape[1]):
            for x in range(Config.board_shape[0]):
                self._legal_chg_pos.append((x,y))
                self._legal_chg_pos_int.append(change_pos_tuple_to_int(x,y))
                if(y == Config.reflect_goal_height and x in Config.goal_pos):
                    x += len(Config.goal_pos)-1
                    
    def set_mouse_pos(self, mouse_pos: tuple[int,int]):
        onboard = self.boardgui.get_selected_pos(mouse_pos)
        self.boardgui.set_emp_pos(onboard)
                    
    def set_clicked_pos(self, mouse_pos):
        onboard = self.boardgui.get_selected_pos(mouse_pos)
        
        isonboard = self.boardgui.set_selected_pos(onboard)
        if isonboard:
            self.boardgui.set_legal_pos(self._legal_chg_pos)
            self.click_board(onboard)
        else: 
            self.boardgui.set_legal_pos([])
            self.selected_pos = None
            
    def click_board(self, pos:tuple[int,int]):
        pos_int = change_pos_tuple_to_int(pos[0],pos[1])
        
        if(pos_int in self._legal_chg_pos_int and self.selected_pos is not None):
            self.swap_pos(self.selected_pos, pos)
            self.boardgui.set_selected_pos((-1,-1))
            self.boardgui.set_legal_pos([])
            self.selected_pos = None
        else:
            self.selected_pos = pos
            
            
    def swap_pos(self, bef:tuple[int,int], aft:tuple[int,int]):
        self.boardgui.swap(bef, aft)
        
        pos1 = self._first_pos_dict[bef[1]][bef[0]]
        pos2 = self._first_pos_dict[aft[1]][aft[0]]
        p1 = self.first_piece[pos1]
        p2 = self.first_piece[pos2]
        self.first_piece[pos1] = p2
        self.first_piece[pos2] = p1
        
    def main_loop(self, screen: pg.Surface) -> np.typing.NDArray[np.int32]:
        app_end = False
        while not app_end:
            pg.display.update()
            
            self.draw(screen)
            
            clicked = False
            for event in pg.event.get():
                if(event.type == QUIT):
                    app_end = True
                    return self.first_piece
                
                if(event.type == MOUSEBUTTONDOWN):
                    if(event.button == 1):
                        clicked = True
                        
            self.set_mouse_pos(pg.mouse.get_pos())
            if(clicked): self.set_clicked_pos(pg.mouse.get_pos())
            
        return self.first_piece
    
    def draw(self, screen: pg.Surface):
        self.boardgui.draw(screen)
            
        