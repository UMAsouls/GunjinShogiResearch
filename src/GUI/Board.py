from src.GUI.const import BOARD_SHAPE, MASS_SIZE, WINDOW_SIZE
from src.GUI.assets import BoardSurface

from src.GUI.Interfaces import IPieceGUI, IBoardGUI

import pygame as pg

CENTER = (WINDOW_SIZE[0]//2, WINDOW_SIZE[1]//2)

class BoardGUI(IBoardGUI):
    def __init__(self, board: list[list[IPieceGUI]], center: tuple[int, int] = CENTER):
        self._bg = BoardSurface.BOARD_IMG
        self._rect = self._bg.get_rect()
        
        self._rect.center = center
        
        self._board: list[list[IPieceGUI]] = board
        
        self.emp_pos = (-1,-1)
        self._selected_pos = (-1,-1)
        self._legal_pos = []
        
        self._piece_setup()
        
    def _piece_setup(self):
        for y,i in enumerate(self._board):
            for x,j in enumerate(i):
                if j is not None: j.set_location((x,y), self._rect.topleft)
        
    def draw(self, screen:pg.Surface):
        screen.blit(self._bg, self._rect)
        
        for i in self._board:
            for j in i:
                if j is not None: j.draw(screen)
                
        if(self.is_onboard(self.emp_pos)):
            pos = self.get_screen_pos_from_onboard(self.emp_pos)
            screen.blit(BoardSurface.EMP_IMG, pos)
            
        if(self.is_onboard(self._selected_pos)):
            pos = self.get_screen_pos_from_onboard(self._selected_pos)
            screen.blit(BoardSurface.SELECTED_IMG, pos)
            
        for p in self._legal_pos:
            if(not self.is_onboard(p)): continue
            pos = self.get_screen_pos_from_onboard(p)
            screen.blit(BoardSurface.LEGAL_IMG, pos)
            
    
    def set_emp_pos(self, pos:tuple[int,int]) -> None:
        self.emp_pos = pos
        
    def set_selected_pos(self, pos:tuple[int,int]) -> None:
        self._selected_pos = pos
        
    def set_legal_pos(self, pos:list[tuple[int,int]]) -> None:
        self._legal_pos = pos
                
    def is_onboard(self, onboard_pos: tuple[int,int]) -> bool:
        judge1 = onboard_pos[0] >= 0 and onboard_pos[0] < BOARD_SHAPE[0]
        judge2 = onboard_pos[1] >= 0 and onboard_pos[1] < BOARD_SHAPE[1]
        return judge1 and judge2
    
    def get_screen_pos_from_onboard(self, onboard_pos: tuple[int,int]) -> tuple[int,int]:
        onbg_pos = (onboard_pos[0]*MASS_SIZE[0], onboard_pos[1]*MASS_SIZE[1])
        screen_pos = (onbg_pos[0] + self._rect.left, onbg_pos[1] + self._rect.top)
        
        return screen_pos
            
    def get_selected_pos(self, screen_pos: tuple[int,int]) -> tuple[int,int]:
        onbg_pos = (screen_pos[0]-self._rect.left, screen_pos[1]-self._rect.top)
        onboard_pos = (onbg_pos[0]//MASS_SIZE[0], onbg_pos[1]//MASS_SIZE[1])
        
        return onboard_pos
    
    def get_piece(self, x_idx:int, y_idx:int) -> IPieceGUI:
        return self._board[y_idx][x_idx]
    
    def chg_appear(self) -> None:
        for i in self._board:
            for j in i:
                if j is not None: j.chg_appear()