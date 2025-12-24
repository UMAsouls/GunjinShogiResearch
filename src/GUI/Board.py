from src.common import Config

from src.GUI.const import MASS_SIZE, WINDOW_SIZE
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
                
    def emphasize_mass(self, screen: pg.Surface, emp_surface: pg.Surface, pos:tuple[int,int]) -> bool:
        if(not self.is_onboard(pos)): return False
        
        p = self.get_screen_pos_from_onboard(pos)
        screen.blit(emp_surface, p)
        
        return True
        
    def draw(self, screen:pg.Surface):
        screen.blit(self._bg, self._rect)
        
        for i in self._board:
            for j in i:
                if j is not None: j.draw(screen)
        
        #合法手強調    
        for p in self._legal_pos:
            self.emphasize_mass(screen, BoardSurface.LEGAL_IMG, p)
            
        #選択中のマス強調
        self.emphasize_mass(screen, BoardSurface.SELECTED_IMG, self._selected_pos)
            
        #マウスに重なってるマス強調
        self.emphasize_mass(screen, BoardSurface.EMP_IMG, self.emp_pos)
            
    
    def set_emp_pos(self, pos:tuple[int,int]) -> bool:
        self.emp_pos = pos
        return self.is_onboard(self.emp_pos)
        
    def set_selected_pos(self, pos:tuple[int,int]) -> bool:
        self._selected_pos = pos
        return self.is_onboard(self._selected_pos)
        
    def set_legal_pos(self, pos:list[tuple[int,int]]) -> None:
        self._legal_pos = pos
                
    def is_onboard(self, onboard_pos: tuple[int,int]) -> bool:
        judge1 = onboard_pos[0] >= 0 and onboard_pos[0] < Config.board_shape[0]
        judge2 = onboard_pos[1] >= 0 and onboard_pos[1] < Config.board_shape[1]
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
                
    def move(self, bef:tuple[int,int], aft:tuple[int,int]) -> bool:
        if(not self.is_onboard(bef) or not self.is_onboard(aft)):
            return False
        
        piece = self._board[bef[1]][bef[0]]
        self._board[aft[1]][aft[0]] = piece
        self._board[bef[1]][bef[0]] = None
        
        piece.set_location(aft, self._rect.topleft)
        
        return True
        
        
    def erase(self, pos:tuple[int,int]) -> bool:
        if(not self.is_onboard(pos)):
            return False
        
        self._board[pos[1]][pos[0]] = None
        
        return True
    
    def swap(self, bef:tuple[int,int], aft:tuple[int,int]) -> bool:
        if(not self.is_onboard(bef) or not self.is_onboard(aft)):
            return False
        
        piece1 = self._board[bef[1]][bef[0]]
        piece2 = self._board[aft[1]][aft[0]]
        self._board[bef[1]][bef[0]] = piece2
        self._board[aft[1]][aft[0]] = piece1
        
        piece1.set_location(aft, self._rect.topleft)
        piece2.set_location(bef, self._rect.topleft)
        
        return True