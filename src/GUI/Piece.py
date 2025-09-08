from src.GUI.Interfaces import IPiece
from src.GUI.const import MASS_SIZE
from src.GUI.assets import PieceSurface

from const import Piece

import pygame as pg

class Piece(IPiece):
    def __init__(self, kind:Piece, dir:int = 0, appear = True) -> None:
        self._surface = PieceSurface.IMG_DICT[kind].copy()
        self._surface = pg.transform.rotate(self._surface, dir)
        self._surf_rect = self._surface.get_rect()
        self._rect = pg.Rect((0,0),MASS_SIZE)
        
        self._kind = kind
        
        self._appear: bool = appear
        
        self._null_surface = PieceSurface.PIECE_IMG
        
    def set_location(self, pos:tuple[int,int]) -> None:
        real_pos = (pos[0]*MASS_SIZE[0], pos[1]*MASS_SIZE[1])
        self._rect.topleft = real_pos
        self._surf_rect.center = self._rect.center
        
    def draw(self, surface:pg.Surface):
        blit_surface: pg.Surface
        if(self._appear): blit_surface = self._surface
        else: blit_surface = self._null_surface
        
        surface.blit(blit_surface, self._surf_rect)
        
    def chg_appear(self):
        self._appear = not self._appear
    