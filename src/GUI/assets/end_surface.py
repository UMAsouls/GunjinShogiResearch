from src.GUI.const import \
    BOARD_SIZE, BOARD_SHAPE, MASS_SIZE, END_SURFACE_SIZE, BOARD_COLOR, WIN_SURFACE_COLOR
    
from src.const import GOAL_POS, ENTRY_HEIGHT, ENTRY_POS

import pygame as pg

class EndSurface:
    PLAYER1_WIN_SURFACE: pg.Surface
    PLAYER2_WIN_SURFACE: pg.Surface
    
    FONT: pg.font.Font
    
    @classmethod
    def make_win_surface(cls, text:str) -> pg.Surface:
        s = pg.Surface(END_SURFACE_SIZE)
        s.fill(WIN_SURFACE_COLOR)
        rect = s.get_rect()
        
        text_surface = cls.FONT.render(text, antialias=True, color=[0,0,0,255])
        text_rect = text_surface.get_rect()
        
        text_rect.center = rect.center
        
        s.blit(text_surface, text_rect)
        
        return s
        
    
    @classmethod
    def init(cls):
        assert (pg.font.get_init() and pg.display.get_init())
        
        cls.FONT = pg.font.SysFont("hg行書体", MASS_SIZE[0])
        
        cls.PLAYER1_WIN_SURFACE = cls.make_win_surface("Player1の勝ち")
        cls.PLAYER2_WIN_SURFACE = cls.make_win_surface("Player2の勝ち")
        