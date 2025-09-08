from src.GUI.const import BOARD_SIZE, BOARD_SHAPE, MASS_SIZE, BOARD_COLOR

import pygame as pg
import pygame.gfxdraw as gfxdraw

LINE_THICKNESS = 2

def draw_rectangle(surface: pg.Surface, rect: pg.Rect, color: tuple[int,int,int,int], thick: int):
    mini_rect = rect.copy()
    
    mini_rect.width -= thick*2
    mini_rect.height -= thick*2
    
    if(mini_rect.width <= 0): mini_rect.width = 1
    if(mini_rect.height <= 0): mini_rect.height = 1
    
    mini_rect.center = [rect.width//2, rect.height//2]
    
    s = pg.Surface(rect.size).convert_alpha()
    s.fill(color)
    s.fill([0,0,0,0], mini_rect)
    
    surface.blit(s, rect)

class BoardSurface:
    BOARD_IMG: pg.Surface = None
    
    @classmethod
    def init(cls):
        assert pg.display.get_init()
        
        cls.BOARD_IMG = pg.Surface(BOARD_SIZE).convert_alpha()
        cls.BOARD_IMG.fill(BOARD_COLOR)
        
        for y in range(BOARD_SHAPE[1]):
            for x in range(BOARD_SHAPE[0]):
                pos = (MASS_SIZE[0]*x, MASS_SIZE[1]*y)
                rect = pg.Rect(pos, MASS_SIZE)
                
                draw_rectangle(cls.BOARD_IMG, rect, color=[0,0,0,255], thick=LINE_THICKNESS)
                
        
            
        
        
    