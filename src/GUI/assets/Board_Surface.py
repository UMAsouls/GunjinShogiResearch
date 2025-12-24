from src.GUI.const import \
    BOARD_SIZE, MASS_SIZE, BOARD_COLOR

from src.common import Config

import pygame as pg
import pygame.gfxdraw as gfxdraw

LINE_THICKNESS = 2

LINE_COLOR = (0,0,0,255)
WALL_COLOR = (15,35,80,255)
GOAL_COLOR = (255,255,255,255)

EMP_COLOR = (0,255,0,255)
SELECTED_COLOR = (0,0,255,255)
LEGAL_COLOR = (255,0,0,255)
EMP_THICKNESS = 5

def draw_rectangle(
    surface: pg.Surface, rect: pg.Rect, 
    line_color: tuple[int,int,int,int] = (0,0,0,255), 
    fill_color: tuple[int,int,int,int] = (0,0,0,0),
    thick: int = 1
    ):
    mini_rect = rect.copy()
    
    mini_rect.width -= thick*2
    mini_rect.height -= thick*2
    
    if(mini_rect.width <= 0): mini_rect.width = 1
    if(mini_rect.height <= 0): mini_rect.height = 1
    
    mini_rect.center = [rect.width//2, rect.height//2]
    
    s = pg.Surface(rect.size).convert_alpha()
    s.fill(line_color)
    s.fill(fill_color, mini_rect)
    
    surface.blit(s, rect)

class BoardSurface:
    BOARD_IMG: pg.Surface = None
    
    EMP_IMG: pg.Surface = None
    SELECTED_IMG: pg.Surface = None
    LEGAL_IMG: pg.Surface = None
    
    
    @classmethod
    def make_emp_img(cls, color) -> None:
        img = pg.Surface(MASS_SIZE).convert_alpha()
        img.fill((0,0,0,0))
        draw_rectangle(
            img, img.get_rect(),
            line_color=color, thick=EMP_THICKNESS
        )
        
        return img
        
    @classmethod
    def draw_mass_line(cls, x:int, y:int) -> None:
        pos = (MASS_SIZE[0]*x, MASS_SIZE[1]*y)
        rect = pg.Rect(pos, MASS_SIZE)
                
        line = LINE_COLOR
        fill = (0,0,0,0)
        if(y == Config.entry_height):
            if(x in Config.entry_pos):fill = GOAL_COLOR
            else: fill = WALL_COLOR
        if(y == Config.goal_height or y == Config.board_shape[1]-1-Config.goal_height):
            if(x in Config.goal_pos): fill = GOAL_COLOR
                        
        draw_rectangle(
            cls.BOARD_IMG, rect, 
            line_color=line, fill_color=fill, 
            thick=LINE_THICKNESS
        )
    
    @classmethod
    def init(cls):
        assert pg.display.get_init()
        
        cls.BOARD_IMG = pg.Surface(BOARD_SIZE).convert_alpha()
        cls.BOARD_IMG.fill(BOARD_COLOR)
        
        for y in range(Config.board_shape[1]):
            for x in range(Config.board_shape[0]):
                cls.draw_mass_line(x,y)
                
        cls.EMP_IMG = cls.make_emp_img(EMP_COLOR)
        cls.SELECTED_IMG = cls.make_emp_img(SELECTED_COLOR)
        cls.LEGAL_IMG = cls.make_emp_img(LEGAL_COLOR)
                
        
            
        
        
    