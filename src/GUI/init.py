from src.GUI.const import WINDOW_SIZE
from src.GUI.assets import make_assets

import pygame as pg

def init() -> pg.Surface:
    pg.display.init()
    pg.font.init()
    pg.mixer.init()
    
    screen = pg.display.set_mode(WINDOW_SIZE)
    
    make_assets()
    return screen