from src.GUI import init

from src.GUI.const import WINDOW_SIZE, MASS_SIZE
from src.GUI.assets import BoardSurface

import pygame as pg
from pygame.locals import *

def blit_test():
    screen = init()
    
    done = False
    while not done:
        pg.display.update()
        screen.blit(BoardSurface.BOARD_IMG.copy())
        screen.blit(BoardSurface.EMP_IMG.copy())
            
        for event in pg.event.get():
            if(event.type == QUIT):
                done = True
                pg.quit()
                return
            
if __name__ == "__main__":
    blit_test()   