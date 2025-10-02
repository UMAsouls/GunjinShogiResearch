from src.GUI import init

from src.GUI.const import WINDOW_SIZE, MASS_SIZE
from src.GUI.assets import PieceSurface

import pygame as pg
from pygame.locals import *

def blit_test():
    screen = init()
    
    done = False
    while not done:
        pg.display.update()
        i = 0
        for k,v in PieceSurface.IMG_DICT.items():
            x,y = i%6, i//6
            pos = (MASS_SIZE[0]*x, MASS_SIZE[1]*y)
            screen.blit(v, pos)
            i += 1
            
        for event in pg.event.get():
            if(event.type == QUIT):
                done = True
                pg.quit()
                return
            
if __name__ == "__main__":
    blit_test()            
        
    
                