from src.const import BOARD_SHAPE, BOARD_SHAPE_INT
from src.common import LogData

from src.GUI.gui import GUI

import pygame as pg
from pygame.locals import *

import numpy as np
import time

class LogPlayGUI(GUI):
    
    def __init__(self, boardgui, env, log_path: str, step_time: float):
        super().__init__(boardgui, env)
        
        self.log: np.ndarray
        self.load(log_path)
        
        self.idx: int = 0
        self.step_time: float = step_time
        
    def load(self, log_path: str):
        self.log = np.load(log_path)
        
    def main_loop(self, screen: pg.Surface) -> None:
        app_end = False
        t0 = time.time()
        while not app_end:
            pg.display.update()
            
            self.draw(screen)
            
            for event in pg.event.get():
                if(event.type == QUIT):
                    app_end = True
                    pg.quit()
                    return
                
            if(self.done): continue
            
            if(time.time() - t0 < self.step_time): continue
            
            data = self.log[self.idx]
            log = LogData(data[0], data[1], data[2], data[3], data[4])
            
            bef_pos, aft_pos = log.action // BOARD_SHAPE_INT, log.action % BOARD_SHAPE_INT
            self.action(bef_pos, aft_pos)
            
            self.idx += 1
            t0 = time.time()
            