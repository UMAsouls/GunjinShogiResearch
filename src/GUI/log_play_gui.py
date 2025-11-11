from src.const import BOARD_SHAPE, BOARD_SHAPE_INT
from src.common import LogData, get_action, Player

from src.GUI.gui import GUI, make_reflect_pos_int

import pygame as pg
from pygame.locals import *

import numpy as np
import time

import GunjinShogiCore as GSC

class LogPlayGUI(GUI):
    
    def __init__(self, boardgui, env, log:np.ndarray, step_time: float = 0.5):
        super().__init__(boardgui, env)
        
        self.log: np.ndarray = log
        
        self.idx: int = 0
        self.step_time: float = step_time
        
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
            
            bef_pos, aft_pos = get_action(log.action)
            if(self._env.get_current_player() == GSC.Player.PLAYER_TWO):
                bef_pos = make_reflect_pos_int(bef_pos)
                aft_pos = make_reflect_pos_int(aft_pos)
            
            self.action(bef_pos, aft_pos)
            
            bef_tuple= (bef_pos%BOARD_SHAPE[0], bef_pos//BOARD_SHAPE[0])
            aft_tuple= (aft_pos%BOARD_SHAPE[0], aft_pos//BOARD_SHAPE[0])
            self._boardgui.set_selected_pos(bef_tuple)
            self._boardgui.set_emp_pos(aft_tuple)
            
            self._boardgui.chg_appear()
            
            self.idx += 1
            t0 = time.time()
            