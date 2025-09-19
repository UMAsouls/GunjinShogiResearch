from src.const import BOARD_SHAPE

from src.GUI.gui import GUI

import pygame as pg
from pygame.locals import *

class PlayGUI(GUI):
    
    def __init__(self, boardgui, env):
        super().__init__(boardgui, env)
    
    def set_mouse_pos(self, mouse_pos: tuple[int,int]):
        onboard = self._boardgui.get_selected_pos(mouse_pos)
        self._boardgui.set_emp_pos(onboard)

    def set_clicked_pos(self, mouse_pos: tuple[int,int]):
        onboard = self._boardgui.get_selected_pos(mouse_pos)
        
        isonboard = self._boardgui.set_selected_pos(onboard)
        if isonboard:
            self._click_board(onboard)
            self._boardgui.set_legal_pos(self.legal_pos[onboard[1]][onboard[0]])
        else: 
            self._boardgui.set_legal_pos([])
        
    def _click_board(self, onboard_pos: tuple[int,int]) -> None:
        pos_int = onboard_pos[1]*BOARD_SHAPE[0] + onboard_pos[0]
        
        if(pos_int in self.judge_legal_pos):
            self.action(self._selected_pos, pos_int)
        
        self._selected_pos = pos_int
        self.judge_legal_pos = self.judge_legal_pos_list[self._selected_pos]
        return
    
    def main_loop(self, screen: pg.Surface) -> None:
        app_end = False
        while not app_end:
            pg.display.update()
            
            self.draw(screen)
            
            for event in pg.event.get():
                if(event.type == QUIT):
                    app_end = True
                    pg.quit()
                    return
                
            if(self.done): continue
            
            clicked = pg.mouse.get_pressed()[0]
            self.set_mouse_pos(pg.mouse.get_pos())
            if(clicked): self.set_clicked_pos(pg.mouse.get_pos())