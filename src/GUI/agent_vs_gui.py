from src.const import BOARD_SHAPE
from src.Interfaces import IAgent

from src.common import get_action,LogMaker ,make_reflect_action

from src.GUI.play_gui import PlayGUI

import pygame as pg
from pygame.locals import *

class AgentVsGUI(PlayGUI):
    
    def __init__(self, boardgui, env, agent:IAgent, log_maker:LogMaker, player_first: bool = True):
        super().__init__(boardgui, env)
        self.agent = agent
        
        self.player_turn = player_first
        self.player_first = player_first
        
        self.log_maker = log_maker
        
    def action(self, bef, aft):
        log = super().action(bef, aft)
        self._boardgui.chg_appear()
        self.player_turn = not self.player_turn
        
        self.log_maker.add_step(log)
        
    def agent_move(self):
        action = self.agent.get_action(self._env)
        if(self.player_first): action = make_reflect_action(action)
        bef,aft = get_action(action)
        self.action(bef,aft)

        
    def main_loop(self, screen: pg.Surface) -> None:
        app_end = False
        while not app_end:
            pg.display.update()
            
            self.draw(screen)
            
            clicked = False
            for event in pg.event.get():
                if(event.type == QUIT):
                    app_end = True
                    pg.quit()
                    return
                if(event.type == MOUSEBUTTONDOWN):
                    if(event.button == 1):
                        clicked = True
                
            if(self.done): continue
            
            if(self.player_turn):
                self.set_mouse_pos(pg.mouse.get_pos())
                if(clicked): self.set_clicked_pos(pg.mouse.get_pos())
            else:
                self.agent_move()