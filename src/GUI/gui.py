from src.GUI.Interfaces import IBoardGUI
from src.GUI.init import init

from src.GUI.assets import EndSurface
from src.GUI.const import END_SURFACE_SIZE

from src.const import BOARD_SHAPE, BOARD_SHAPE_INT
from src.common import make_action,get_action, EraseFrag, Player
from src.Interfaces import IEnv

import pygame as pg
from pygame.locals import *

#Envは現在の手番側から見た選択可能な行動を出力する
#そのため、gui側から変換が必須
def make_reflect_pos(pos:tuple[int,int]) -> tuple[int,int]:
    return (BOARD_SHAPE[0] - pos[0], BOARD_SHAPE[1] - pos[1])

def make_reflect_pos_int(pos_int:int) -> int:
    return BOARD_SHAPE_INT - (pos_int+1)

class GUI:
    def __init__(self, boardgui: IBoardGUI, env: IEnv):
        self._boardgui = boardgui
        
        #移動可能判定用リスト
        self.judge_legal_pos_list: list[list[int]] = \
            [[] for _ in range(BOARD_SHAPE[0]*BOARD_SHAPE[1])]
        self.judge_legal_pos: list[int] = []
        
        #board_guiに渡す用のリスト
        self.legal_pos: list[list[list[tuple[int,int]]]] = \
            [[[] for _ in range(BOARD_SHAPE[0])] for _ in range(BOARD_SHAPE[1])]
            
        self._selected_pos: int = -1
        
        self._env = env
        
        self.done: bool = False
        
        self.set_legal_move()
        
    def _legal_pos_reset(self):
        self.judge_legal_pos_list: list[list[int]] = \
            [[] for _ in range(BOARD_SHAPE[0]*BOARD_SHAPE[1])]
        self.judge_legal_pos: list[int] = []
        
        self.legal_pos: list[list[list[tuple[int,int]]]] = \
            [[[] for _ in range(BOARD_SHAPE[0])] for _ in range(BOARD_SHAPE[1])]
            
        
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
    
    def set_legal_move(self) -> None:
        legal_tensor = self._env.legal_move()
        self._legal_pos_reset()
        
        for i in legal_tensor:
            act = int(i.item())
            i_bef,i_aft = get_action(act)
            if(self._env.get_current_player() == Player.PLAYER2):
                i_bef,i_aft = make_reflect_pos_int(i_bef),make_reflect_pos_int(i_aft)
            
            self.judge_legal_pos_list[i_bef].append(i_aft)
            
            i_bef_pos = (i_bef%BOARD_SHAPE[0], i_bef//BOARD_SHAPE[0])
            i_aft_pos = (i_aft%BOARD_SHAPE[0], i_aft//BOARD_SHAPE[0])
            self.legal_pos[i_bef_pos[1]][i_bef_pos[0]].append(i_aft_pos)
    
    def action(self, bef: int, aft: int) -> bool:        
        aciton: int
        if(self._env.get_current_player() == Player.PLAYER2):
            b,a = make_reflect_pos_int(bef),make_reflect_pos_int(aft)
            action = make_action(b,a)
        else:
            action = make_action(bef,aft)
        
        _, log, self.done = self._env.step(action)
        
        bef_pos = (bef%BOARD_SHAPE[0], bef//BOARD_SHAPE[0])
        aft_pos = (aft%BOARD_SHAPE[0], aft//BOARD_SHAPE[0])
        
        if(log.erase == EraseFrag.AFTER or log.erase == EraseFrag.NO):
            self._boardgui.move(bef_pos, aft_pos)
        elif(log.erase == EraseFrag.BEFORE):
            self._boardgui.erase(bef_pos)
        else:
            self._boardgui.erase(bef_pos)
            self._boardgui.erase(aft_pos)
            
        self.set_legal_move()
        self._boardgui.chg_appear()
        
        
        
    def draw(self, screen: pg.Surface) -> None:
        self._boardgui.draw(screen)
        
        if(self.done): 
            rect = pg.Rect((0,0),END_SURFACE_SIZE)
            rect.center = screen.get_rect().center
            if(self._env.get_current_player() == Player.PLAYER2):
                screen.blit(EndSurface.PLAYER1_WIN_SURFACE, rect)
            else:
                screen.blit(EndSurface.PLAYER2_WIN_SURFACE, rect)
                
        
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

    
        
        
    
    