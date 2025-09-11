from abc import ABC, abstractmethod

from src.GUI.Interfaces.IPiece import IPieceGUI

import pygame as pg

class IBoardGUI(ABC):
    @abstractmethod
    def set_emp_pos(self, pos:tuple[int,int]) -> bool:
        pass
    
    @abstractmethod
    def set_selected_pos(self, pos:tuple[int,int]) -> bool:
        pass
    
    @abstractmethod
    def set_legal_pos(self, pos:list[tuple[int,int]]) -> None:
        pass
    
    @abstractmethod
    def get_selected_pos(self, screen_pos: tuple[int,int]) -> tuple[int,int]:
        pass
    
    @abstractmethod
    def get_piece(self, x_idx:int, y_idx:int) -> IPieceGUI:
        pass
    
    @abstractmethod
    def chg_appear(self) -> None:
        pass
    
    @abstractmethod
    def draw(self, screen:pg.Surface):
        pass
    
    @abstractmethod
    def move(self, bef: tuple[int,int], aft: tuple[int,int]) -> bool:
        pass
    
    @abstractmethod
    def erase(self, pos: tuple[int,int]) -> bool:
        pass