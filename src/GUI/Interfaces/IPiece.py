from abc import ABC, abstractmethod

import pygame as pg

class IPieceGUI(ABC):
    @abstractmethod
    def set_location(self, pos:tuple[int,int]) -> None:
        pass
    
    @abstractmethod
    def draw(self, surface: pg.Surface) -> None:
        pass
    
    @abstractmethod
    def chg_appear(self) -> None:
        pass