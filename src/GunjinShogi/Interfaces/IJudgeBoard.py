from abc import ABC, abstractmethod
import numpy as np

from GunjinShogi.Interfaces.IBoard import IBoard

class IJudgeBoard(IBoard):
    
    @abstractmethod
    def judge(self, player: int) ->bool:
        pass