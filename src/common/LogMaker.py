from src.common.LogData import LogData
from src.common.make_log_pathes import make_log_pathes

from src.const import LOG_DIR

import numpy as np
import os

class LogMaker:
    def __init__(self, log_name: str):
        self.log_name = log_name
        
        self.pieces1: np.ndarray
        self.pieces2: np.ndarray
        
        self.steps: list[list[int]] = []
        
    def add_pieces(self, pieces1: np.ndarray, pieces2: np.ndarray):
        self.pieces1 = pieces1
        self.pieces2 = pieces2
        
    def add_step(self, log: LogData) -> None:
        self.steps.append([log.action, log.player, log.erase, log.bef, log.aft])
        
    def save(self) -> None:
        step_array = np.array(self.steps)
        
        p1_path, p2_path, s_path = make_log_pathes(self.log_name)
        os.makedirs(name=f"{LOG_DIR}/{self.log_name}", exist_ok=True)
        np.save(p1_path, self.pieces1)
        np.save(p2_path, self.pieces2)
        np.save(s_path, step_array)
        
    @classmethod
    def load(cls, log_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p1_path, p2_path, s_path = make_log_pathes(log_name)
        
        pieces1 = np.load(p1_path)
        pieces2 = np.load(p2_path)
        steps = np.load(s_path)
        
        return pieces1, pieces2, steps