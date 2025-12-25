from src.common.EraseFrag import EraseFrag
from src.common.Player import Player

from dataclasses import dataclass

@dataclass
class LogData:
    action: int
    player: int
    erase: int
    bef: int
    aft: int
    
    