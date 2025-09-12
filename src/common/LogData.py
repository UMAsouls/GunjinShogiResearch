from src.common.EraseFrag import EraseFrag
from src.common.Player import Player

from dataclasses import dataclass

@dataclass
class LogData:
    action: int
    player: Player
    erase: EraseFrag
    bef: int
    aft: int
    
    