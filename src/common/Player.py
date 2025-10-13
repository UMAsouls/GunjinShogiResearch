from enum import IntEnum

class Player(IntEnum):
    PLAYER1 = 1
    PLAYER2 = 2
    
def get_opponent(p: Player):
    return Player.PLAYER2 if(p == Player.PLAYER1) else Player.PLAYER1