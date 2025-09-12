from src.const import BOARD_SHAPE, BOARD_SHAPE_INT


def get_action(action:int) -> tuple[int, int]:
    (bef, aft) = [action//BOARD_SHAPE_INT, action%BOARD_SHAPE_INT]
        
    return bef,aft

def make_action(bef:int, aft:int) -> int:
    action = bef * BOARD_SHAPE_INT + aft
    
    return action