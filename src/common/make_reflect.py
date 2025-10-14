from src.const import BOARD_SHAPE, BOARD_SHAPE_INT

def make_reflect_pos(pos:tuple[int,int]) -> tuple[int,int]:
    return (BOARD_SHAPE[0] - pos[0]-1, BOARD_SHAPE[1] - pos[1]-1)

def make_reflect_pos_int(pos_int:int) -> int:
    return BOARD_SHAPE_INT - (pos_int+1)