from src.const import BOARD_SHAPE, BOARD_SHAPE_INT

from src.common.Config import Config

def make_reflect_pos(pos:tuple[int,int]) -> tuple[int,int]:
    return (Config.board_shape[0] - pos[0]-1, Config.board_shape[1] - pos[1]-1)

def make_reflect_pos_int(pos_int:int) -> int:
    return Config.board_shape_int - (pos_int+1)