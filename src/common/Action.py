from src.const import BOARD_SHAPE, BOARD_SHAPE_INT
from src.common.make_reflect import make_reflect_pos_int

from src.common.Config import Config



def get_action(action:int) -> tuple[int, int]:
    (bef, aft) = [action//Config.board_shape_int, action%Config.board_shape_int]
        
    return bef,aft

def make_action(bef:int, aft:int) -> int:
    action = bef * Config.board_shape_int + aft
    
    return action

def make_reflect_action(action:int) -> int:
    bef, aft = get_action(action)
    ref_bef = make_reflect_pos_int(bef)
    ref_aft = make_reflect_pos_int(aft)
    return make_action(ref_bef, ref_aft)

#int単体の位置をx,yに変換
def change_pos_int_to_tuple(pos:int) -> tuple[int,int]:
    return pos%Config.board_shape[0], pos//Config.board_shape[0]

def change_pos_tuple_to_int(x:int, y:int) -> int:
    return y*Config.board_shape[0] + x
