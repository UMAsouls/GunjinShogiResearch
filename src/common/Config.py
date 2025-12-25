import json
import numpy as np

from src.const import BOARD_SHAPE, BOARD_SHAPE_INT, ENTRY_HEIGHT, ENTRY_POS,\
      GOAL_POS, GOAL_HEIGHT, PIECE_LIMIT, Piece, PIECE_KINDS, PIECE_DICT


class Config:
    data: dict[str] = {}
    loaded:bool = False

    board_shape: tuple[int,int] = BOARD_SHAPE
    board_shape_int: int = BOARD_SHAPE_INT

    entry_height: int = ENTRY_HEIGHT
    entry_pos: list[int] = ENTRY_POS

    goal_height: int = GOAL_HEIGHT
    reflect_goal_height: int = BOARD_SHAPE[1] - 1 - GOAL_HEIGHT
    goal_pos: list[int] = GOAL_POS

    piece_limit: int = PIECE_LIMIT

    tensor_piece_id: dict[Piece,int] = {}

    piece_kinds = PIECE_KINDS

    judge_table: np.typing.NDArray[np.int8] = ()
    
    first_dict: list[Piece] = PIECE_DICT

    @classmethod
    def load(cls,path:str,judge_table):
        with open(path, "r") as f:
            cls.data = json.load(f)

        cls.loaded = True

        cls.board_shape = cls.data["BOARD"]["SHAPE"]
        cls.board_shape_int = cls.board_shape[0] * cls.board_shape[1]

        cls.entry_height = cls.data["BOARD"]["ENTRY"]["HEIGHT"]
        cls.entry_pos = cls.data["BOARD"]["ENTRY"]["POS"]

        cls.goal_height = cls.data["BOARD"]["GOAL"]["HEIGHT"]
        cls.goal_pos = cls.data["BOARD"]["GOAL"]["POS"]
        cls.reflect_goal_height = cls.board_shape[1] - 1 - cls.goal_height

        cls.piece_limit = cls.data["BOARD"]["PIECE_LIMIT"]

        use_piece = cls.data["USE_PIECES"]
        for v,p in enumerate(use_piece):
            cls.tensor_piece_id[p] = v

        cls.piece_kinds = PIECE_KINDS

        cls.judge_table = np.zeros((PIECE_KINDS,PIECE_KINDS), dtype=np.int8)

        for v1,p1 in enumerate(use_piece):
            for v2,p2 in enumerate(use_piece):
                cls.judge_table[v1,v2] = judge_table[p1][p2]
                
        first_piece = cls.data["FIRST_PIECE_ID"]
        cls.first_dict = [eval(f"Piece.{p}") for p in first_piece]
        

    @classmethod
    def get_tensor_id(cls,piece):
        return cls.tensor_piece_id[piece]


