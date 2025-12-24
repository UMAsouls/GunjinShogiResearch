import json

from src.const import BOARD_SHAPE, BOARD_SHAPE_INT, ENTRY_HEIGHT, ENTRY_POS, GOAL_POS, GOAL_HEIGHT, PIECE_LIMIT


class Config:
    data: dict[str] = {}
    loaded:bool = False

    board_shape: tuple[int,int] = BOARD_SHAPE
    board_shape_int: int = BOARD_SHAPE_INT

    entry_height: int = ENTRY_HEIGHT
    entry_pos: list[int] = ENTRY_POS

    goal_height: int = GOAL_HEIGHT
    goal_pos: list[int] = GOAL_POS

    piece_limit: int = PIECE_LIMIT

    @classmethod
    def load(cls,path:str):
        with open(path, "r") as f:
            cls.data = json.load(f)

        cls.loaded = True

        cls.board_shape = cls.data["BOARD"]["SHAPE"]
        cls.board_shape_int = cls.board_shape[0] * cls.board_shape[1]

        cls.entry_height = cls.data["BOARD"]["ENTRY"]["HEIGHT"]
        cls.entry_pos = cls.data["BOARD"]["ENTRY"]["POS"]

        cls.goal_height = cls.data["BOARD"]["GOAL"]["HEIGHT"]
        cls.goal_pos = cls.data["BOARD"]["GOAL"]["POS"]

        cls.piece_limit = cls.data["BOARD"]["PIECE_LIMIT"]


