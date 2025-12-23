import json

class Config:
    data: dict[str] = {}
    loaded:bool = False

    board_shape: tuple[int,int] = ()
    board_shape_int: int = -1

    entry_height: int = -1
    entry_pos: list[int] = []

    goal_height: int = -1
    goal_pos: list[int] = []

    piece_limit: int = -1

    @classmethod
    def load(cls,path:str):
        cls.data = json.load(path)
        cls.loaded = False

        cls.board_shape = cls.data["BOARD"]["SHAPE"]
        cls.board_shape_int = cls.board_shape[0] * cls.board_shape[1]

        cls.entry_height = cls.data["BOARD"]["ENTRY"]["HEIGHT"]
        cls.entry_pos = cls.data["BOARD"]["ENTRY"]["POS"]

        cls.goal_height = cls.data["BOARD"]["GOAL"]["HEIGHT"]
        cls.goal_pos = cls.data["BOARD"]["GOAL"]["POS"]

        cls.piece_limit = cls.data["BOARD"]["GOAL"]["PIECE_LIMIT"]


