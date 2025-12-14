"""
rule management
"""
from __future__ import annotations
import enum
import numpy
import numpy.typing
import typing
__all__: list[str] = ['AFT', 'BEF', 'BOTH', 'BattleEndFrag', 'CONTINUE', 'Config', 'DEPLOY_END', 'DRAW', 'EraseFrag', 'JudgeBoard', 'JudgeFrag', 'LOSE', 'MakeJudgeBoard', 'PIECE_DRAW', 'PIECE_LOSE', 'PIECE_WIN', 'PLAYER_ONE', 'PLAYER_TWO', 'Player', 'WIN']
class BattleEndFrag(enum.Enum):
    CONTINUE: typing.ClassVar[BattleEndFrag]  # value = <BattleEndFrag.CONTINUE: 2>
    DEPLOY_END: typing.ClassVar[BattleEndFrag]  # value = <BattleEndFrag.DEPLOY_END: 4>
    DRAW: typing.ClassVar[BattleEndFrag]  # value = <BattleEndFrag.DRAW: 3>
    LOSE: typing.ClassVar[BattleEndFrag]  # value = <BattleEndFrag.LOSE: 1>
    WIN: typing.ClassVar[BattleEndFrag]  # value = <BattleEndFrag.WIN: 0>
class Config:
    def getBoardShape(self) -> tuple[int, int]:
        ...
    def getEntryHeight(self) -> int:
        ...
    def getGoalHeight(self) -> int:
        ...
    def loadFromJson(self, arg0: str) -> bool:
        ...
class EraseFrag(enum.Enum):
    AFT: typing.ClassVar[EraseFrag]  # value = <EraseFrag.AFT: 1>
    BEF: typing.ClassVar[EraseFrag]  # value = <EraseFrag.BEF: 0>
    BOTH: typing.ClassVar[EraseFrag]  # value = <EraseFrag.BOTH: 2>
class JudgeBoard:
    def erase(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2: Player) -> None:
        ...
    def get(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2: Player) -> int:
        ...
    def getConfig(self) -> Config:
        ...
    def getDefinedBoard(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], arg1: Player) -> JudgeBoard:
        ...
    def getJudge(self, arg0: typing.SupportsInt, arg1: Player) -> JudgeFrag:
        ...
    def getLegalMove(self, arg0: Player) -> numpy.typing.NDArray[numpy.int32]:
        ...
    def get_int_board(self, arg0: Player) -> numpy.typing.NDArray[numpy.int32]:
        ...
    def isWin(self, arg0: Player) -> BattleEndFrag:
        ...
    def move(self, arg0: typing.SupportsInt, arg1: Player) -> None:
        ...
    def reset(self) -> None:
        ...
    def setBoard(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> None:
        ...
    def step(self, arg0: typing.SupportsInt, arg1: Player, arg2: EraseFrag) -> BattleEndFrag:
        ...
class JudgeFrag(enum.Enum):
    PIECE_DRAW: typing.ClassVar[JudgeFrag]  # value = <JudgeFrag.PIECE_DRAW: 2>
    PIECE_LOSE: typing.ClassVar[JudgeFrag]  # value = <JudgeFrag.PIECE_LOSE: 1>
    PIECE_WIN: typing.ClassVar[JudgeFrag]  # value = <JudgeFrag.PIECE_WIN: 0>
class Player(enum.Enum):
    PLAYER_ONE: typing.ClassVar[Player]  # value = <Player.PLAYER_ONE: 0>
    PLAYER_TWO: typing.ClassVar[Player]  # value = <Player.PLAYER_TWO: 1>
def MakeJudgeBoard(arg0: str) -> JudgeBoard:
    """
    Make JudgeBoard function
    """

