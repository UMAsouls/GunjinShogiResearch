"""
rule management
"""
from __future__ import annotations
import enum
import numpy
import numpy.typing
import typing
__all__: list[str] = ['AFT', 'BEF', 'BOTH', 'BattleEndFrag', 'CONTINUE', 'EraseFrag', 'JudgeBoard', 'JudgeFrag', 'LOSE', 'MakeJudgeBoard', 'PIECE_DRAW', 'PIECE_LOSE', 'PIECE_WIN', 'PLAYER_ONE', 'PLAYER_TWO', 'Player', 'WIN']
class BattleEndFrag(enum.Enum):
    CONTINUE: typing.ClassVar[BattleEndFrag]  # value = <BattleEndFrag.CONTINUE: 2>
    LOSE: typing.ClassVar[BattleEndFrag]  # value = <BattleEndFrag.LOSE: 1>
    WIN: typing.ClassVar[BattleEndFrag]  # value = <BattleEndFrag.WIN: 0>
class EraseFrag(enum.Enum):
    AFT: typing.ClassVar[EraseFrag]  # value = <EraseFrag.AFT: 1>
    BEF: typing.ClassVar[EraseFrag]  # value = <EraseFrag.BEF: 0>
    BOTH: typing.ClassVar[EraseFrag]  # value = <EraseFrag.BOTH: 2>
class JudgeBoard:
    def erase(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2: Player) -> None:
        ...
    def getJudge(self, arg0: typing.SupportsInt, arg1: Player) -> JudgeFrag:
        ...
    def getLegalMove(self, arg0: Player) -> numpy.typing.NDArray[numpy.int32]:
        ...
    def isWin(self, arg0: Player) -> BattleEndFrag:
        ...
    def move(self, arg0: typing.SupportsInt, arg1: Player) -> None:
        ...
    def reset(self) -> None:
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
def MakeJudgeBoard(arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32], arg2: str) -> JudgeBoard:
    """
    Make JudgeBoard function
    """
AFT: EraseFrag  # value = <EraseFrag.AFT: 1>
BEF: EraseFrag  # value = <EraseFrag.BEF: 0>
BOTH: EraseFrag  # value = <EraseFrag.BOTH: 2>
CONTINUE: BattleEndFrag  # value = <BattleEndFrag.CONTINUE: 2>
LOSE: BattleEndFrag  # value = <BattleEndFrag.LOSE: 1>
PIECE_DRAW: JudgeFrag  # value = <JudgeFrag.PIECE_DRAW: 2>
PIECE_LOSE: JudgeFrag  # value = <JudgeFrag.PIECE_LOSE: 1>
PIECE_WIN: JudgeFrag  # value = <JudgeFrag.PIECE_WIN: 0>
PLAYER_ONE: Player  # value = <Player.PLAYER_ONE: 0>
PLAYER_TWO: Player  # value = <Player.PLAYER_TWO: 1>
WIN: BattleEndFrag  # value = <BattleEndFrag.WIN: 0>
