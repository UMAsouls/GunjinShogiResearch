from src.const.Piece import Piece

piece_dict = [
    Piece.General,
    Piece.LieutenantGeneral,
    Piece.MajorGeneral,
    Piece.Colonel,
    Piece.LieutenantColonel,
    Piece.Major,
    Piece.Captain,
    Piece.Captain,
    Piece.FirstLieunant,
    Piece.FirstLieunant,
    Piece.SecondLieunant,
    Piece.SecondLieunant,
    Piece.Plane,
    Piece.Plane,
    Piece.Tank,
    Piece.Tank,
    Piece.Cavalry,
    Piece.Engineer,
    Piece.Engineer,
    Piece.Spy,
    Piece.LandMine,
    Piece.LandMine,
    Piece.Frag
]

PIECE_LIMIT = len(piece_dict)

PIECE_DICT = [int(i) for i in piece_dict]

GOAL_PIECES = [
    Piece.General,
    Piece.LieutenantGeneral,
    Piece.MajorGeneral,
    Piece.Colonel,
    Piece.LieutenantColonel,
    Piece.Major
]
