from enum import IntEnum

PIECE_KINDS = 16

class Piece(IntEnum):
    Space = 0
    """_summary_ 空きマス
    """
    Enemy = -1
    """_summary_ 敵駒
    """
    
    General = 1
    """_summary_ 大将
    """
    LieutenantGeneral = 2
    """_summary_ 中将
    """
    MajorGeneral = 3
    """_summary_ 少将
    """
    Colonel = 4
    """_summary_ 大佐
    """
    LieutenantColonel = 5
    """_summary_ 中佐
    """
    Major = 6
    """_summary_ 少佐
    """
    Captain = 7
    """_summary_ 大尉
    """
    FirstLieunant = 8
    """_summary_ 中尉
    """
    SecondLieunant = 9
    """_summary_ 少尉
    """
    Plane = 10
    """_summary_ ヒコーキ
    """
    Tank = 11
    """_summary_ タンク
    """
    Cavalry = 12
    """_summary_ 騎兵
    """
    Engineer = 13
    """_summary_ 工兵
    """
    Spy = 14
    """_summary_ スパイ
    """
    LandMine = 15
    """_summary_ 地雷
    """
    Frag = 16
    """_summary_ 軍旗
    """
    
    
    