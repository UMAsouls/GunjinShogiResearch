from src.const.Piece import Piece, PIECE_KINDS
from src.GunjinShogi.const.JudgeFrag import JudgeFrag

import numpy as np

JUDGE_TABLE = np.full((PIECE_KINDS+1,PIECE_KINDS+1), int(JudgeFrag.Lose), dtype = np.int8)

for i in range(PIECE_KINDS+1):
    JUDGE_TABLE[i,i] = int(JudgeFrag.Draw)

#大将の設定
JUDGE_TABLE[Piece.General] = np.full(PIECE_KINDS+1, int(JudgeFrag.Win), dtype = np.int8)
JUDGE_TABLE[Piece.General][Piece.Spy] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.General][Piece.LandMine] = int(JudgeFrag.Lose)

#中将の設定
JUDGE_TABLE[Piece.LieutenantGeneral] = np.full(PIECE_KINDS+1, int(JudgeFrag.Win), dtype = np.int8)
JUDGE_TABLE[Piece.LieutenantGeneral][Piece.General] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.LieutenantGeneral][Piece.LandMine] = int(JudgeFrag.Lose)

#少将の設定
JUDGE_TABLE[Piece.MajorGeneral] = np.full(PIECE_KINDS+1, int(JudgeFrag.Win), dtype = np.int8)
JUDGE_TABLE[Piece.MajorGeneral][Piece.General] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.MajorGeneral][Piece.LieutenantGeneral] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.MajorGeneral][Piece.LandMine] = int(JudgeFrag.Lose)

#大佐の設定
JUDGE_TABLE[Piece.Colonel] = np.full(PIECE_KINDS+1, int(JudgeFrag.Win), dtype = np.int8)
JUDGE_TABLE[Piece.Colonel][Piece.General] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Colonel][Piece.LieutenantGeneral] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Colonel][Piece.MajorGeneral] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Colonel][Piece.Plane] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Colonel][Piece.Tank] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Colonel][Piece.LandMine] = int(JudgeFrag.Lose)

#中佐の設定
JUDGE_TABLE[Piece.LieutenantColonel] = np.full(PIECE_KINDS+1, int(JudgeFrag.Win), dtype = np.int8)
JUDGE_TABLE[Piece.LieutenantColonel][Piece.General] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.LieutenantColonel][Piece.LieutenantGeneral] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.LieutenantColonel][Piece.MajorGeneral] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.LieutenantColonel][Piece.Colonel] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.LieutenantColonel][Piece.Plane] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.LieutenantColonel][Piece.Tank] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.LieutenantColonel][Piece.LandMine] = int(JudgeFrag.Lose)

#少佐の設定
JUDGE_TABLE[Piece.Major] = np.full(PIECE_KINDS+1, int(JudgeFrag.Win), dtype = np.int8)
JUDGE_TABLE[Piece.Major][Piece.General] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Major][Piece.LieutenantGeneral] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Major][Piece.MajorGeneral] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Major][Piece.Colonel] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Major][Piece.LieutenantColonel] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Major][Piece.Plane] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Major][Piece.Tank] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Major][Piece.LandMine] = int(JudgeFrag.Lose)

#大尉の設定
JUDGE_TABLE[Piece.Captain][Piece.FirstLieunant] = int(JudgeFrag.Win)
JUDGE_TABLE[Piece.Captain][Piece.SecondLieunant] = int(JudgeFrag.Win)
JUDGE_TABLE[Piece.Captain][Piece.Cavalry] = int(JudgeFrag.Win)
JUDGE_TABLE[Piece.Captain][Piece.Engineer] = int(JudgeFrag.Win)
JUDGE_TABLE[Piece.Captain][Piece.Spy] = int(JudgeFrag.Win)

#中尉の設定
JUDGE_TABLE[Piece.FirstLieunant][Piece.SecondLieunant] = int(JudgeFrag.Win)
JUDGE_TABLE[Piece.FirstLieunant][Piece.Cavalry] = int(JudgeFrag.Win)
JUDGE_TABLE[Piece.FirstLieunant][Piece.Engineer] = int(JudgeFrag.Win)
JUDGE_TABLE[Piece.FirstLieunant][Piece.Spy] = int(JudgeFrag.Win)

#少尉の設定
JUDGE_TABLE[Piece.SecondLieunant][Piece.Cavalry] = int(JudgeFrag.Win)
JUDGE_TABLE[Piece.SecondLieunant][Piece.Engineer] = int(JudgeFrag.Win)
JUDGE_TABLE[Piece.SecondLieunant][Piece.Spy] = int(JudgeFrag.Win)

#騎兵の設定
JUDGE_TABLE[Piece.Cavalry][Piece.Engineer] = int(JudgeFrag.Win)
JUDGE_TABLE[Piece.Cavalry][Piece.Spy] = int(JudgeFrag.Win)

#工兵の設定
JUDGE_TABLE[Piece.Engineer][Piece.Spy] = int(JudgeFrag.Win)
JUDGE_TABLE[Piece.Engineer][Piece.LandMine] = int(JudgeFrag.Win)

#ヒコーキの設定
JUDGE_TABLE[Piece.Plane] = np.full(PIECE_KINDS+1, int(JudgeFrag.Win), dtype = np.int8)
JUDGE_TABLE[Piece.Plane][Piece.General] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Plane][Piece.LieutenantGeneral] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Plane][Piece.MajorGeneral] = int(JudgeFrag.Lose)

#タンクの設定
JUDGE_TABLE[Piece.Tank] = np.full(PIECE_KINDS+1, int(JudgeFrag.Win), dtype = np.int8)
JUDGE_TABLE[Piece.Tank][Piece.General] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Tank][Piece.LieutenantGeneral] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Tank][Piece.MajorGeneral] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.Tank][Piece.LandMine] = int(JudgeFrag.Lose)

#スパイの設定
JUDGE_TABLE[Piece.Spy][Piece.General] = int(JudgeFrag.Win)

#地雷の設定
JUDGE_TABLE[Piece.LandMine] = np.full(PIECE_KINDS+1, int(JudgeFrag.Win), dtype = np.int8)
JUDGE_TABLE[Piece.LandMine][Piece.Plane] = int(JudgeFrag.Lose)
JUDGE_TABLE[Piece.LandMine][Piece.Engineer] = int(JudgeFrag.Lose)
