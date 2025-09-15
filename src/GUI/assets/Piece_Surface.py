from src.GUI.const import MASS_SIZE, PIECE_POINTS, PIECE_COLOR

from src.const import Piece

import pygame as pg
import pygame.gfxdraw as gfxdraw

class PieceSurface:
    L_FONT:pg.Font = None
    M_FONT:pg.Font = None
    S_FONT:pg.Font = None

    PIECE_IMG:pg.Surface = None
    IMG_DICT:dict[Piece, pg.Surface] = {}

    @classmethod
    def get_piece_img(cls, s: str) -> pg.Surface:
        img = cls.PIECE_IMG.copy()
        rect = img.get_rect()

        font: pg.font.Font
        if(len(s) >= 3): font = cls.S_FONT
        elif(len(s) == 2): font = cls.M_FONT
        else: font = cls.L_FONT

        c = ""
        for i in s:
            c += f"{i}\n"

        text: pg.Surface = font.render(c, antialias=True, color=[0,0,0,255]).convert_alpha()
        t_rect = text.get_rect()
        t_rect.center = rect.center

        t_rect.top += 5

        img.blit(text, t_rect)

        return img

    @classmethod
    def init(cls):

        assert pg.font.get_init() and pg.display.get_init()

        cls.L_FONT = pg.font.SysFont("hg行書体", MASS_SIZE[0]*4//6)
        cls.M_FONT = pg.font.SysFont("hg行書体", MASS_SIZE[0]*4//12)
        cls.S_FONT = pg.font.SysFont("hg行書体", MASS_SIZE[0]*4//18)

        cls.PIECE_IMG = pg.Surface(MASS_SIZE).convert_alpha()
        cls.PIECE_IMG.fill([0,0,0,0])
        gfxdraw.filled_polygon(cls.PIECE_IMG, PIECE_POINTS,PIECE_COLOR)

        cls.IMG_DICT = {
            Piece.General: cls.get_piece_img("大将"),
            Piece.LieutenantGeneral: cls.get_piece_img("中将"),
            Piece.MajorGeneral: cls.get_piece_img("少将"),
            Piece.Colonel: cls.get_piece_img("大佐"),
            Piece.LieutenantColonel: cls.get_piece_img("中佐"),
            Piece.Major: cls.get_piece_img("少佐"),
            Piece.Captain: cls.get_piece_img("大尉"),
            Piece.FirstLieunant: cls.get_piece_img("中尉"),
            Piece.SecondLieunant: cls.get_piece_img("少尉"),
            Piece.Plane: cls.get_piece_img("飛行機"),
            Piece.Tank: cls.get_piece_img("戦車"),
            Piece.Cavalry: cls.get_piece_img("騎兵"),
            Piece.Engineer: cls.get_piece_img("工兵"),
            Piece.Spy: cls.get_piece_img("スパイ"),
            Piece.LandMine: cls.get_piece_img("地雷"),
            Piece.Frag: cls.get_piece_img("軍旗")
        }


