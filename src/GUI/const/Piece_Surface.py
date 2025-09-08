from src.GUI.const.Sizes import MASS_SIZE, PIECE_POINTS
from src.GUI.const.Colors import PIECE_COLOR

from const import Piece

import pygame as pg
import pygame.gfxdraw as gfxdraw

L_FONT = pg.font.SysFont("hg行書体", MASS_SIZE[0]*5/6)
M_FONT = pg.font.SysFont("hg行書体", MASS_SIZE[0]*5/12)
S_FONT = pg.font.SysFont("hg行書体", MASS_SIZE[0]*5/18)

PIECE_IMG = pg.Surface(MASS_SIZE).convert_alpha()
PIECE_IMG.fill([0,0,0,0])
gfxdraw.filled_polygon(PIECE_IMG,PIECE_POINTS,PIECE_COLOR)

def get_piece_img(s: str) -> pg.Surface:
    img = PIECE_IMG.copy()
    rect = img.get_rect()
    
    font: pg.font.Font
    if(len(s) >= 3): font = S_FONT
    elif(len(s) == 2): font = M_FONT
    else: font = L_FONT
    
    text: pg.Surface = font.render(s, color=[0,0,0,255], bgcolor=[0,0,0,0])
    t_rect = text.get_rect()
    t_rect.center = rect.center
    
    img.blit(text, t_rect)
    
    return img

IMG_DICT = {
    Piece.General: get_piece_img("大将"),
    Piece.LieutenantGeneral: get_piece_img("中将"),
    Piece.MajorGeneral: get_piece_img("少将"),
    Piece.Colonel: get_piece_img("大佐"),
    Piece.LieutenantColonel: get_piece_img("中佐"),
    Piece.Major: get_piece_img("少佐"),
    Piece.Captain: get_piece_img("大尉"),
    Piece.FirstLieunant: get_piece_img("中尉"),
    Piece.SecondLieunant: get_piece_img("少尉"),
    Piece.Plane: get_piece_img("飛行機"),
    Piece.Tank: get_piece_img("戦車"),
    Piece.Cavalry: get_piece_img("騎兵"),
    Piece.Engineer: get_piece_img("工兵"),
    Piece.Spy: get_piece_img("スパイ"),
    Piece.LandMine: get_piece_img("地雷"),
    Piece.Frag: get_piece_img("軍旗")
}

