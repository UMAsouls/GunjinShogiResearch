from src.const import \
    PIECE_DICT, Piece

from src.common.Config import Config

import torch

import numpy as np

PIECE_TENSOR_DICT = np.array(PIECE_DICT, dtype=np.int32)

def make_int_board(pieces1: np.ndarray, pieces2: np.ndarray) -> tuple[list[int], list[int]]:
    BOARD_SHAPE = Config.board_shape
    ENTRY_HEIGHT = Config.entry_height
    GOAL_POS = Config.goal_pos
    PIECE_LIMIT = Config.piece_limit
    

    board = [[0 for _ in range(BOARD_SHAPE[0])] for _ in range(BOARD_SHAPE[1])]
    
    i = PIECE_LIMIT-1
    for y in range(ENTRY_HEIGHT):
        x = 0
        while x < BOARD_SHAPE[0]:
           if(x in GOAL_POS and y == 0): x += len(GOAL_POS)-1 
           board[y][x] = PIECE_DICT[pieces2[i]]
           i -= 1
           x += 1
           
    i = 0
    for y in range(ENTRY_HEIGHT+1, BOARD_SHAPE[1]):
        x = 0
        while x < BOARD_SHAPE[0]: 
           board[y][x] = PIECE_DICT[pieces1[i]]
           i += 1
           if(x in GOAL_POS and y == BOARD_SHAPE[1]-1): x += len(GOAL_POS)-1 
           x += 1
           
    return board

#ndarrayのboard作成関数
def make_ndarray_board(pieces: np.ndarray) -> np.ndarray:
    BOARD_SHAPE = Config.board_shape
    BOARD_SHAPE_INT = Config.board_shape_int
    ENTRY_HEIGHT = Config.entry_height
    ENTRY_POS = Config.entry_pos
    GOAL_POS = Config.goal_pos

    tensor_board = np.full((BOARD_SHAPE_INT,), int(Piece.Enemy), dtype=np.int32)
    
    #entryとwallの部分入力
    wall_area = np.arange(start=BOARD_SHAPE[0]*ENTRY_HEIGHT, stop=BOARD_SHAPE[0]*(ENTRY_HEIGHT+1))
    tensor_board[wall_area] = int(Piece.Wall)
    entry = np.array([BOARD_SHAPE[0]*ENTRY_HEIGHT+i for i in ENTRY_POS])
    tensor_board[entry] = int(Piece.Entry)
    
    #敵側の司令部空白作成
    tensor_board[GOAL_POS[0]] = 0
    
    #司令部の空白部分作成
    space_pos = pieces.shape[0] - BOARD_SHAPE[0] + GOAL_POS[-1]
    real_pieces = np.concatenate((pieces[:space_pos+1],np.array([int(0)]),pieces[space_pos+1:]))
    
    #tensorの後半部分に代入
    area = np.arange(start=BOARD_SHAPE_INT-real_pieces.shape[0], stop=BOARD_SHAPE_INT)
    tensor_board[area] = PIECE_TENSOR_DICT[real_pieces]
    
    #空白部分に空白を代入（dictにspaceが無いため）
    tensor_board[BOARD_SHAPE_INT - BOARD_SHAPE[0] + GOAL_POS[-1]] = int(Piece.Space)
    
    return tensor_board