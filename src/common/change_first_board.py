from src.const import \
    BOARD_SHAPE, BOARD_SHAPE_INT, ENTRY_HEIGHT, ENTRY_POS, GOAL_POS, \
    PIECE_LIMIT, PIECE_DICT, Piece

import torch


PIECE_TENSOR_DICT = torch.as_tensor(PIECE_DICT, dtype=torch.int32)

def make_int_board(pieces1: torch.Tensor, pieces2: torch.Tensor) -> tuple[list[int], list[int]]:
    board = [[0 for _ in range(BOARD_SHAPE[0])] for _ in range(BOARD_SHAPE[1])]
    
    i = PIECE_LIMIT-1
    for y in range(ENTRY_HEIGHT):
        x = 0
        while x < BOARD_SHAPE[0]:
           if(x in GOAL_POS and y == 0): x += len(GOAL_POS)-1 
           board[y][x] = PIECE_DICT[pieces2[i].item()]
           i -= 1
           x += 1
           
    i = 0
    for y in range(ENTRY_HEIGHT+1, BOARD_SHAPE[1]):
        x = 0
        while x < BOARD_SHAPE[0]: 
           board[y][x] = PIECE_DICT[pieces1[i].item()]
           i += 1
           if(x in GOAL_POS and y == BOARD_SHAPE[1]-1): x += len(GOAL_POS)-1 
           x += 1
           
    return board

#tensorのboard作成関数
def make_tensor_board(pieces: torch.Tensor) -> torch.Tensor:
    tensor_board = torch.full((BOARD_SHAPE_INT,), int(Piece.Enemy), dtype=torch.int32)
    
    #entryとwallの部分入力
    wall_area = torch.arange(start=BOARD_SHAPE[0]*ENTRY_HEIGHT,end=BOARD_SHAPE[0]*(ENTRY_HEIGHT+1))
    tensor_board[wall_area] = int(Piece.Wall)
    entry = torch.as_tensor([BOARD_SHAPE[0]*ENTRY_HEIGHT+i for i in ENTRY_POS])
    tensor_board[entry] = int(Piece.Entry)
    
    #司令部の空白部分作成
    space_pos = pieces.shape[0] - BOARD_SHAPE[0] + GOAL_POS[-1]
    real_pieces = torch.cat((pieces[:space_pos+1],torch.tensor([int(0)]),pieces[space_pos+1:]))
    
    #tensorの後半部分に代入
    area = torch.arange(start=BOARD_SHAPE_INT-real_pieces.shape[0], end=BOARD_SHAPE_INT)
    tensor_board[area] = PIECE_TENSOR_DICT[real_pieces]
    
    #空白部分に空白を代入（dictにspaceが無いため）
    tensor_board[BOARD_SHAPE_INT - BOARD_SHAPE[0] + GOAL_POS[-1]] = int(Piece.Space)
    
    return tensor_board