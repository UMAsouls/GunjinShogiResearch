#include <map>
#include <vector>
#include <tuple>
#include <algorithm>

#include "cpp/common/Piece.h"
#include "cpp/Board/Board.h"

void Board::set(int x, int y, Piece piece) {
    verticalBoard[x][y] = piece;
    horizontalBoard[y][x] = piece;
}

Board::Board(std::vector<int> pieces, const Config& config) : config(config) {
    first_pieces = pieces;
    reset();

}

void Board::erase(int x, int y) {
    set(x, y, Piece::Space);
}

void Board::move(int fromX, int fromY, int toX, int toY) {
    Piece piece = get(fromX, fromY);
    erase(fromX, fromY);
    set(toX, toY, piece);
}

void Board::reset() {
    int entry_height = config.getEntryHeight();
    std::vector<int> entry_pos = config.getEntryPos();
    std::sort(entry_pos.begin(), entry_pos.end());

    int goal_height = config.getGoalHeight();
    std::vector<int> goal_pos = config.getGoalPos();
    std::sort(goal_pos.begin(), goal_pos.end());

    int shape_x = config.getBoardShape().first;
    int shape_y = config.getBoardShape().second;

    int piece_limit = config.getPieceLimit();

    this->verticalBoard = std::vector<std::vector<Piece>>(shape_x, std::vector<Piece>(shape_y));
    this->horizontalBoard = std::vector<std::vector<Piece>>(shape_y, std::vector<Piece>(shape_x));


    int f_goal = goal_pos[0];
    for (int i = 0; i < entry_height; i++) {
        for (int j = 0; j < shape_x; j++) {
            if(j == f_goal && i == goal_height) {
                j += goal_pos.size() - 1;
            }

            set(j, i, Piece::Entry);
        }
    }

    for (int i = 0; i < shape_x; i++) {
        if(std::lower_bound(entry_pos.begin(), entry_pos.end(), i) != entry_pos.end()) {
            set(i, entry_height, Piece::Entry);
        } else {
            set(i, entry_height, Piece::Wall);
        }
    }

    int fliped_goal_height = shape_y - goal_height - 1;
    int idx = 0;
    for (int i = entry_height + 1; i < shape_y; i++) {
        for (int j = 0; j < shape_x; j++) {
            set(j, i, config.getPiece(first_pieces[idx++]));

            if(j == f_goal && i == goal_height) {
                j += goal_pos.size() - 1;
            }
        }
    }
}

std::vector<std::vector<Piece>>& Board::getCrossRange(int x, int y) const {
    std::vector<std::vector<Piece>> ranges(4);

    ranges[0] = std::vector<Piece>(horizontalBoard[y].begin() + x + 1, horizontalBoard[y].end());
    ranges[1] = std::vector<Piece>(horizontalBoard[y].begin(), horizontalBoard[y].begin() + x);
    std::reverse(ranges[1].begin(), ranges[1].end());
    ranges[2] = std::vector<Piece>(verticalBoard[x].begin() + y + 1, verticalBoard[x].end());
    ranges[3] = std::vector<Piece>(verticalBoard[x].begin(), verticalBoard[x].begin() + y);
    std::reverse(ranges[3].begin(), ranges[3].end());

    return ranges;
}