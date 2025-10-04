#ifndef BOARD_H
#define BOARD_H

#include <vector>
#include "Action/Action.h"
#include "common/Piece.h"
#include "common/Config.h"

class Board {
private:
    std::vector<std::vector<Piece>> verticalBoard; // Example board representation
    std::vector<std::vector<Piece>> horizontalBoard; // Example board representation

    const Config config;

    std::vector<int> first_pieces;

    void set(int x, int y, Piece piece);

public:
    Board(const Config& config) : config(config) {};
    ~Board();

    Board(std::vector<int> pieces, const Config& config);
    void reset();

    void erase(int x, int y);
    void move(Action action);

    Piece get(int x, int y) const { return verticalBoard[x][y]; }

    std::vector<std::vector<Piece>>& getCrossRange(int x, int y) const;

};

#endif