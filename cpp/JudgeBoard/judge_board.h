#ifndef JUDGE_BOARD_H
#define JUDGE_BOARD_H

#include <vector>
#include <map>

#include "cpp/common/Piece.h"
#include "cpp/common/Player.h"
#include "cpp/common/Config.h"

#include "cpp/Board/Board.h"

class JudgeBoard {
private:
    Board board_p1;
    Board board_p2;

    Config config;

    std::map<Player, Board*> playerBoards;

    Board& getBoard(Player player) const {
        return *playerBoards.at(player);
    }

public:
    JudgeBoard(Board& p1, Board& p2, Config& c) : board_p1(p1), board_p2(p2), config(c) {
        playerBoards[Player::PLAYER_ONE] = &board_p1;
        playerBoards[Player::PLAYER_TWO] = &board_p2;
    }
    ~JudgeBoard() {};

    void reset();
    void erase(int x, int y, Player player);
    void move(int fromX, int fromY, int toX, int toY, Player player);

    bool isGameOver() const;
    
    std::vector<int>& getLegalMoves(Player player) const;

    // Add more methods as needed
};

#endif // JUDGE_BOARD_H