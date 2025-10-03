#ifndef JUDGE_BOARD_H
#define JUDGE_BOARD_H

#include <vector>
#include <tuple>
#include <map>

#include "cpp/common/Piece.h"
#include "cpp/common/Player.h"
#include "cpp/common/Config.h"
#include "cpp/common/JudgeFrag.h"

#include "cpp/Board/Board.h"
#include "cpp/JudgeTable/judge_table.h"

class JudgeBoard {
private:
    Board board_p1;
    Board board_p2;

    Config config;

    JudgeTable judgeTable;

    std::map<Player, Board*> playerBoards;

    std::pair<int,int> get_reflect_pos(int x, int y) const {
        std::pair<int,int> shape = config.getBoardShape();
        return {shape.first - 1 - x, shape.second - 1 - y};
    }

    Board& getBoard(Player player) const {
        return *playerBoards.at(player);
    }

public:
    JudgeBoard(Board& p1, Board& p2, Config& c, JudgeTable& j) : board_p1(p1), board_p2(p2), config(c), judgeTable(j) {
        playerBoards[Player::PLAYER_ONE] = &board_p1;
        playerBoards[Player::PLAYER_TWO] = &board_p2;
    }
    ~JudgeBoard() {};

    void reset();
    void erase(int x, int y, Player player);
    void move(int fromX, int fromY, int toX, int toY, Player player);

    JudgeFrag getJudge(int fromX, int fromY, int toX, int toY, Player player);

    bool isGameOver(Player player) const;
    
    std::vector<int>& getLegalMoves(Player player) const;

    // Add more methods as needed
};

#endif // JUDGE_BOARD_H