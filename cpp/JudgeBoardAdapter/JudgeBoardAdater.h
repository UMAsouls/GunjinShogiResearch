#ifndef JUDGE_BOARD_A_H
#define JUDGE_BOARD_A_H

#include "cpp/JudgeBoard/judge_board.h"
#include "cpp/common/Config.h"
#include "Player.h"
#include "JudgeFrag.h"
#include "Action/Action.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class JudgeBoardAdapter {
private:
    JudgeBoard judgeBoard;
    Config config;

public:
    JudgeBoardAdapter(JudgeBoard& jb, Config& c) : judgeBoard(jb), config(c) {}
    ~JudgeBoardAdapter() {}

    void reset() { judgeBoard.reset(); } ;
    void erase(int x, int y, Player player) { judgeBoard.erase(x,y,player); };
    void move(int action, Player player);

    JudgeFrag getJudge(int action, Player player) { return judgeBoard.getJudge(GetActionFromInt(action), player); };

    bool isGameOver(Player player) const { return judgeBoard.isGameOver(player); };

    py::array_t<int>& getLegalMoves(Player player) const;
}

#endif