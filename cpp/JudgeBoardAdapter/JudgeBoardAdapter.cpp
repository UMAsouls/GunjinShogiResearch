#include "JudgeBoardAdater.h"

#include <vector>


py::array_t<int>& JudgeBoardAdapter::getLegalMoves(Player player) const {
    std::vector<int> m = judgeBoard.getLegalMoves(player);

    py::array_t<int> moves(m.size());
    for(int i = 0; i < m.size(); i++) moves[i] = m[i];

    return moves;
};