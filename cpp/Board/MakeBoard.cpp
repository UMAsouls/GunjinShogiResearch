#include "MakeBoard.h"
#include "Board.h"

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

Board& MakeBoard(py::array_t<int> pieces, const Config& config) {
    int size = pieces.size();
    std::vector<int> p(size);

    for(int i = 0; i < size; i++) {
        p[i] = pieces.at(i);
    }

    return Board(p, const);
}

