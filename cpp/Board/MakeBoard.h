#ifndef MAKE_BOARD_H
#define MAKE_BOARD_H

#include "Board.h"
#include "Config.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11

Board& MakeBoard(py::array_t<int> pieces, const Config& config);

#endif