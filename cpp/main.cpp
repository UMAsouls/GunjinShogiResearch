#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(JudgeBoard, m) {
    m.doc() = "盤面のルール管理"; // optional module docstring
}