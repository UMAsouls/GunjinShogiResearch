cmake_minimum_required(VERSION 3.15)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

add_subdirectory(cpp/GunjinShogiCore)

pybind11_add_module(JudgeBoard cpp/GunjinShogiCore/main.cpp)

target_link_libraries(JudgeBoard PRIVATE pybind11::headers)

# The install directory is the output (wheel) directory
install(TARGETS JudgeBoard DESTINATION GunjinShogiCore)