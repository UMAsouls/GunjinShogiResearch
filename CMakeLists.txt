cmake_minimum_required(VERSION 3.15)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

add_subdirectory(cpp/json)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(JudgeBoard cpp/main.cpp)

target_link_libraries(JudgeBoard PRIVATE pybind11::headers)

# This is passing in the version as a define just as an example
target_compile_definitions(JudgeBoard PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS JudgeBoard DESTINATION GunjinShogiCore)