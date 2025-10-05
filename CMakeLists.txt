cmake_minimum_required(VERSION 3.18.2)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")


add_subdirectory(cpp/GunjinShogiCore)

pybind11_add_module(GunjinShogiCore cpp/GunjinShogiCore/main.cpp)

target_link_libraries(GunjinShogiCore PRIVATE CppGunjinShogiCore)

# The install directory is the output (wheel) directory
install(TARGETS GunjinShogiCore DESTINATION .)