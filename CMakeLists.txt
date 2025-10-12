cmake_minimum_required(VERSION 3.18.2)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

if(CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Toolchain file used: ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(STATUS "No custom toolchain file specified.")
endif()

find_package(Python 3.13 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11)

find_package(cpp_gunjinshogi_core REQUIRED)
#add_subdirectory(cpp/GunjinShogiCore)
pybind11_add_module(GunjinShogiCore cpp/GunjinShogiCore/main.cpp)

target_link_libraries(GunjinShogiCore PRIVATE cpp_gunjinshogi_core::cpp_gunjinshogi_core)

# The install directory is the output (wheel) directory
install(TARGETS GunjinShogiCore DESTINATION .)