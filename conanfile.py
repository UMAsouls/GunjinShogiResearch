from conan import ConanFile
from conan.tools.cmake import cmake_layout

class GunjinShogiResearchConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("pybind11/3.0.1")
        self.requires("cpp_gunjinshogi_core/0.1")