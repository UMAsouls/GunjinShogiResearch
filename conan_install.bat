echo editable add GunjinShogiCore
conan editable add cpp/GunjinShogiCore
echo GunjinShogiCore Debug install
conan install cpp/GunjinShogiCore -s build_type=Debug
echo GunjinShogiCore Release install
conan install cpp/GunjinShogiCore -s build_type=Release
echo GunjinShogiResearch Debug install
conan install . -s build_type=Debug
echo GunjinShogiResearch Release install
conan install . -s build_type=Release