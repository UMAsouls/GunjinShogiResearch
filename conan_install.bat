echo GunjinShogiResearch Debug install
conan install . -s build_type=Debug --build=missing
echo GunjinShogiResearch Release install
conan install . -s build_type=Release --build=missing