echo move to CppCore
cd cpp/GunjinShogiCore
echo core install
call conan_install.bat
echo create
conan create .
echo return
cd ../..
echo install
call conan_install.bat
echo pip install
pip install .