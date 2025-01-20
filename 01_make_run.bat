rmdir /S /Q build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -S .. -B .
@REM cmake -S .. -B .
cd ..
"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe" build\RTWeekend_CUDA.vcxproj
bin\Debug\RTWeekend_CUDA.exe