cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

if (UNIX)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()


if(WIN32)
message(STATUS "Project is building for Windows")
set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_PATH}")
set(CMAKE_CUDA_COMPILER "$ENV{CUDA_PATH}/bin/nvcc.exe")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
endif()

project(RTWeekend_CUDA LANGUAGES C CXX CUDA)

message("Project name: " ${CMAKE_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if (MSVC)
    message(STATUS "Building for Windows with MSVC")
    set(_CRT_SECURE_NO_WARNINGS 1)
    set(CURL_STATICLIB 1)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif()

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )

include_directories(
    "src"
)

include_directories(${CMAKE_SOURCE_DIR}/external/stb_image)
link_directories(${CMAKE_SOURCE_DIR}/external/stb_image)

add_executable(${CMAKE_PROJECT_NAME})

target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    src/main.cu
)

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:C>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr>
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61 75 87) # Running on GTX 1080 Ti, GTX 1660 Ti, and Jetson AGX Orin

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _DEBUG)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()



