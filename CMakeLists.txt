include(env.cmake OPTIONAL RESULT_VARIABLE LOCAL_ENV)
message(STATUS "Local .env.cmake: ${LOCAL_ENV}")

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

find_package(vulkan REQUIRED)

if (DEFINED GLFW_PATH)
  message(STATUS "Using GLFW path specified in .env")
  set(GLFW_INCLUDE_DIRS "${GLFW_PATH}/include")
  if (MSVC)
    set(GLFW_LIB "${GLFW_PATH}/lib-vc2022") # 2.1 Update lib-vc2019 to use same version as your visual studio
  endif()
else()
  find_package(glfw3 REQUIRED)
endif()

 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if (MSVC)
    message(STATUS "Building for Windows with MSVC")
    set(_CRT_SECURE_NO_WARNINGS 1)
    set(CURL_STATICLIB 1)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT /NODEFAULTLIB:MSVCRT")
endif()

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )

set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/external/ImGui)

include_directories(${CMAKE_SOURCE_DIR}/external/stb_image)
link_directories(${CMAKE_SOURCE_DIR}/external/stb_image)

include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends)
link_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends)

set(IMGUI_SRC
        external/ImGui/backends/imgui_impl_glfw.h
        external/ImGui/backends/imgui_impl_glfw.cpp
        external/ImGui/backends/imgui_impl_vulkan.h
        external/ImGui/backends/imgui_impl_vulkan.cpp
        external/ImGui/imgui.h
        external/ImGui/imgui.cpp
        external/ImGui/imgui_draw.cpp
        external/ImGui/imgui_tables.cpp
        external/ImGui/imgui_widgets.cpp
        # external/ImGui/imgui_demo.cpp
        )


include_directories(
    "src"
)
        

add_executable(${CMAKE_PROJECT_NAME})

target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    src/main.cu
   
    ${IMGUI_SRC}
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

target_include_directories(${PROJECT_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${Vulkan_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${GLM_PATH}
    ${IMGUI_DIR} ${IMGUI_DIR}/backends
    )

target_link_directories(${PROJECT_NAME} PUBLIC
    ${Vulkan_SDK_PATH}/Libs
    ${GLFW_LIB}
    ${IMGUI_DIR} ${IMGUI_DIR}/backends
  )

target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan)
target_link_libraries(${PROJECT_NAME} glfw3)