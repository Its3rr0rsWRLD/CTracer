cmake_minimum_required(VERSION 3.18)

# Allow CUDA 12.8 because VS doesn't support it, I don't feel like downgrading CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(ctracer LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

include(FetchContent)

FetchContent_Declare(yyjson
    URL https://github.com/ibireme/yyjson/archive/refs/tags/0.10.0.tar.gz
)
FetchContent_MakeAvailable(yyjson)

FetchContent_Declare(glfw
    URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.8.tar.gz
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(mongoose
    URL https://github.com/cesanta/mongoose/archive/refs/tags/7.16.tar.gz
)
FetchContent_MakeAvailable(mongoose)

FetchContent_Declare(stb
    URL https://github.com/nothings/stb/archive/refs/heads/master.tar.gz
)
FetchContent_MakeAvailable(stb)

add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw)

add_library(mongoose_lib STATIC
    ${mongoose_SOURCE_DIR}/mongoose.c
)
target_include_directories(mongoose_lib PUBLIC ${mongoose_SOURCE_DIR})
target_compile_definitions(mongoose_lib PUBLIC MG_ENABLE_LOG=0)
if(WIN32)
    target_link_libraries(mongoose_lib PUBLIC ws2_32)
endif()

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(ctracer
    src/main.c
    src/scene.c
    src/server.c
    src/renderer.c

    src/ui.cpp

    src/cuda/ray_gen.cu
    src/cuda/intersect.cu
    src/cuda/shade.cu
    src/cuda/accumulate.cu
    src/cuda/postprocess.cu
    src/cuda/scene_upload.cu
)

set_property(TARGET ctracer PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(ctracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Xcompiler=/W3
        --expt-relaxed-constexpr
        -allow-unsupported-compiler
    >
    $<$<COMPILE_LANGUAGE:C>:/W3 /O2>
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2>
)

if(NOT MSVC)
    target_compile_options(ctracer PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -O2>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-O2>
    )
endif()

target_include_directories(ctracer PRIVATE
    src
    ${stb_SOURCE_DIR}
    ${mongoose_SOURCE_DIR}
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(ctracer PRIVATE
    yyjson
    imgui_lib
    mongoose_lib
    glfw
    OpenGL::GL
    CUDA::cudart
)

if(WIN32)
    target_link_libraries(ctracer PRIVATE ws2_32)
endif()

install(TARGETS ctracer DESTINATION bin)