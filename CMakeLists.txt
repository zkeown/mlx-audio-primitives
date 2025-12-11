cmake_minimum_required(VERSION 3.25)
project(mlx_audio_primitives LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

# Find Python
find_package(
    Python 3.10
    COMPONENTS Interpreter Development.Module
    REQUIRED
)

# Find MLX via Python's mlx package
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MLX_ROOT
    RESULT_VARIABLE MLX_RESULT
    ERROR_VARIABLE MLX_ERROR
)
if(NOT MLX_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Failed to find MLX cmake directory. Is MLX installed?\n"
        "Install with: pip install mlx>=0.30.0\n"
        "Error: ${MLX_ERROR}"
    )
endif()

find_package(MLX CONFIG REQUIRED)

# Build the extension
add_subdirectory(csrc)
