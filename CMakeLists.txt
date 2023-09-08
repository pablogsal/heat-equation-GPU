cmake_minimum_required(VERSION 3.12)
project(HeatEquation)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

enable_language(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
add_executable(heat_equation main.cu)

set_property(TARGET heat_equation PROPERTY CUDA_ARCHITECTURES OFF)

target_compile_options(heat_equation PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>")
target_compile_options(heat_equation PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
target_compile_options(heat_equation PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")
target_compile_options(heat_equation PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3>")
target_link_libraries(heat_equation ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})
