# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required(VERSION 3.20)

project("VulkanCompute")

add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)
# Include sub-projects.
add_subdirectory("VulkanCompute")


