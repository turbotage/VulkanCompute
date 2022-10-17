# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required(VERSION 3.20)

project("VulkanCompute")

add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)
# Include sub-projects.



add_executable(VulkanCompute "src/VulkanCompute.cpp")
add_subdirectory("src")

set_property(TARGET VulkanCompute PROPERTY CXX_STANDARD 20)

include(cmake/avk_toolkit.cmake)
target_include_directories(VulkanCompute PUBLIC Auto_Vk_Toolkit)
target_link_libraries(VulkanCompute PUBLIC Auto_Vk_Toolkit)


set(DKOMPUTE_VK_API_MAJOR_VERSION 1)
set(DKOMPUTE_VK_API_MINOR_VERSION 3)
include(cmake/kompute.cmake)
target_include_directories(VulkanCompute PUBLIC kompute)
target_link_libraries(VulkanCompute PUBLIC kompute)


find_package(SymEngine CONFIG REQUIRED)
message(${SYMENGINE_LIBRARIES})
target_include_directories(VulkanCompute PUBLIC symengine)
target_link_libraries(VulkanCompute PUBLIC symengine)

if (MSVC)
	target_compile_options(VulkanCompute PRIVATE "/Zc:__cplusplus")
endif()
target_precompile_headers(VulkanCompute PRIVATE 
	[[src/pch.hpp]])

add_custom_command(TARGET VulkanCompute PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		${CMAKE_SOURCE_DIR}/src/python/export/ $<TARGET_FILE_DIR:VulkanCompute>)

# TODO: Add tests and install targets if needed.
