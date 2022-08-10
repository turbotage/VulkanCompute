# CMakeList.txt : CMake project for VulkanCompute, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.14)


set(DKOMPUTE_VK_API_MAJOR_VERSION 1)
set(DKOMPUTE_VK_API_MINOR_VERSION 3)

add_subdirectory("kompute" ${CMAKE_CURRENT_BINARY_DIR}/kompute_build)


# Add source to this project's executable.
add_executable(VulkanCompute 
	"VulkanCompute.cpp" 
	"VulkanCompute.h" 
	"linalg/symm.ixx"  
	
	
	"util.ixx" 
	"linalg/glsl.ixx" 
	"linalg/linalg.ixx" 
	"linalg/solver.ixx")

set_property(TARGET VulkanCompute PROPERTY CXX_STANDARD 20)


target_link_libraries(VulkanCompute
				# Libraries from kompute
				kompute)

# TODO: Add tests and install targets if needed.
