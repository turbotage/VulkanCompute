# CMakeList.txt : CMake project for VulkanCompute, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.14)

set(DKOMPUTE_VK_API_MAJOR_VERSION 1)
set(DKOMPUTE_VK_API_MINOR_VERSION 3)

add_subdirectory("external/kompute" ${CMAKE_CURRENT_BINARY_DIR}/kompute_build)
# Add source to this project's executable.
add_executable(VulkanCompute 
	"VulkanCompute.cpp"
	"util.ixx"
	"glsl/glsl.ixx"
	"glsl/glsl.cpp"
	"glsl/linalg/linalg.ixx"
	"glsl/linalg/solver.ixx"
	"glsl/linalg/symm.ixx"
	"glsl/linalg/copy.ixx"
	"glsl/linalg/permute.ixx"
	"glsl/nlsq/symbolic.ixx"
	"glsl/nlsq/nlsq.ixx"
	"vc.ixx"
	"expression/parser/token.ixx" 
	"expression/parser/lexer.ixx" 
	"expression/parser/defaultexp.ixx"
	"expression/parser/shunter.ixx"
	"expression/algebra/token_algebra.ixx"
	"expression/algebra/unary/neg.ixx"
	"expression/algebra/unary/trig.ixx"
	"expression/algebra/unary/unary.ixx"
	"expression/algebra/binary/add.ixx"
	"expression/algebra/binary/div.ixx"
	"expression/algebra/binary/mul.ixx"
	"expression/algebra/binary/pow.ixx"
	"expression/algebra/binary/sub.ixx"
	"expression/expr.ixx" 
	"expression/expr.cpp" )

set_property(TARGET VulkanCompute PROPERTY CXX_STANDARD 20)

target_link_libraries(VulkanCompute
				# Libraries from kompute
				kompute)

find_package(SymEngine CONFIG REQUIRED)
message(${SYMENGINE_LIBRARIES})
target_link_libraries(VulkanCompute symengine)

# TODO: Add tests and install targets if needed.
