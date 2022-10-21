# CMakeList.txt : CMake project for VulkanCompute, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.20)


#add_subdirectory("external/kompute" ${CMAKE_CURRENT_BINARY_DIR}/kompute_build)
# Add source to this project's executable.
target_sources(VulkanCompute PUBLIC
	"util.ixx"
	"glsl/glsl.ixx"
	"glsl/glsl.cpp"
	"glsl/function.ixx" 
	"glsl/variable.ixx"
	"glsl/func_factory.ixx" 
	"glsl/shader.ixx"
	"glsl/symbolic.ixx"
	"glsl/tensor_var.ixx" 
	"glsl/tensor_var.cpp"
	"glsl/linalg/linalg.ixx"
	"glsl/linalg/solver.ixx"
	"glsl/linalg/symm.ixx"
	"glsl/linalg/copy.ixx"
	"glsl/linalg/permute.ixx"
	"glsl/nlsq/nlsq_symbolic.ixx"
	"glsl/nlsq/nlsq.ixx"
	"glsl/lsq/lsq.ixx"
	"glsl/qmri/qmri.ixx"
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
	"expression/expr.cpp"
	"app/compute_test_app.ixx"
	"app/test_avk.ixx"
)