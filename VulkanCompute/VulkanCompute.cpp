// VulkanCompute.cpp : Defines the entry point for the application.
//
#include <kompute/Kompute.hpp>
#include <memory>
#include <optional>
#include <random>
#include <chrono>
#include <symengine/expression.h>
#include <symengine/refine.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>
#include <symengine/parser/parser.h>


import util;
import glsl;
import linalg;
import solver;

/*
void test_glsl() {
	glsl::Shader shader;
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 0, "float", "global_in"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 1, "float", "global_out"));

	static std::string code =
		R"glsl(

float mat[ndim*ndim];
float omat[ndim*ndim];
int pivot[ndim];

void main() {
	uint index = gl_GlobalInvocationID.x;
	uint startindex = index*ndim*ndim;	

	// Copy global mat into temp matrix
	for (int i = 0; i < ndim; ++i) {
		for (int j = 0; j < ndim; ++j) {
			mat[i*ndim + j] = global_in[startindex + i*ndim + j];
		}
	}

	// Perform LU decomp

	lu(mat, pivot);

	mul_unit_lower_upper_square(mat, mat, omat);

	// Copy global mat into temp matrix
	for (int i = 0; i < ndim; ++i) {
		for (int j = 0; j < ndim; ++j) {
			global_out[startindex + i*ndim + j] = omat[i*ndim + j];
		}
	}

}
)glsl";

	int ndim = 3;
	unsigned int nmat = 1000000;

	std::function<std::string()> code_func = [ndim]() -> std::string
	{
		std::string temp = code;
		util::replace_all(temp, "ndim", std::to_string(ndim));
		return temp;
	};

	glsl::Function func("main", {}, code_func,
		std::make_optional<std::vector<glsl::Function>>({
			glsl::linalg::solver::lu(3),
			glsl::linalg::mul_unit_lower_upper_square(3),
		})
	);

	shader.addFunction(func);

	std::string glsl_shader = shader.compile();

	std::cout << glsl_shader << std::endl;

	auto spirv = glsl::compileSource(glsl_shader);

	std::string spirv_str(spirv.begin(), spirv.end());

	kp::Manager mgr;


	
	//auto tensor1 = mgr.tensor({
	//	1.0, 2.0, 3.0, // work1 - row 1
	//	4.0, 5.0, 6.0, // work1 - row 2
	//	7.0, 8.0, 9.0, // work1 - row 3
	//
	//	3.0, 1.0, 1.0, // work2 - row 1
	//	1.0, 5.0, 1.0, // work2 - row 2
	//	1.0, 1.0, 9.0, // work2 - row 3
	//	});
	//
	//
	//
	//auto tensor2 = mgr.tensor({
	//	0.0, 0.0, 0.0,	// work1 - row 1
	//	0.0, 0.0, 0.0,	// work1 - row 2
	//	0.0, 0.0, 0.0,	// work1 - row 3
	//
	//	0.0, 0.0, 0.0,	// work2 - row 1
	//	0.0, 0.0, 0.0,	// work2 - row 2
	//	0.0, 0.0, 0.0,	// work2 - row 3
	//	});
	


	std::vector<float> t1(nmat * ndim);
	std::vector<float> t2(nmat * ndim);


	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0, 10);
	for (int i = 0; i < nmat * ndim; ++i) {
		t1[i] = dist(gen);
	}


	auto tensor1 = mgr.tensor(t1);
	auto tensor2 = mgr.tensor(t2);

	std::vector<std::shared_ptr<kp::Tensor>> params = { tensor1, tensor2 };

	kp::Workgroup wg({ nmat,1,1 });
	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv, wg);

	auto start = std::chrono::steady_clock::now();

	mgr.sequence()
		->record<kp::OpTensorSyncDevice>(params)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpTensorSyncLocal>(params)
		->eval();

	auto end = std::chrono::steady_clock::now();

	std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< std::endl;


	bool print = false;
	if (print) {
		std::cout << "tensor1: [ \n";
		auto v1 = tensor1->vector();
		for (int k = 0; k < 2; ++k) {
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					std::cout << v1[k * 3 * 3 + i * 3 + j] << "  ";
				}
				std::cout << "\n";
			}
		}
		std::cout << "]" << std::endl;
		std::cout << "[" << std::endl;

		std::cout << "tensor2: [ \n";
		v1 = tensor2->vector();
		for (int k = 0; k < 2; ++k) {
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					std::cout << v1[k * 3 * 3 + i * 3 + j] << "  ";
				}
				std::cout << "\n";
			}
		}
		std::cout << "]" << std::endl;
	}

}
*/

void test_expr_glsl() {
	glsl::Shader shader;
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 0, "float", "global_param"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 1, "float", "global_const"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 1, "float", "global_residuals"));

	static std::string code =
		R"glsl(

float param[nparam];
float const[ndata*nconst];

void main() {
	uint index = gl_GlobalInvocationID.x;
	uint startindex = index*ndim*ndim;	

	for (int i = 0; i < nparam; ++i) {
		param[i] = global_param[i];
	}
	
	for (int i = 0; i < 

	// Perform LU decomp

	lu(mat, pivot);

	mul_unit_lower_upper_square(mat, mat, omat);

	// Copy global mat into temp matrix
	for (int i = 0; i < ndim; ++i) {
		for (int j = 0; j < ndim; ++j) {
			global_out[startindex + i*ndim + j] = omat[i*ndim + j];
		}
	}

}
)glsl";

	int ndim = 3;
	unsigned int nmat = 1000000;

	std::function<std::string()> code_func = [ndim]() -> std::string
	{
		std::string temp = code;
		util::replace_all(temp, "ndim", std::to_string(ndim));
		return temp;
	};

	glsl::Function func("main", {}, code_func,
		std::make_optional<std::vector<glsl::Function>>({
			glsl::linalg::solver::lu(3),
			glsl::linalg::mul_unit_lower_upper_square(3),
			})
			);

	shader.addFunction(func);

	std::string glsl_shader = shader.compile();

	std::cout << glsl_shader << std::endl;

	auto spirv = glsl::compileSource(glsl_shader);

	std::string spirv_str(spirv.begin(), spirv.end());

	kp::Manager mgr;



	//auto tensor1 = mgr.tensor({
	//	1.0, 2.0, 3.0, // work1 - row 1
	//	4.0, 5.0, 6.0, // work1 - row 2
	//	7.0, 8.0, 9.0, // work1 - row 3
	//
	//	3.0, 1.0, 1.0, // work2 - row 1
	//	1.0, 5.0, 1.0, // work2 - row 2
	//	1.0, 1.0, 9.0, // work2 - row 3
	//	});
	//
	//
	//
	//auto tensor2 = mgr.tensor({
	//	0.0, 0.0, 0.0,	// work1 - row 1
	//	0.0, 0.0, 0.0,	// work1 - row 2
	//	0.0, 0.0, 0.0,	// work1 - row 3
	//
	//	0.0, 0.0, 0.0,	// work2 - row 1
	//	0.0, 0.0, 0.0,	// work2 - row 2
	//	0.0, 0.0, 0.0,	// work2 - row 3
	//	});



	std::vector<float> t1(nmat * ndim);
	std::vector<float> t2(nmat * ndim);


	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0, 10);
	for (int i = 0; i < nmat * ndim; ++i) {
		t1[i] = dist(gen);
	}


	auto tensor1 = mgr.tensor(t1);
	auto tensor2 = mgr.tensor(t2);

	std::vector<std::shared_ptr<kp::Tensor>> params = { tensor1, tensor2 };

	kp::Workgroup wg({ nmat,1,1 });
	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv, wg);

	auto start = std::chrono::steady_clock::now();

	mgr.sequence()
		->record<kp::OpTensorSyncDevice>(params)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpTensorSyncLocal>(params)
		->eval();

	auto end = std::chrono::steady_clock::now();

	std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< std::endl;


	bool print = false;
	if (print) {
		std::cout << "tensor1: [ \n";
		auto v1 = tensor1->vector();
		for (int k = 0; k < 2; ++k) {
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					std::cout << v1[k * 3 * 3 + i * 3 + j] << "  ";
				}
				std::cout << "\n";
			}
		}
		std::cout << "]" << std::endl;
		std::cout << "[" << std::endl;

		std::cout << "tensor2: [ \n";
		v1 = tensor2->vector();
		for (int k = 0; k < 2; ++k) {
			for (int i = 0; i < 3; ++i) {
				for (int j = 0; j < 3; ++j) {
					std::cout << v1[k * 3 * 3 + i * 3 + j] << "  ";
				}
				std::cout << "\n";
			}
		}
		std::cout << "]" << std::endl;
	}
}

void test_expr() {
	using SymEngine::Expression;
	auto x = SymEngine::Symbol("x");
	auto y = SymEngine::Symbol("y");
	std::map<const std::string, const SymEngine::RCP<const SymEngine::Basic>> symbol_map;
	symbol_map.emplace("x", x.rcp_from_this());
	symbol_map.emplace("y", y.rcp_from_this());

	Expression test1 = Expression(SymEngine::parse("x+x-y+(x+y)^2+cos(x)^2+sin(x)^2", true));
	Expression simplified1 = Expression(SymEngine::simplify(test1));

	std::cout << test1 << std::endl;
	std::cout << simplified1 << std::endl;

	Expression test1x = test1.diff(x.rcp_from_this_cast<SymEngine::Symbol>());
	Expression test1y = test1.diff(y.rcp_from_this_cast<SymEngine::Symbol>());

	std::cout << test1x << std::endl;
	std::cout << test1y << std::endl;

	Expression simplified1x = simplified1.diff(x.rcp_from_this_cast<SymEngine::Symbol>());
	Expression simplified1y = simplified1.diff(y.rcp_from_this_cast<SymEngine::Symbol>());

	std::cout << simplified1x << std::endl;
	std::cout << simplified1y << std::endl;
}

int main() {
	

	return 0;
}



