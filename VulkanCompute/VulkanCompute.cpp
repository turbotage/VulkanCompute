// VulkanCompute.cpp : Defines the entry point for the application.
//

#include <vector>
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
#include <string>

import util;
import glsl;
import linalg;
import solver;
import symbolic;
import expr;

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

void test_expr_res_glsl() {
	glsl::Shader shader;
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 0, "float", "global_param"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 1, "float", "global_const"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 2, "float", "global_data"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 3, "float", "global_residuals"));

	static std::string code =
R"glsl(

void main() {

	float param[nparam];
	float consts[ndata*nconst];
	float data[ndata];
	float residuals[ndata];

	uint index = gl_GlobalInvocationID.x;
	uint startindex = index*nparam;	

	// copy params
	for (int i = 0; i < nparam; ++i) {
		param[i] = global_param[startindex + i];
	}
	
	startindex = index*ndata*nconst;
	// copy consts
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < nconst; ++j) {
			consts[i*nconst + j] = global_const[startindex + i*nconst + j];
		}
	}

	startindex = index*ndata;
	// copy data
	for (int i = 0; i < ndata; ++i) {
		data[i] = global_data[startindex + i];
	}

	mod_residuals(param, consts, data, residuals);

	startindex = index*ndata;
	// copy back residuals
	for (int i = 0; i < ndata; ++i) {
		global_residuals[startindex + i] = residuals[i];
	}

}
)glsl";

	int ndata = 4;
	int nparam = 2;
	int nconst = 2;

	std::function<std::string()> code_func = [ndata, nparam, nconst]() -> std::string
	{
		std::string temp = code;
		util::replace_all(temp, "ndata", std::to_string(ndata));
		util::replace_all(temp, "nparam", std::to_string(nparam));
		util::replace_all(temp, "nconst", std::to_string(nconst));
		return temp;
	};

	glsl::Function func("main", {}, code_func,
		std::make_optional<std::vector<glsl::Function>>({
			glsl::symbolic::nlsq::nlsq_residual("mod", "x1+x2+y1-y2", ndata, nparam, nconst, true)
		})
	);

	shader.addFunction(func);
	

	std::string glsl_shader = shader.compile();

	std::cout << glsl_shader << std::endl;

	auto spirv = glsl::compileSource(glsl_shader);
	
	std::string spirv_str(spirv.begin(), spirv.end());

	kp::Manager mgr;

	auto data = mgr.tensor({
		1.0, 2.0, 3.0, 4.0, // work1
		5.0, 6.0, 7.0, 8.0	// work2
		});
	
	auto param = mgr.tensor({
		1.0, 10.0,	// work1
		5.0, 7.0	// work2
		});

	auto consts = mgr.tensor({
		0.0, 1.0,	// work1
		1.0, 1.0,
		2.0, 1.0,
		3.0, 1.0,
		4.0, 0.0,	// work2
		8.0, 0.0,
		12.0, 0.0,
		16.0, 0.0,
		});

	auto residuals = mgr.tensor({
		0.0, 0.0, 0.0, 0.0, // work1
		0.0, 0.0, 0.0, 0.0, // work2
		});

	std::vector<std::shared_ptr<kp::Tensor>> params = { param, consts, data, residuals };

	kp::Workgroup wg({ 2,1,1 });
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


	bool print = true;
	if (print) {
		auto res = residuals->vector();
		std::string printr = "residuals: \n";
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < ndata; ++j) {
				printr += std::to_string(res[i * ndata + j]) + "  ";
			}
			printr += "\n";
		}
		std::cout << printr;
	}
}

void test_expr_res_jac_hes_glsl() {
	glsl::Shader shader;
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 0, "float", "global_param"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 1, "float", "global_const"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 2, "float", "global_data"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 3, "float", "global_residuals"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 4, "float", "global_jacobian"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 5, "float", "global_hessian"));

	static std::string code =
		R"glsl(

void main() {

	float param[nparam];
	float consts[ndata*nconst];
	float data[ndata];
	float residuals[ndata];
	float jacobian[ndata*nparam];
	float hessian[nparam*nparam];

	uint index = gl_GlobalInvocationID.x;

	// copy params
	uint startindex = index*nparam;	
	for (int i = 0; i < nparam; ++i) {
		param[i] = global_param[startindex + i];
	}
	
	// copy consts
	startindex = index*ndata*nconst;
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < nconst; ++j) {
			consts[i*nconst + j] = global_const[startindex + i*nconst + j];
		}
	}

	// copy data
	startindex = index*ndata;
	for (int i = 0; i < ndata; ++i) {
		data[i] = global_data[startindex + i];
	}

	mod_lsq_residual_jacobian_hessian(param, consts, data, residuals, jacobian, hessian);

	// copy back residuals
	startindex = index*ndata;
	for (int i = 0; i < ndata; ++i) {
		global_residuals[startindex + i] = residuals[i];
	}

	
	// copy back jacobian
	startindex = index*ndata*nparam;
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < nparam; ++j) {
			global_jacobian[startindex + i*nparam + j] = jacobian[i*nparam + j];
		}
	}

	// copy back hessian
	startindex = index*nparam*nparam;
	for (int i = 0; i < nparam; ++i) {
		for (int j = 0; j < nparam; ++j) {
			global_hessian[startindex + i*nparam + j] = hessian[i*nparam + j];
		}
	}

}
)glsl";

	int ndata = 4;
	int nparam = 2;
	int nconst = 2;

	std::function<std::string()> code_func = [ndata, nparam, nconst]() -> std::string
	{
		std::string temp = code;
		util::replace_all(temp, "ndata", std::to_string(ndata));
		util::replace_all(temp, "nparam", std::to_string(nparam));
		util::replace_all(temp, "nconst", std::to_string(nconst));
		return temp;
	};

	glsl::Function func("main", {}, code_func,
		std::make_optional<std::vector<glsl::Function>>({
			glsl::symbolic::nlsq::nlsq_residual_jacobian_hessian("mod", "sin(x0)*y1*x0+cos(x1)*y0*x0", ndata, nparam, nconst, true)
			})
	);

	shader.addFunction(func);


	std::string glsl_shader = shader.compile();

	std::cout << glsl_shader << std::endl;

	auto spirv = glsl::compileSource(glsl_shader);

	std::string spirv_str(spirv.begin(), spirv.end());

	kp::Manager mgr;

	auto data = mgr.tensor({
		1.0, 2.0, 3.0, 4.0, // work1
		5.0, 6.0, 7.0, 8.0	// work2
		});

	auto param = mgr.tensor({
		1.0, 10.0,	// work1
		5.0, 7.0	// work2
		});

	auto consts = mgr.tensor({
		0.0, 1.0,	// work1
		1.0, 1.0,
		2.0, 1.0,
		3.0, 1.0,
		4.0, 1.0,	// work2
		8.0, 1.0,
		12.0, 1.0,
		16.0, 1.0,
		});

	auto residuals = mgr.tensor({
		0.0, 0.0, 0.0, 0.0, // work1
		0.0, 0.0, 0.0, 0.0, // work2
		});

	auto jacobian = mgr.tensor(std::vector<float>(2 * ndata * nparam, 0.0f));
	auto hessian = mgr.tensor(std::vector<float>(2 * nparam * nparam, 0.0f));

	std::vector<std::shared_ptr<kp::Tensor>> params = { param, consts, data, residuals, jacobian, hessian };

	kp::Workgroup wg({ 2,1,1 });
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


	bool print = true;
	if (print) {
		auto res = residuals->vector();
		auto jac = jacobian->vector();
		auto hes = hessian->vector();
		std::string printr = ""; 
		for (int i = 0; i < 2; ++i) {
			printr += "residuals: \n";
			for (int j = 0; j < ndata; ++j) {
				printr += std::to_string(res[i * ndata + j]) + "  ";
			}
			printr += "\n jacobian: \n";
			for (int j = 0; j < ndata; ++j) {
				for (int k = 0; k < nparam; ++k) {
					printr += std::to_string(jac[i*ndata*nparam + j*nparam + k]) + "  ";
				}
				printr += "\n";
			}
			printr += "\n hessian: \n";
			for (int j = 0; j < nparam; ++j) {
				for (int k = 0; k < nparam; ++k) {
					printr += std::to_string(hes[i*nparam*nparam + j*nparam + k]) + "  ";
				}
				printr += "\n";
			}
		}

		std::cout << printr;
	}
}

void test_symengine_expr() {
	using SymEngine::Expression;
	auto x = SymEngine::symbol("x1");
	auto y = SymEngine::symbol("y1");

	//std::map<const std::string, const SymEngine::RCP<const SymEngine::Basic>> symbol_map;
	//symbol_map.emplace("x1", x.rcp_from_this());
	//symbol_map.emplace("y1", y.rcp_from_this());

	auto parsed = SymEngine::parse("cosh(x1)^2+sinh(y1)^2+abs(sin(x1))", true);

	std::cout << parsed->__str__() << std::endl;

	auto parsedx = parsed->diff(x);
	auto parsedy = parsed->diff(y);

	std::cout << parsedx->__str__() << std::endl;
	std::cout << parsedy->__str__() << std::endl;

}

void test_expr() {
	std::string expr = "abs(S0*(1+FA*exp(-TI/T1)+exp(-TR/T1)))";
	std::vector<std::string> vars = { "S0", "FA", "TI", "T1", "TR" };

	expression::Expression expression(expr, vars);

	std::cout << expression.str() << std::endl;
	std::cout << expression.glsl_str() << std::endl;

	std::cout << expression.diff("S0")->str() << std::endl;
	std::cout << expression.diff("S0")->glsl_str() << std::endl;

	std::cout << expression.diff("T1")->str() << std::endl;
	std::cout << expression.diff("T1")->glsl_str() << std::endl;

	
	std::cout << expression.diff("S0")->diff("S0")->str() << std::endl;
	std::cout << expression.diff("S0")->diff("T1")->str() << std::endl;
	std::cout << expression.diff("T1")->diff("S0")->str() << std::endl;
	std::cout << expression.diff("T1")->diff("T1")->str() << std::endl;
	

}

void test_symengine() {

	auto x1 = SymEngine::symbol("x1");
	auto x2 = SymEngine::symbol("x2");
	auto x3 = SymEngine::symbol("x3");

	auto x1asm = SymEngine::contains(x1, SymEngine::reals());
	auto x2asm = SymEngine::contains(x2, SymEngine::reals());
	auto x3asm = SymEngine::contains(x3, SymEngine::reals());

	SymEngine::set_basic sb;
	sb.insert(x1asm);
	sb.insert(x2asm);
	sb.insert(x3asm);

	SymEngine::Assumptions abs(sb);

	std::cout << abs.is_real(x1) << std::endl;
	std::cout << abs.is_real(x2) << std::endl;
	std::cout << abs.is_real(x3) << std::endl;

	std::string expr = "abs(S0*(1+FA*exp(-TI/T1)+exp(-TR/T1)))";
	std::vector<std::string> vars = { "S0", "FA", "TI", "T1", "TR" };

	expression::Expression expression(expr, vars);

	auto diff1str = expression.diff("T1")->str();

	std::cout << "diff1: " << diff1str << "\n\n";

	auto diff1 = SymEngine::parse(diff1str);

	auto diff2 = diff1->diff(SymEngine::symbol("t1"));

	std::cout << "diff2: " << diff2->__str__() << "\n\n";


	std::set<std::string> args;
	std::function<void(const SymEngine::RCP<const SymEngine::Basic>&)> get_args;
	get_args = [&args, &get_args](const SymEngine::RCP<const SymEngine::Basic>& subexpr) {
		auto vec_args = subexpr->get_args();

		if (vec_args.size() == 0) {
			if (SymEngine::is_a_Number(*subexpr)) {
				return;
			}
			else if (SymEngine::is_a<SymEngine::FunctionSymbol>(*subexpr)) {
				return;
			}
			else if (SymEngine::is_a<SymEngine::Constant>(*subexpr)) {
				return;
			}

			args.insert(subexpr->__str__());
		}
		else {
			for (auto& varg : vec_args) {
				get_args(varg);
			}
		}
	};

	get_args(diff2);

	for (auto& arg : args) {
		std::cout << "arg: " << arg << std::endl;
	}

}

int main() {
	
	test_expr();

	return 0;
}



