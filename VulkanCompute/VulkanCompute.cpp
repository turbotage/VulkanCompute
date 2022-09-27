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

import vc;
import util;
import glsl;
import linalg;
import solver;
import symbolic;
import expr;
import symm;


void test_nlsq()
{
	using namespace glsl;
	using namespace vc;

	auto start = std::chrono::steady_clock::now();

	Shader shader;

	ui16 ndata = 21;
	ui16 nparam = 4;
	ui16 nconst = 1;

	auto residuals = std::make_shared<glsl::VectorVariable>("residuals", ndata, true);
	auto jacobian = std::make_shared<glsl::MatrixVariable>("jacobian", ndata, nparam, true);
	auto hessian = std::make_shared<glsl::MatrixVariable>("hessian", nparam, nparam, true);
	auto data = std::make_shared<glsl::VectorVariable>("data", ndata, true);
	auto params = std::make_shared<glsl::VectorVariable>("params", nparam, true);
	auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, nconst, true);
	auto lambda = std::make_shared<glsl::SimpleVariable>("lambda", "float", "2.0");
	auto step = std::make_shared<glsl::VectorVariable>("step", nparam, true);

	shader.addOutputVector(residuals, 0);
	shader.addOutputMatrix(jacobian, 1);
	shader.addOutputMatrix(hessian, 2);
	shader.addInputVector(data, 3);
	shader.addInputVector(params, 4);
	shader.addInputMatrix(consts, 5);
	shader.addOutputVector(step, 6);
	
	std::vector<std::string> vars = { "s0","f","d1","d2","b" };
	std::string expresh = "s0*(f*exp(-b*d1)+(1-f)*exp(-b*d2))";
	expression::Expression expr(expresh, vars);
	SymbolicContext context;
	context.insert_const(std::make_pair("b", 0));
	context.insert_param(std::make_pair("s0", 0));
	context.insert_param(std::make_pair("f", 1));
	context.insert_param(std::make_pair("d1", 2));
	context.insert_param(std::make_pair("d2", 3));

	shader.apply(symbolic::nlsq_residuals_jacobian_hessian_l(expr, context, ndata, nparam, nconst, true),
		nullptr,
		{ params, consts, data, lambda, residuals, jacobian, hessian });

	shader.apply(linalg::gmw81(nparam, true), nullptr, { hessian });

	auto rhs = std::make_shared<glsl::VectorVariable>("rhs", nparam, true);

	shader.apply(linalg::mul_transpose_vec(ndata, nparam, true), nullptr, { jacobian, residuals, rhs });
	shader.apply(linalg::vec_neg(nparam, true), nullptr, { rhs });

	shader.apply(linalg::ldl_solve(nparam, true), nullptr, { hessian, rhs, step });

	std::string glsl_shader = shader.compile();

	std::string gs_with_lines = util::add_line_numbers(glsl_shader);

	auto spirv = glsl::compileSource(glsl_shader, true);

	auto end = std::chrono::steady_clock::now();

	std::cout << gs_with_lines << std::endl;
	std::cout << "time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	ui32 n = 5000000;

	kp::Manager mgr;

	auto res_tensor = mgr.tensor(std::vector<float>(ndata * n));
	auto jac_tensor = mgr.tensor(std::vector<float>(ndata * nparam * n));
	auto hes_tensor = mgr.tensor(std::vector<float>(nparam * nparam * n));

	auto data_tensor = mgr.tensor(std::vector<float>(ndata * n));
	auto param_tensor = mgr.tensor(std::vector<float>(nparam * n));
	auto consts_tensor = mgr.tensor(std::vector<float>(ndata * nconst * n));
	auto step_tensor = mgr.tensor(std::vector<float>(nparam * n));

	std::vector<std::shared_ptr<kp::Tensor>> kp_params = { res_tensor, jac_tensor, hes_tensor,
		data_tensor, param_tensor, consts_tensor, step_tensor };

	kp::Workgroup wg({ (size_t)n, 1, 1 });

	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(kp_params, spirv, wg);


	auto seq = mgr.sequence()->record<kp::OpTensorSyncDevice>(kp_params)->eval();

	start = std::chrono::steady_clock::now();

	seq = seq->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpAlgoDispatch>(algo)
		->eval();
		
	end = std::chrono::steady_clock::now();
		
	seq	= seq->record<kp::OpTensorSyncLocal>(kp_params)->eval();


	std::cout << "time: " <<
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

}

void test_gmw81()
{
	using namespace glsl;
	using namespace vc;

	Shader shader;
	
	ui32 n = 1;
	ui16 ndim = 3;

	auto mat = std::make_shared<glsl::MatrixVariable>("mat", ndim, ndim, true);
	auto rhs = std::make_shared<glsl::VectorVariable>("rhs", ndim, true);
	auto sol = std::make_shared<glsl::VectorVariable>("sol", ndim, true);

	shader.addInputOutputMatrix(mat, 0);
	shader.addInputVector(rhs, 1);
	shader.addOutputVector(sol, 2);

	shader.apply(linalg::gmw81(ndim, true), nullptr, { mat });
	shader.apply(linalg::ldl_solve(ndim, true), nullptr, { mat, rhs, sol });

	std::string glsl_shader = shader.compile();
	
	std::cout << util::add_line_numbers(glsl_shader) << std::endl;

	auto spirv = glsl::compileSource(glsl_shader);

	kp::Manager mgr;

	auto mat_tensor = mgr.tensor({
		3,2,1,
		2,4,2,
		1,2,5
		});

	auto rhs_tensor = mgr.tensor({
		1,1,1
		});

	auto sol_tensor = mgr.tensor({
		0,0,0
		});

	std::vector<std::shared_ptr<kp::Tensor>> params = { mat_tensor, rhs_tensor, sol_tensor };

	kp::Workgroup wg({ (size_t)n,1,1 });

	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv, wg);


	auto seq = mgr.sequence()
		->record<kp::OpTensorSyncDevice>(params)->eval();

	auto start = std::chrono::steady_clock::now();

	seq = seq->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpAlgoDispatch>(algo)
		->eval();
		
	auto end = std::chrono::steady_clock::now();
	
	seq = seq->record<kp::OpTensorSyncLocal>(params)
		->eval();

	std::cout << "time: " << 
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	bool print = true;
	if (print) {
		auto res = mat_tensor->vector();
		std::string printr = "decomp: \n";
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < ndim; ++j) {
				for (int k = 0; k < ndim; ++k) {
					printr += std::to_string(res[i * ndim * ndim + j * ndim + k]) + "  ";
				}
				printr += "\n";
			}
			printr += "\n\n";
		}
		res = sol_tensor->vector();
		printr += "sol: \n";
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < ndim; ++j) {
				printr += std::to_string(res[i * ndim + j]) + "  ";
			}
			printr += "\n\n";
		}
		std::cout << printr;
	}

}

void test_backward() {
	using namespace glsl;
	using namespace vc;

	Shader shader;

	ui32 n = 1;
	ui16 ndim = 3;

	auto mat = std::make_shared<glsl::MatrixVariable>("mat", ndim, ndim, true);
	auto rhs = std::make_shared<glsl::VectorVariable>("rhs", ndim, true);
	auto sol = std::make_shared<glsl::VectorVariable>("sol", ndim, true);

	shader.addInputOutputMatrix(mat, 0);
	shader.addInputVector(rhs, 1);
	shader.addOutputVector(sol, 2);

	//shader.apply(linalg::backward_subs_unit_t(ndim, true), nullptr, { mat, rhs, sol });
	std::string after_copying_from =
		R"glsl(
	sol[0] = rhs[0];
	sol[1] = rhs[1];
	sol[2] = rhs[2];
)glsl";
	//shader.setAfterCopyingFrom(after_copying_from);

	std::string glsl_shader = shader.compile();

	auto spirv = glsl::compileSource(glsl_shader);
	
	std::cout << util::add_line_numbers(glsl_shader) << std::endl;

	kp::Manager mgr;

	auto mat_tensor = mgr.tensor({
		3,2,1,
		2,4,2,
		1,2,5
		});

	auto rhs_tensor = mgr.tensor({
		1,1,1
		});

	auto sol_tensor = mgr.tensor({
		0,0,0
		});

	std::vector<std::shared_ptr<kp::Tensor>> params = { mat_tensor, rhs_tensor, sol_tensor };

	kp::Workgroup wg({ (size_t)n, 1,1 });

	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv, wg);

	auto start = std::chrono::steady_clock::now();

	mgr.sequence()
		->record<kp::OpTensorSyncDevice>(params)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpTensorSyncLocal>(params)
		->eval();

	auto end = std::chrono::steady_clock::now();

	std::cout << "time: " <<
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	bool print = true;
	if (print) {
		auto res = mat_tensor->vector();
		std::string printr = "decomp: \n";
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < ndim; ++j) {
				for (int k = 0; k < ndim; ++k) {
					printr += std::to_string(res[i * ndim * ndim + j * ndim + k]) + "  ";
				}
				printr += "\n";
			}
			printr += "\n\n";
		}
		res = sol_tensor->vector();
		printr += "sol: \n";
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < ndim; ++j) {
				printr += std::to_string(res[i * ndim + j]) + "  ";
			}
			printr += "\n\n";
		}
		std::cout << printr;
	}
}

int main() {
	
	test_nlsq();

	return 0;
}



