// VulkanCompute.cpp : Defines the entry point for the application.
//

#include <kompute/Kompute.hpp>
#include <symengine/expression.h>
#include <symengine/refine.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>
#include <symengine/parser/parser.h>

import <vector>;
import <memory>;
import <optional>;
import <random>;
import <chrono>;
import <string>;

import vc;
import util;
import glsl;
import linalg;
import solver;
import symbolic;
import expr;
import symm;

void test_copying()
{
	using namespace glsl;
	using namespace vc;

	constexpr ui32 n = 7;

	AutogenShader shader;
	
	kp::Manager mgr;


	std::vector<int32_t> converges = { 1, 0, 1, 0, 1, 0, 0 };
	std::vector<int32_t> convergent_index(n, 0);
	std::vector<int32_t> num_convergent = { 0 };

	auto converges_tensor = mgr.tensor(converges.data(), converges.size(), sizeof(int32_t), kp::Tensor::TensorDataTypes::eInt);
	auto convergent_index_tensor = mgr.tensor(convergent_index.data(), convergent_index.size(), sizeof(int32_t), kp::Tensor::TensorDataTypes::eInt);
	auto num_convergent_tensor = mgr.tensor(num_convergent.data(), num_convergent.size(), sizeof(int32_t), kp::Tensor::TensorDataTypes::eInt);

	auto converges_bind = std::make_unique<BufferBinding>(0, "int", "converges");
	shader.addBinding(std::move(converges_bind));
	
	auto convergent_index_bind = std::make_unique<BufferBinding>(1, "int", "convergent_index");
	shader.addBinding(std::move(convergent_index_bind));

	auto num_convergent_bind = std::make_unique<BufferBinding>(2, "int", "num_convergent");
	shader.addBinding(std::move(num_convergent_bind));

	
	shader.setBeforeCopyingFrom(
R"glsl(
	uint index = gl_GlobalInvocationID.x;
	int conv = global_converges[index];
	if (conv == 1) {
		int converges_index = atomicAdd(global_num_convergent[0], 1);
		global_convergent_index[index] = converges_index;
	}
	else if (conv == 0) {
		global_convergent_index[index] = -1;
	}
	else {
		global_convergent_index[index] = -2;
	}
)glsl");
	
	std::string glsl_shader = shader.compile();
	std::string gs_with_lines = util::add_line_numbers(glsl_shader);

	std::cout << gs_with_lines << std::endl;

	auto spirv = glsl::compileSource(glsl_shader, true);


	kp::Workgroup wg({ (size_t)n, 1, 1 });
	std::vector<std::shared_ptr<kp::Tensor>> kp_params = 
		{ converges_tensor, convergent_index_tensor, num_convergent_tensor };

	auto algo = mgr.algorithm(kp_params, spirv, wg);

	mgr.sequence()->record<kp::OpTensorSyncDevice>(kp_params)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpTensorSyncLocal>(kp_params)->eval();

	std::string ret = "\nconverges: ";
	for (auto c : converges_tensor->vector<int32_t>()) {
		ret += std::to_string(c) + " ";
	}

	ret += "\n\nconvergent_index: ";
	for (auto c : convergent_index_tensor->vector<int32_t>()) {
		ret += std::to_string(c) + " ";
	}

	ret += "\n\nnum_convergent: ";
	for (auto c : num_convergent_tensor->vector<int32_t>()) {
		ret += std::to_string(c) + " ";
	}

	std::cout << ret << std::endl;

}

void test_nlsq()
{
	using namespace glsl;
	using namespace vc;

	auto start = std::chrono::steady_clock::now();

	AutogenShader shader;

	ui16 ndata = 21;
	ui16 nparam = 4;
	ui16 nconst = 1;

	auto residuals = std::make_shared<glsl::VectorVariable>("residuals", ndata, ShaderVariableType::FLOAT);
	auto jacobian = std::make_shared<glsl::MatrixVariable>("jacobian", ndata, nparam, ShaderVariableType::FLOAT);
	auto hessian = std::make_shared<glsl::MatrixVariable>("hessian", nparam, nparam, ShaderVariableType::FLOAT);
	auto data = std::make_shared<glsl::VectorVariable>("data", ndata, ShaderVariableType::FLOAT);
	auto params = std::make_shared<glsl::VectorVariable>("params", nparam, ShaderVariableType::FLOAT);
	auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, nconst, ShaderVariableType::FLOAT);
	auto lambda = std::make_shared<glsl::TextedVariable>("lambda", "float", "2.0");
	auto step = std::make_shared<glsl::VectorVariable>("step", nparam, ShaderVariableType::FLOAT);
	auto add_test = std::make_shared<glsl::VectorVariable>("add_test", 1, ShaderVariableType::FLOAT);


	shader.addOutputVector(residuals, 0);
	shader.addOutputMatrix(jacobian, 1);
	shader.addOutputMatrix(hessian, 2);
	shader.addInputVector(data, 3);
	shader.addInputVector(params, 4);
	shader.addInputMatrix(consts, 5);
	shader.addOutputVector(step, 6);

	shader.addBinding(std::make_unique<glsl::BufferBinding>(7, "float", "add_test"));
	
	std::vector<std::string> vars = { "s0","f","d1","d2","b" };
	std::string expresh = "s0*(f*exp(-b*d1)+(1-f)*exp(-b*d2))";
	expression::Expression expr(expresh, vars);
	SymbolicContext context;
	context.insert_const(std::make_pair("b", 0));
	context.insert_param(std::make_pair("s0", 0));
	context.insert_param(std::make_pair("f", 1));
	context.insert_param(std::make_pair("d1", 2));
	context.insert_param(std::make_pair("d2", 3));

	shader.apply(nlsq::nlsq_residuals_jacobian_hessian_l(expr, context, ndata, nparam, nconst, true),
		nullptr,
		{ params, consts, data, lambda, residuals, jacobian, hessian });

	shader.apply(linalg::gmw81(nparam, true), nullptr, { hessian });

	auto rhs = std::make_shared<glsl::VectorVariable>("rhs", nparam, ShaderVariableType::FLOAT);

	shader.apply(linalg::mul_transpose_vec(ndata, nparam, true), nullptr, { jacobian, residuals, rhs });
	shader.apply(linalg::vec_neg(nparam, true), nullptr, { rhs });

	shader.apply(linalg::ldl_solve(nparam, true), nullptr, { hessian, rhs, step });

	shader.setBeforeCopyingBack(
R"glsl(
	global_add_test[gl_GlobalInvocationID.x] += float(1.0);
)glsl"
);

	std::string glsl_shader = shader.compile();

	std::string gs_with_lines = util::add_line_numbers(glsl_shader);

	std::vector<uint32_t> spirv;
	try {
		spirv = glsl::compileSource(glsl_shader, true);
	}
	catch (std::exception& e) {
		std::cout << gs_with_lines << std::endl;
		std::cout << e.what() << std::endl;
		throw e;
	}

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

	auto add_test_tensor = mgr.tensor(std::vector<float>(n, 0));

	std::vector<std::shared_ptr<kp::Tensor>> kp_params = { res_tensor, jac_tensor, hes_tensor,
		data_tensor, param_tensor, consts_tensor, step_tensor, add_test_tensor };

	kp::Workgroup wg({ (size_t)n, 1, 1 });

	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(kp_params, spirv, wg);

	int runs = 2;
	auto seq = mgr.sequence()->record<kp::OpTensorSyncDevice>(kp_params)->eval();

	start = std::chrono::steady_clock::now();

	for (int i = 0; i < runs; ++i) {
		seq->record<kp::OpAlgoDispatch>(algo);
		std::cout << "run: " << i << std::endl;
	}
	seq->eval();

	end = std::chrono::steady_clock::now();
		
	seq->record<kp::OpTensorSyncLocal>(kp_params)->eval();

	std::cout << "time: " <<
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	auto atv = add_test_tensor->vector();
	std::cout << "atv size: " << atv.size() << std::endl;
	std::cout << "first:" << atv.front() << std::endl;
	std::cout << "middle" << atv.at(std::max((uint32_t)atv.size() / 2, (uint32_t)(atv.size() - 1))) << std::endl;
	std::cout << "back: " << atv.back() << std::endl;
}

void test_nlsq_step()
{
	using namespace glsl;
	using namespace vc;

	auto start = std::chrono::steady_clock::now();

	AutogenShader shader;

	ui16 ndata = 21;
	ui16 nparam = 4;
	ui16 nconst = 1;

	auto residuals = std::make_shared<glsl::VectorVariable>("residuals", ndata, ShaderVariableType::FLOAT);
	auto jacobian = std::make_shared<glsl::MatrixVariable>("jacobian", ndata, nparam, ShaderVariableType::FLOAT);
	auto hessian = std::make_shared<glsl::MatrixVariable>("hessian", nparam, nparam, ShaderVariableType::FLOAT);
	auto data = std::make_shared<glsl::VectorVariable>("data", ndata, ShaderVariableType::FLOAT);
	auto params = std::make_shared<glsl::VectorVariable>("params", nparam, ShaderVariableType::FLOAT);
	auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, nconst, ShaderVariableType::FLOAT);
	auto lambda = std::make_shared<glsl::TextedVariable>("lambda", "float", "2.0");
	auto step = std::make_shared<glsl::VectorVariable>("step", nparam, ShaderVariableType::FLOAT);
	auto add_test = std::make_shared<glsl::VectorVariable>("converges", 1, ShaderVariableType::FLOAT);



}

void test_gmw81()
{
	using namespace glsl;
	using namespace vc;

	AutogenShader shader;
	
	ui32 n = 1;
	ui16 ndim = 3;

	auto mat = std::make_shared<glsl::MatrixVariable>("mat", ndim, ndim, ShaderVariableType::FLOAT);
	auto rhs = std::make_shared<glsl::VectorVariable>("rhs", ndim, ShaderVariableType::FLOAT);
	auto sol = std::make_shared<glsl::VectorVariable>("sol", ndim, ShaderVariableType::FLOAT);

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

	AutogenShader shader;

	ui32 n = 1;
	ui16 ndim = 3;

	auto mat = std::make_shared<glsl::MatrixVariable>("mat", ndim, ndim, ShaderVariableType::FLOAT);
	auto rhs = std::make_shared<glsl::VectorVariable>("rhs", ndim, ShaderVariableType::FLOAT);
	auto sol = std::make_shared<glsl::VectorVariable>("sol", ndim, ShaderVariableType::FLOAT);

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



