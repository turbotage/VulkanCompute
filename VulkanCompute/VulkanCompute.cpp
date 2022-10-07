// VulkanCompute.cpp : Defines the entry point for the application.
//

#include <kompute/Kompute.hpp>
#include <symengine/expression.h>
#include <symengine/refine.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>
#include <symengine/parser/parser.h>

#include "glsl/tensor_var.h"

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
import nlsq;
import nlsq_symbolic;

import variable;
import function;
import shader;
import func_factory;

import tensor_var;


void test_function_factory()
{
	using namespace glsl;
	using namespace vc;
	using namespace nlsq;

	auto start = std::chrono::steady_clock::now();

	AutogenShader shader;

	ui16 ndata = 21;
	ui16 nparam = 4;
	ui16 nconst = 1;

	std::vector<std::string> vars = { "s0","f","d1","d2","b" };
	std::string expresh = "s0*(f*exp(-b*d1)+(1-f)*exp(-b*d2))+1";
	expression::Expression expr(expresh, vars);
	SymbolicContext context;
	context.insert_const(std::make_pair("b", 0));
	context.insert_param(std::make_pair("s0", 0));
	context.insert_param(std::make_pair("f", 1));
	context.insert_param(std::make_pair("d1", 2));
	context.insert_param(std::make_pair("d2", 3));

	auto residuals = std::make_shared<glsl::VectorVariable>("residuals", ndata, ShaderVariableType::FLOAT);
	auto jacobian = std::make_shared<glsl::MatrixVariable>("jacobian", ndata, nparam, ShaderVariableType::FLOAT);
	auto hessian = std::make_shared<glsl::MatrixVariable>("hessian", nparam, nparam, ShaderVariableType::FLOAT);
	auto data = std::make_shared<glsl::VectorVariable>("data", ndata, ShaderVariableType::FLOAT);
	auto params = std::make_shared<glsl::VectorVariable>("params", nparam, ShaderVariableType::FLOAT);
	auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, nconst, ShaderVariableType::FLOAT);
	auto lambda = std::make_shared<glsl::SingleVariable>("lambda", ShaderVariableType::FLOAT, std::nullopt);
	auto step = std::make_shared<glsl::VectorVariable>("step", nparam, ShaderVariableType::FLOAT);
	auto step_type = std::make_shared<glsl::SingleVariable>("step_type", ShaderVariableType::INT, std::nullopt);

	FunctionFactory factory("nlsq_slm_step", ShaderVariableType::VOID);
	
	factory.addVector(residuals, FunctionFactory::InputType::INOUT);
	factory.addMatrix(jacobian, FunctionFactory::InputType::INOUT);
	factory.addMatrix(hessian, FunctionFactory::InputType::INOUT);
	factory.addVector(data, FunctionFactory::InputType::IN);
	factory.addVector(params, FunctionFactory::InputType::INOUT);
	factory.addMatrix(consts, FunctionFactory::InputType::IN);
	factory.addSingle(lambda, FunctionFactory::InputType::INOUT);
	factory.addSingle(step_type, FunctionFactory::InputType::INOUT);

	factory.apply(nlsq_residuals_jacobian(expr, context, ndata, nparam, nconst, true), nullptr,
		{ params, consts, data, residuals, jacobian });

	auto& for_scope = factory.apply_scope(ForScope::make("int i = 0; i < 3; ++i"));

	for_scope.apply(nlsq_residuals_jacobian_hessian(expr, context, ndata, nparam, nconst, true), nullptr,
		{ params, consts, data, residuals, jacobian, hessian });

	auto built_func = factory.build_function();

	std::cout << built_func->getCode() << std::endl;


}

void test_function_factorized() {
	using namespace glsl;
	using namespace vc;
	using namespace nlsq;

	auto start = std::chrono::steady_clock::now();

	ui16 ndata = 21;
	ui16 nparam = 4;
	ui16 nconst = 1;

	std::vector<std::string> vars = { "s0","f","d1","d2","b" };
	std::string expresh = "s0*(f*exp(-b*d1)+(1-f)*exp(-b*d2))+1";
	expression::Expression expr(expresh, vars);
	SymbolicContext context;
	context.insert_const(std::make_pair("b", 0));
	context.insert_param(std::make_pair("s0", 0));
	context.insert_param(std::make_pair("f", 1));
	context.insert_param(std::make_pair("d1", 2));
	context.insert_param(std::make_pair("d2", 3));

	auto residuals = std::make_shared<glsl::VectorVariable>("residuals", ndata, ShaderVariableType::FLOAT);
	auto jacobian = std::make_shared<glsl::MatrixVariable>("jacobian", ndata, nparam, ShaderVariableType::FLOAT);
	auto hessian = std::make_shared<glsl::MatrixVariable>("hessian", nparam, nparam, ShaderVariableType::FLOAT);
	auto data = std::make_shared<glsl::VectorVariable>("data", ndata, ShaderVariableType::FLOAT);
	auto params = std::make_shared<glsl::VectorVariable>("params", nparam, ShaderVariableType::FLOAT);
	auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, nconst, ShaderVariableType::FLOAT);
	auto lambda = std::make_shared<glsl::SingleVariable>("lambda", ShaderVariableType::FLOAT, std::nullopt);
	auto nlstep = std::make_shared<glsl::VectorVariable>("nlstep", nparam, ShaderVariableType::FLOAT);
	auto step_type = std::make_shared<glsl::SingleVariable>("step_type", ShaderVariableType::INT, std::nullopt);
	auto mu = std::make_shared<glsl::SingleVariable>("mu", ShaderVariableType::FLOAT, "0.25");
	auto eta = std::make_shared<glsl::SingleVariable>("eta", ShaderVariableType::FLOAT, "0.75");

	auto nlsq_step = nlsq_slmh_step(
		expr, context,
		params, consts, data,
		residuals, jacobian, hessian,
		lambda, mu, eta, nlstep, step_type);

	AutogenShader shader;

	shader.addInputOutputVector(residuals, 0);
	shader.addInputOutputMatrix(jacobian, 1);
	shader.addInputOutputMatrix(hessian, 2);
	shader.addInputVector(data,	3);
	shader.addInputOutputVector(params, 4);
	shader.addInputMatrix(consts, 5);
	shader.addInputOutputSingle(lambda, 6);
	shader.addInputOutputSingle(step_type, 7);
	shader.addInputSingle(mu, 8);
	shader.addInputSingle(eta, 9);

	shader.apply(nlsq_step.func, nullptr,
		nlsq_step.args);

	shader.setAfterCopyingFrom("\tfor (int iter = 0; iter < 1; ++iter) {");
	shader.setBeforeCopyingBack("\t}");

	auto shader_code = shader.compile();

	auto end = std::chrono::steady_clock::now();

	std::cout << util::add_line_numbers(shader_code) << std::endl;
	//std::cout << util::add_line_numbers(shader_code) << std::endl;

	std::cout << "shader build time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	/*
	OptimizationType opt_type =
		static_cast<OptimizationType>(
			static_cast<int>(OptimizationType::OPTIMIZE_FOR_SPEED) |
			static_cast<int>(OptimizationType::REMAP));
	*/

	OptimizationType opt_type = OptimizationType::NO_OPTIMIZATION;

	auto spirv = glsl::compileSource(shader_code, opt_type);

	glsl::decompileSPIRV();

	uint32_t nelem = 1;

	auto mgr = std::make_shared<kp::Manager>();

	auto kp_residuals = glsl::tensor_from_vector(mgr, residuals, nelem);
	auto kp_jacobian = glsl::tensor_from_matrix(mgr, jacobian, nelem);
	auto kp_hessian = glsl::tensor_from_matrix(mgr, hessian, nelem);
	auto kp_data = glsl::tensor_from_vector(mgr, data, nelem);
	auto kp_params = glsl::tensor_from_vector(mgr, params, nelem);
	auto kp_consts = glsl::tensor_from_matrix(mgr, consts, nelem);
	auto kp_lambda = glsl::tensor_from_single(mgr, lambda, nelem);
	auto kp_step_type = glsl::tensor_from_single(mgr, step_type, nelem);
	auto kp_mu = glsl::tensor_from_single(mgr, mu, nelem);
	auto kp_eta = glsl::tensor_from_single(mgr, eta, nelem);
	
	std::vector<std::shared_ptr<kp::Tensor>> shader_inputs = {
		kp_residuals, kp_jacobian, kp_hessian,
		kp_data, kp_params, kp_consts,
		kp_lambda, kp_step_type, kp_mu, kp_eta };

	kp::Workgroup wg{ (size_t)nelem, 1, 1 };

	std::shared_ptr<kp::Algorithm> algo = mgr->algorithm(shader_inputs, spirv, wg);
	
	start = std::chrono::steady_clock::now();

	auto seq = mgr->sequence()
		->record<kp::OpTensorSyncDevice>(shader_inputs)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpTensorSyncLocal>(shader_inputs)
		->eval();

	end = std::chrono::steady_clock::now();


	std::cout << "run time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	std::cout << "residuals: " << std::endl;
	std::cout << glsl::print_vector<float>(kp_residuals, residuals, 0, std::nullopt) << std::endl;
	std::cout << "jacobian: " << std::endl;
	std::cout << glsl::print_matrix<float>(kp_jacobian, jacobian, 0, std::nullopt) << std::endl;
	std::cout << "hessian: " << std::endl;
	std::cout << glsl::print_matrix<float>(kp_hessian, hessian, 0, std::nullopt) << std::endl;
	std::cout << "data: " << std::endl;
	std::cout << glsl::print_vector<float>(kp_data, data, 0, std::nullopt) << std::endl;
	std::cout << "params: " << std::endl;
	std::cout << glsl::print_vector<float>(kp_params, params, 0, std::nullopt) << std::endl;
	std::cout << "consts: " << std::endl;
	std::cout << glsl::print_matrix<float>(kp_consts, consts, 0, std::nullopt) << std::endl;
	std::cout << "lambda: " << std::endl;
	std::cout << glsl::print_single<float>(kp_lambda, lambda, 0, std::nullopt) << std::endl;
	std::cout << "step_type: " << std::endl;
	std::cout << glsl::print_single<int32_t>(kp_step_type, step_type, 0, std::nullopt) << std::endl;
	std::cout << "mu: " << std::endl;
	std::cout << glsl::print_single<float>(kp_mu, mu, 0, std::nullopt) << std::endl;
	std::cout << "eta: " << std::endl;
	std::cout << glsl::print_single<float>(kp_eta, eta, 0, std::nullopt) << std::endl;

}

int main() {
	
	//test_function_factory();
	test_function_factorized();

	return 0;
}



