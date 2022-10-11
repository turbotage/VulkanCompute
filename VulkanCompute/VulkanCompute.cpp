// VulkanCompute.cpp : Defines the entry point for the application.
//

#define KOMPUTE_LOG_LEVEL KOMPUTE_LOG_LEVEL_OFF
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
import <iomanip>;

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

void test_slmj() {
	using namespace glsl;
	using namespace vc;
	using namespace nlsq;

	auto start = std::chrono::steady_clock::now();

	ui16 ndata = 21;
	ui16 nparam = 4;
	ui16 nconst = 1;

	std::vector<std::string> vars = { "s0","f","d1","d2","b" };
	std::string expresh = "s0*(f*exp(-b*d1)+(1-f)*exp(-b*d2))";
	expression::Expression expr(expresh, vars);
	SymbolicContext context;
	context.insert_const(std::make_pair("b", 0));
	context.insert_param(std::make_pair("s0", 0));
	context.insert_param(std::make_pair("f", 1));
	context.insert_param(std::make_pair("d1", 2));
	context.insert_param(std::make_pair("d2", 3));

	auto params = std::make_shared<glsl::VectorVariable>("params", nparam, ShaderVariableType::FLOAT);
	auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, nconst, ShaderVariableType::FLOAT);
	auto data = std::make_shared<glsl::VectorVariable>("data", ndata, ShaderVariableType::FLOAT);
	auto lambda = std::make_shared<glsl::SingleVariable>("lambda", ShaderVariableType::FLOAT, std::nullopt);
	auto step_type = std::make_shared<glsl::SingleVariable>("step_type", ShaderVariableType::INT, std::nullopt);
	auto mu = std::make_shared<glsl::SingleVariable>("mu", ShaderVariableType::FLOAT, "0.25");
	auto eta = std::make_shared<glsl::SingleVariable>("eta", ShaderVariableType::FLOAT, "0.75");
	auto acc = std::make_shared<glsl::SingleVariable>("acc", ShaderVariableType::FLOAT, "0.5");
	auto dec = std::make_shared<glsl::SingleVariable>("dec", ShaderVariableType::FLOAT, "2");
	auto nlstep = std::make_shared<glsl::VectorVariable>("nlstep", nparam, ShaderVariableType::FLOAT);
	auto error = std::make_shared<glsl::SingleVariable>("error", ShaderVariableType::FLOAT, std::nullopt);
	auto new_error = std::make_shared<glsl::SingleVariable>("new_error", ShaderVariableType::FLOAT, std::nullopt);
	auto residuals = std::make_shared<glsl::VectorVariable>("residuals", ndata, ShaderVariableType::FLOAT);
	auto jacobian = std::make_shared<glsl::MatrixVariable>("jacobian", ndata, nparam, ShaderVariableType::FLOAT);
	auto hessian = std::make_shared<glsl::MatrixVariable>("hessian", nparam, nparam, ShaderVariableType::FLOAT);
	auto lambda_hessian = std::make_shared<glsl::MatrixVariable>("lambda_hessian", nparam, nparam, ShaderVariableType::FLOAT);

	auto nlsq_step = nlsq_slmj_step(
		expr, context,
		params, consts, data,
		lambda, step_type, mu, eta, acc, dec,
		nlstep, error, new_error,
		residuals, jacobian, hessian);

	AutogenShader shader;

	shader.addInputOutputVector(params, 0);
	shader.addInputMatrix(consts, 1);
	shader.addInputVector(data,	2);
	shader.addInputOutputSingle(lambda, 3);
	shader.addInputOutputSingle(step_type, 4);
	shader.addInputSingle(mu, 5);
	shader.addInputSingle(eta, 6);
	shader.addOutputVector(nlstep, 7);
	shader.addOutputSingle(error, 8);
	shader.addOutputSingle(new_error, 9);
	shader.addOutputVector(residuals, 10);
	shader.addOutputMatrix(jacobian, 11);
	shader.addOutputMatrix(hessian, 12);
	shader.addOutputMatrix(lambda_hessian, 13);

	shader.setAfterCopyingFrom(
R"glsl(
	step_type = 0;
	for (int i = 0; i < 12; ++i) {
)glsl");
	shader.apply(nlsq_step.func, nullptr,
		nlsq_step.args);
	shader.setBeforeCopyingBack(
R"glsl(
	}
)glsl");

	auto shader_code = shader.compile();

	auto end = std::chrono::steady_clock::now();

	//std::cout << shader_code << std::endl;
	std::cout << util::add_line_numbers(shader_code) << std::endl;

	std::cout << "shader build time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	
	OptimizationType opt_type =
		static_cast<OptimizationType>(
			static_cast<int>(OptimizationType::OPTIMIZE_FOR_SPEED) |
			static_cast<int>(OptimizationType::REMAP));
	
	//OptimizationType opt_type = OptimizationType::NO_OPTIMIZATION;

	auto spirv = glsl::compileSource(shader_code, opt_type);

	glsl::decompileSPIRV();

	uint32_t nelem = 500000;

	auto mgr = std::make_shared<kp::Manager>();

	auto kp_data = glsl::tensor_from_vector(mgr, data, nelem);
	auto kp_params = glsl::tensor_from_vector(mgr, params, nelem);
	auto kp_consts = glsl::tensor_from_matrix(mgr, consts, nelem);
	auto kp_lambda = glsl::tensor_from_single(mgr, lambda, nelem);
	auto kp_step_type = glsl::tensor_from_single(mgr, step_type, nelem);
	auto kp_mu = glsl::tensor_from_single(mgr, mu, nelem);
	auto kp_eta = glsl::tensor_from_single(mgr, eta, nelem);
	auto kp_nlstep = glsl::tensor_from_vector(mgr, nlstep, nelem);
	auto kp_error = glsl::tensor_from_single(mgr, error, nelem);
	auto kp_new_error = glsl::tensor_from_single(mgr, new_error, nelem);
	auto kp_residuals = glsl::tensor_from_vector(mgr, residuals, nelem);
	auto kp_jacobian = glsl::tensor_from_matrix(mgr, jacobian, nelem);
	auto kp_hessian = glsl::tensor_from_matrix(mgr, hessian, nelem);
	auto kp_lambda_hessian = glsl::tensor_from_matrix(mgr, lambda_hessian, nelem);

	// INITIALIZE DATA
	std::vector<float> data_data = { 908.02686, 905.39154, 906.08997, 700.7829, 753.0848, 859.9136,
	   870.48846, 755.96893, 617.3499, 566.2044 , 746.62067, 662.47424,
	   628.8806, 459.7746, 643.30554, 318.58453, 416.5493, 348.34335,
	   411.74026, 284.17468, 290.30487 };
	std::memcpy(kp_data->data<float>(), data_data.data(), sizeof(float)*data_data.size());

	std::vector<float> data_consts = { 0.,   10.,   20.,   30.,   40.,   60.,   80.,  100.,  120.,
		140.,  160.,  180.,  200.,  300.,  400.,  500.,  600.,  700.,
		800.,  900., 1000. };
	std::memcpy(kp_consts->data<float>(), data_consts.data(), sizeof(float)*data_consts.size());

	std::vector<float> data_params = { 700, 0.2, 0.1, 0.001 };
	std::memcpy(kp_params->data<float>(), data_params.data(), sizeof(float)*data_params.size());

	float data_lambda = 1.0f;
	std::memcpy(kp_lambda->data<float>(), &data_lambda, sizeof(float));

	float data_mu = 0.25f;
	std::memcpy(kp_mu->data<float>(), &data_mu, sizeof(float));

	float data_eta = 0.75f;
	std::memcpy(kp_eta->data<float>(), &data_eta, sizeof(float));
	// END INITIALIZE DATA

	std::vector<std::shared_ptr<kp::Tensor>> shader_inputs = {
		kp_params, kp_consts, kp_data, kp_lambda, kp_step_type,
		kp_mu, kp_eta, kp_nlstep, kp_error, kp_new_error,
		kp_residuals, kp_jacobian, kp_hessian, kp_lambda_hessian
	};


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

	std::cout << "FIRST ELEMENTS" << std::endl;
	std::cout << "residuals: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_residuals, residuals, 0) << std::endl;
	std::cout << "jacobian: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_jacobian, jacobian, 0) << std::endl;
	std::cout << "hessian: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_hessian, hessian, 0) << std::endl;
	std::cout << "lambda_hessian: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_lambda_hessian, lambda_hessian, 0) << std::endl;
	std::cout << "data: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_data, data, 0) << std::endl;
	std::cout << "params: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_params, params, 0) << std::endl;
	std::cout << "consts: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_consts, consts, 0) << std::endl;
	std::cout << "lambda: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_lambda, lambda, 0) << std::endl;
	std::cout << "step_type: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_step_type, step_type, 0) << std::endl;
	std::cout << "mu: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_mu, mu, 0) << std::endl;
	std::cout << "eta: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_eta, eta, 0) << std::endl;
	std::cout << "step: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_nlstep, nlstep, 0) << std::endl;

	/*
	std::cout << "LAST ELEMENTS" << std::endl;
	std::cout << "residuals: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_residuals, residuals, nelem - 1) << std::endl;
	std::cout << "jacobian: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_jacobian, jacobian, nelem - 1) << std::endl;
	std::cout << "hessian: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_hessian, hessian, nelem - 1) << std::endl;
	std::cout << "lambda_hessian: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_lambda_hessian, lambda_hessian, nelem - 1) << std::endl;
	std::cout << "data: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_data, data, nelem - 1) << std::endl;
	std::cout << "params: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_params, params, nelem - 1) << std::endl;
	std::cout << "consts: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_consts, consts, nelem - 1) << std::endl;
	std::cout << "lambda: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_lambda, lambda, nelem - 1) << std::endl;
	std::cout << "step_type: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_step_type, step_type, nelem - 1) << std::endl;
	std::cout << "mu: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_mu, mu, nelem - 1) << std::endl;
	std::cout << "eta: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_eta, eta, nelem - 1) << std::endl;
	std::cout << "step: " << std::endl;
	std::cout << glsl::print_shader_variable(kp_nlstep, nlstep, nelem - 1) << std::endl;
	*/

}

void test_slmh() {
	using namespace glsl;
	using namespace vc;
	using namespace nlsq;

	auto start = std::chrono::steady_clock::now();

	ui16 ndata = 21;
	ui16 nparam = 4;
	ui16 nconst = 1;

	std::vector<std::string> vars = { "s0","f","d1","d2","b" };
	std::string expresh = "s0*(f*exp(-b*d1)+(1-f)*exp(-b*d2))";
	expression::Expression expr(expresh, vars);
	SymbolicContext context;
	context.insert_const(std::make_pair("b", 0));
	context.insert_param(std::make_pair("s0", 0));
	context.insert_param(std::make_pair("f", 1));
	context.insert_param(std::make_pair("d1", 2));
	context.insert_param(std::make_pair("d2", 3));

	auto params = std::make_shared<glsl::VectorVariable>("params", nparam, ShaderVariableType::FLOAT);
	auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, nconst, ShaderVariableType::FLOAT);
	auto data = std::make_shared<glsl::VectorVariable>("data", ndata, ShaderVariableType::FLOAT);
	auto lambda = std::make_shared<glsl::SingleVariable>("lambda", ShaderVariableType::FLOAT, std::nullopt);
	auto step_type = std::make_shared<glsl::SingleVariable>("step_type", ShaderVariableType::INT, std::nullopt);
	auto mu = std::make_shared<glsl::SingleVariable>("mu", ShaderVariableType::FLOAT, "0.25");
	auto eta = std::make_shared<glsl::SingleVariable>("eta", ShaderVariableType::FLOAT, "0.75");
	auto acc = std::make_shared<glsl::SingleVariable>("acc", ShaderVariableType::FLOAT, "0.5");
	auto dec = std::make_shared<glsl::SingleVariable>("dec", ShaderVariableType::FLOAT, "2");
	auto nlstep = std::make_shared<glsl::VectorVariable>("nlstep", nparam, ShaderVariableType::FLOAT);
	auto error = std::make_shared<glsl::SingleVariable>("error", ShaderVariableType::FLOAT, std::nullopt);
	auto new_error = std::make_shared<glsl::SingleVariable>("new_error", ShaderVariableType::FLOAT, std::nullopt);
	auto residuals = std::make_shared<glsl::VectorVariable>("residuals", ndata, ShaderVariableType::FLOAT);
	auto jacobian = std::make_shared<glsl::MatrixVariable>("jacobian", ndata, nparam, ShaderVariableType::FLOAT);
	auto hessian = std::make_shared<glsl::MatrixVariable>("hessian", nparam, nparam, ShaderVariableType::FLOAT);
	auto lambda_hessian = std::make_shared<glsl::MatrixVariable>("lambda_hessian", nparam, nparam, ShaderVariableType::FLOAT);

	auto nlsq_step = nlsq_slmh_step(
		expr, context,
		params, consts, data,
		lambda, step_type, mu, eta, acc, dec,
		nlstep, error, new_error,
		residuals, jacobian, hessian, lambda_hessian);

	AutogenShader shader;

	shader.addInputOutputVector(params, 0);
	shader.addInputMatrix(consts, 1);
	shader.addInputVector(data, 2);
	shader.addInputOutputSingle(lambda, 3);
	shader.addInputOutputSingle(step_type, 4);
	shader.addInputSingle(mu, 5);
	shader.addInputSingle(eta, 6);
	shader.addOutputVector(nlstep, 7);
	shader.addOutputSingle(error, 8);
	shader.addOutputSingle(new_error, 9);
	shader.addOutputVector(residuals, 10);
	shader.addOutputMatrix(jacobian, 11);
	shader.addOutputMatrix(hessian, 12);
	shader.addOutputMatrix(lambda_hessian, 13);

	shader.setAfterCopyingFrom("for (int i = 0; i < 9; ++i) {");
	shader.apply(nlsq_step.func, nullptr,
		nlsq_step.args);
	shader.setBeforeCopyingBack("}");

	auto shader_code = shader.compile();

	auto end = std::chrono::steady_clock::now();

	//std::cout << shader_code << std::endl;
	//std::cout << util::add_line_numbers(shader_code) << std::endl;

	std::cout << "shader build time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;


	OptimizationType opt_type =
		static_cast<OptimizationType>(
			static_cast<int>(OptimizationType::OPTIMIZE_FOR_SPEED) |
			static_cast<int>(OptimizationType::REMAP));

	//OptimizationType opt_type = OptimizationType::NO_OPTIMIZATION;

	auto spirv = glsl::compileSource(shader_code, opt_type);

	glsl::decompileSPIRV();

	uint32_t nelem = 500000;

	auto mgr = std::make_shared<kp::Manager>();

	auto kp_data = glsl::tensor_from_vector(mgr, data, nelem);
	auto kp_params = glsl::tensor_from_vector(mgr, params, nelem);
	auto kp_consts = glsl::tensor_from_matrix(mgr, consts, nelem);
	auto kp_lambda = glsl::tensor_from_single(mgr, lambda, nelem);
	auto kp_step_type = glsl::tensor_from_single(mgr, step_type, nelem);
	auto kp_mu = glsl::tensor_from_single(mgr, mu, nelem);
	auto kp_eta = glsl::tensor_from_single(mgr, eta, nelem);
	auto kp_nlstep = glsl::tensor_from_vector(mgr, nlstep, nelem);
	auto kp_error = glsl::tensor_from_single(mgr, error, nelem);
	auto kp_new_error = glsl::tensor_from_single(mgr, new_error, nelem);
	auto kp_residuals = glsl::tensor_from_vector(mgr, residuals, nelem);
	auto kp_jacobian = glsl::tensor_from_matrix(mgr, jacobian, nelem);
	auto kp_hessian = glsl::tensor_from_matrix(mgr, hessian, nelem);
	auto kp_lambda_hessian = glsl::tensor_from_matrix(mgr, lambda_hessian, nelem);


	float S0_div = 100.0f;
	float b_div = 10.0f;
	// INITIALIZE DATA
	std::vector<float> data_data = { 908.02686, 905.39154, 906.08997, 700.7829, 753.0848, 859.9136,
	   870.48846, 755.96893, 617.3499, 566.2044 , 746.62067, 662.47424,
	   628.8806, 459.7746, 643.30554, 318.58453, 416.5493, 348.34335,
	   411.74026, 284.17468, 290.30487 };
	for (auto& dd : data_data) {
		dd /= S0_div;
	}

	std::memcpy(kp_data->data<float>(), data_data.data(), sizeof(float) * data_data.size());

	std::vector<float> data_consts = { 0.0f,   10.0f,   20.0f,   30.0f,   40.0f,   60.0f,   80.0f,  100.0f,  120.0f,
		140.,  160.,  180.,  200.,  300.,  400.,  500.,  600.,  700.,
		800.0f,  900.0f, 1000.0f };
	for (auto& dc : data_consts) {
		dc /= b_div;
	}

	std::memcpy(kp_consts->data<float>(), data_consts.data(), sizeof(float) * data_consts.size());

	std::vector<float> data_params = { 700.0f / S0_div, 0.2, 0.1f * b_div, 0.001f * b_div };
	std::memcpy(kp_params->data<float>(), data_params.data(), sizeof(float) * data_params.size());

	float data_lambda = 2.0f;
	std::memcpy(kp_lambda->data<float>(), &data_lambda, sizeof(float));

	float data_mu = 0.25f;
	std::memcpy(kp_mu->data<float>(), &data_mu, sizeof(float));

	float data_eta = 0.75f;
	std::memcpy(kp_eta->data<float>(), &data_eta, sizeof(float));
	// END INITIALIZE DATA

	std::vector<std::shared_ptr<kp::Tensor>> shader_inputs = {
		kp_params, kp_consts, kp_data, kp_lambda, kp_step_type,
		kp_mu, kp_eta, kp_nlstep, kp_error, kp_new_error,
		kp_residuals, kp_jacobian, kp_hessian, kp_lambda_hessian
	};


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

	bool print = true;
	if (print) {
		std::cout.precision(10);
		std::cout << "FIRST ELEMENTS" << std::endl;
		std::cout << "residuals: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_residuals, residuals, 0) << std::endl;
		std::cout << "jacobian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_jacobian, jacobian, 0) << std::endl;
		std::cout << "hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_hessian, hessian, 0) << std::endl;
		std::cout << "lambda_hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda_hessian, lambda_hessian, 0) << std::endl;
		std::cout << "data: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_data, data, 0) << std::endl;
		std::cout << "params: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_params, params, 0) << std::endl;
		std::cout << "consts: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_consts, consts, 0) << std::endl;
		std::cout << "lambda: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda, lambda, 0) << std::endl;
		std::cout << "step_type: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_step_type, step_type, 0) << std::endl;
		std::cout << "mu: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_mu, mu, 0) << std::endl;
		std::cout << "eta: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_eta, eta, 0) << std::endl;
		std::cout << "step: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_nlstep, nlstep, 0) << std::endl;

		/*
		std::cout << "LAST ELEMENTS" << std::endl;
		std::cout << "residuals: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_residuals, residuals, nelem - 1) << std::endl;
		std::cout << "jacobian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_jacobian, jacobian, nelem - 1) << std::endl;
		std::cout << "hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_hessian, hessian, nelem - 1) << std::endl;
		std::cout << "lambda_hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda_hessian, lambda_hessian, nelem - 1) << std::endl;
		std::cout << "data: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_data, data, nelem - 1) << std::endl;
		std::cout << "params: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_params, params, nelem - 1) << std::endl;
		std::cout << "consts: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_consts, consts, nelem - 1) << std::endl;
		std::cout << "lambda: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda, lambda, nelem - 1) << std::endl;
		std::cout << "step_type: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_step_type, step_type, nelem - 1) << std::endl;
		std::cout << "mu: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_mu, mu, nelem - 1) << std::endl;
		std::cout << "eta: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_eta, eta, nelem - 1) << std::endl;
		std::cout << "step: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_nlstep, nlstep, nelem - 1) << std::endl;
		*/
	}

}

void test_slmh_w() {
	using namespace glsl;
	using namespace vc;
	using namespace nlsq;

	auto start = std::chrono::steady_clock::now();

	ui16 ndata = 21;
	ui16 nparam = 4;
	ui16 nconst = 1;

	std::vector<std::string> vars = { "s0","f","d1","d2","b" };
	std::string expresh = "s0*(f*exp(-b*d1)+(1-f)*exp(-b*d2))";
	expression::Expression expr(expresh, vars);
	SymbolicContext context;
	context.insert_const(std::make_pair("b", 0));
	context.insert_param(std::make_pair("s0", 0));
	context.insert_param(std::make_pair("f", 1));
	context.insert_param(std::make_pair("d1", 2));
	context.insert_param(std::make_pair("d2", 3));

	auto params = std::make_shared<glsl::VectorVariable>("params", nparam, ShaderVariableType::FLOAT);
	auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, nconst, ShaderVariableType::FLOAT);
	auto data = std::make_shared<glsl::VectorVariable>("data", ndata, ShaderVariableType::FLOAT);
	auto weights = std::make_shared<glsl::VectorVariable>("weights", ndata, ShaderVariableType::FLOAT);
	auto lambda = std::make_shared<glsl::SingleVariable>("lambda", ShaderVariableType::FLOAT, std::nullopt);
	auto step_type = std::make_shared<glsl::SingleVariable>("step_type", ShaderVariableType::INT, std::nullopt);
	auto mu = std::make_shared<glsl::SingleVariable>("mu", ShaderVariableType::FLOAT, "0.25");
	auto eta = std::make_shared<glsl::SingleVariable>("eta", ShaderVariableType::FLOAT, "0.75");
	auto acc = std::make_shared<glsl::SingleVariable>("acc", ShaderVariableType::FLOAT, "0.2");
	auto dec = std::make_shared<glsl::SingleVariable>("dec", ShaderVariableType::FLOAT, "5.0");
	auto nlstep = std::make_shared<glsl::VectorVariable>("nlstep", nparam, ShaderVariableType::FLOAT);
	auto error = std::make_shared<glsl::SingleVariable>("error", ShaderVariableType::FLOAT, std::nullopt);
	auto new_error = std::make_shared<glsl::SingleVariable>("new_error", ShaderVariableType::FLOAT, std::nullopt);
	auto residuals = std::make_shared<glsl::VectorVariable>("residuals", ndata, ShaderVariableType::FLOAT);
	auto jacobian = std::make_shared<glsl::MatrixVariable>("jacobian", ndata, nparam, ShaderVariableType::FLOAT);
	auto hessian = std::make_shared<glsl::MatrixVariable>("hessian", nparam, nparam, ShaderVariableType::FLOAT);
	auto lambda_hessian = std::make_shared<glsl::MatrixVariable>("lambda_hessian", nparam, nparam, ShaderVariableType::FLOAT);
	auto upper_bound = std::make_shared<glsl::VectorVariable>("upper_bound", nparam, ShaderVariableType::FLOAT);
	auto lower_bound = std::make_shared<glsl::VectorVariable>("lower_bound", nparam, ShaderVariableType::FLOAT);

	auto nlsq_step = nlsq_slmh_w_step(
		expr, context,
		params, consts, data, weights,
		lambda, step_type, mu, eta, acc, dec,
		nlstep, error, new_error,
		residuals, jacobian, hessian, lambda_hessian);

	AutogenShader shader;

	shader.addInputOutputVector(params, 0);
	shader.addInputMatrix(consts, 1);
	shader.addInputVector(data, 2);
	shader.addInputVector(weights, 3);
	shader.addInputOutputSingle(lambda, 4);
	shader.addInputOutputSingle(step_type, 5);
	shader.addInputSingle(mu, 6);
	shader.addInputSingle(eta, 7);
	shader.addOutputVector(nlstep, 8);
	shader.addOutputSingle(error, 9);
	shader.addOutputSingle(new_error, 10);
	shader.addOutputVector(residuals, 11);
	shader.addOutputMatrix(jacobian, 12);
	shader.addOutputMatrix(hessian, 13);
	shader.addOutputMatrix(lambda_hessian, 14);
	shader.addInputVector(upper_bound, 15);
	shader.addInputVector(lower_bound, 16);


	shader.setAfterCopyingFrom(
R"glsl(
	for (int i = 0; i < 25; ++i) {
)glsl");
	shader.apply(nlsq_step.func, nullptr,
		nlsq_step.args);
	shader.setBeforeCopyingBack(
R"glsl(
	for (int j = 0; j < 4; ++j) {
		if (isnan(params[j]) || isinf(params[j])) {
			params[j] = 0.5 * (upper_bound[j] - lower_bound[j]);
		}
		params[j] = clamp(params[j], lower_bound[j], upper_bound[j]);
	}
	}
)glsl");

	auto shader_code = shader.compile();

	auto end = std::chrono::steady_clock::now();

	//std::cout << shader_code << std::endl;
	std::cout << util::add_line_numbers(shader_code) << std::endl;

	std::cout << "shader build time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;


	OptimizationType opt_type =
		static_cast<OptimizationType>(
			static_cast<int>(OptimizationType::OPTIMIZE_FOR_SPEED) |
			static_cast<int>(OptimizationType::REMAP));

	//OptimizationType opt_type = OptimizationType::NO_OPTIMIZATION;

	auto spirv = glsl::compileSource(shader_code, opt_type);

	glsl::decompileSPIRV();

	uint32_t nelem = 500000;

	auto mgr = std::make_shared<kp::Manager>();

	auto kp_data = glsl::tensor_from_vector(mgr, data, nelem);
	auto kp_weights = glsl::tensor_from_vector(mgr, weights, nelem);
	auto kp_params = glsl::tensor_from_vector(mgr, params, nelem);
	auto kp_consts = glsl::tensor_from_matrix(mgr, consts, nelem);
	auto kp_lambda = glsl::tensor_from_single(mgr, lambda, nelem);
	auto kp_step_type = glsl::tensor_from_single(mgr, step_type, nelem);
	auto kp_mu = glsl::tensor_from_single(mgr, mu, nelem);
	auto kp_eta = glsl::tensor_from_single(mgr, eta, nelem);
	auto kp_nlstep = glsl::tensor_from_vector(mgr, nlstep, nelem);
	auto kp_error = glsl::tensor_from_single(mgr, error, nelem);
	auto kp_new_error = glsl::tensor_from_single(mgr, new_error, nelem);
	auto kp_residuals = glsl::tensor_from_vector(mgr, residuals, nelem);
	auto kp_jacobian = glsl::tensor_from_matrix(mgr, jacobian, nelem);
	auto kp_hessian = glsl::tensor_from_matrix(mgr, hessian, nelem);
	auto kp_lambda_hessian = glsl::tensor_from_matrix(mgr, lambda_hessian, nelem);
	auto kp_upper_bound = glsl::tensor_from_vector(mgr, upper_bound, nelem);
	auto kp_lower_bound = glsl::tensor_from_vector(mgr, lower_bound, nelem);


	float S0_div = 1.0f;
	float b_div = 1.0f;
	// INITIALIZE DATA
	std::vector<float> data_data = { 908.02686, 905.39154, 906.08997, 700.7829, 753.0848, 859.9136,
	   870.48846, 755.96893, 617.3499, 566.2044 , 746.62067, 662.47424,
	   628.8806, 459.7746, 643.30554, 318.58453, 416.5493, 348.34335,
	   411.74026, 284.17468, 290.30487 };
	for (auto& dd : data_data) {
		dd /= S0_div;
	}
	std::memcpy(kp_data->data<float>(), data_data.data(), sizeof(float) * data_data.size());

	std::vector<float> data_weights(21, 1.0f);
	std::memcpy(kp_weights->data<float>(), data_weights.data(), sizeof(float) * data_weights.size());

	std::vector<float> data_consts = { 0.0f,   10.0f,   20.0f,   30.0f,   40.0f,   60.0f,   80.0f,  100.0f,  120.0f,
		140.,  160.,  180.,  200.,  300.,  400.,  500.,  600.,  700.,
		800.0f,  900.0f, 1000.0f };
	for (auto& dc : data_consts) {
		dc /= b_div;
	}
	std::memcpy(kp_consts->data<float>(), data_consts.data(), sizeof(float) * data_consts.size());

	std::vector<float> data_params = { 700.0f / S0_div, 0.2, 0.1f * b_div, 0.001f * b_div };
	std::memcpy(kp_params->data<float>(), data_params.data(), sizeof(float) * data_params.size());

	float data_lambda = 0.1f;
	std::memcpy(kp_lambda->data<float>(), &data_lambda, sizeof(float));

	float data_mu = 0.25f;
	std::memcpy(kp_mu->data<float>(), &data_mu, sizeof(float));

	float data_eta = 0.75f;
	std::memcpy(kp_eta->data<float>(), &data_eta, sizeof(float));
	
	std::vector<float> data_upper_bound = { 1200.0, 1.0, 0.1, 0.01 };
	std::memcpy(kp_upper_bound->data<float>(), data_upper_bound.data(), sizeof(float) * data_upper_bound.size());
	
	std::vector<float> data_lower_bound = { 500.0, 0.01, 0.0005, 0.00005 };
	std::memcpy(kp_lower_bound->data<float>(), data_lower_bound.data(), sizeof(float) * data_lower_bound.size());

	// END INITIALIZE DATA

	std::vector<std::shared_ptr<kp::Tensor>> shader_inputs = {
		kp_params, kp_consts, kp_data, kp_weights, kp_lambda, kp_step_type,
		kp_mu, kp_eta, kp_nlstep, kp_error, kp_new_error,
		kp_residuals, kp_jacobian, kp_hessian, kp_lambda_hessian, 
		kp_upper_bound, kp_lower_bound
	};


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

	bool print = true;
	if (print) {
		std::cout.precision(10);
		std::cout << "FIRST ELEMENTS" << std::endl;
		std::cout << "residuals: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_residuals, residuals, 0) << std::endl;
		std::cout << "jacobian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_jacobian, jacobian, 0) << std::endl;
		std::cout << "hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_hessian, hessian, 0) << std::endl;
		std::cout << "lambda_hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda_hessian, lambda_hessian, 0) << std::endl;
		std::cout << "data: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_data, data, 0) << std::endl;
		std::cout << "weights: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_weights, weights, 0) << std::endl;
		std::cout << "params: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_params, params, 0) << std::endl;
		std::cout << "consts: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_consts, consts, 0) << std::endl;
		std::cout << "lambda: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda, lambda, 0) << std::endl;
		std::cout << "step_type: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_step_type, step_type, 0) << std::endl;
		std::cout << "mu: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_mu, mu, 0) << std::endl;
		std::cout << "eta: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_eta, eta, 0) << std::endl;
		std::cout << "step: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_nlstep, nlstep, 0) << std::endl;
		std::cout << "upper_bound: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_upper_bound, upper_bound, 0) << std::endl;
		std::cout << "lower_bound: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lower_bound, lower_bound, 0) << std::endl;

		/*
		std::cout << "LAST ELEMENTS" << std::endl;
		std::cout << "residuals: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_residuals, residuals, nelem - 1) << std::endl;
		std::cout << "jacobian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_jacobian, jacobian, nelem - 1) << std::endl;
		std::cout << "hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_hessian, hessian, nelem - 1) << std::endl;
		std::cout << "lambda_hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda_hessian, lambda_hessian, nelem - 1) << std::endl;
		std::cout << "data: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_data, data, nelem - 1) << std::endl;
		std::cout << "params: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_params, params, nelem - 1) << std::endl;
		std::cout << "consts: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_consts, consts, nelem - 1) << std::endl;
		std::cout << "lambda: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda, lambda, nelem - 1) << std::endl;
		std::cout << "step_type: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_step_type, step_type, nelem - 1) << std::endl;
		std::cout << "mu: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_mu, mu, nelem - 1) << std::endl;
		std::cout << "eta: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_eta, eta, nelem - 1) << std::endl;
		std::cout << "step: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_nlstep, nlstep, nelem - 1) << std::endl;
		*/
	}

}


int main() {
	
	//test_function_factory();
	//test_slmj();
	test_slmh_w();
	//test_slmh();
	//test_slmh();
	//test_slmh();

	return 0;
}



