// VulkanCompute.cpp : Defines the entry point for the application.
//

#include "VulkanCompute.hpp"

import <vector>;
import <memory>;
import <optional>;
import <random>;
import <chrono>;
import <string>;
import <iomanip>;
import <filesystem>;

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

import qmri;

import variable;
import function;
import shader;
import func_factory;

import tensor_var;

import test_avk;

void test_new_autogen() {
	using namespace glsl;

	auto shader1 = qmri::ivim_guess_shader(21, true);
	auto shader2 = qmri::ivim_full_nlsq_shader(21, true);

	auto sh1_compiled = shader1->compile();
	std::cout << util::add_line_numbers(
		sh1_compiled
	);

	auto spirv1 = glsl::compileSource(sh1_compiled);

	std::cout << "\n\n\n\n";

	auto sh2_compiled = shader2->compile();
	std::cout << util::add_line_numbers(
		sh2_compiled
	);

	auto spirv2 = glsl::compileSource(sh2_compiled);
	
}

void run_qmri_ivim() {

	uint32_t ndata = 21;
	uint32_t nconst = 1;

	auto params = std::make_shared<glsl::VectorVariable>("params", 4, glsl::ShaderVariableType::eFloat);
	auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, nconst, glsl::ShaderVariableType::eFloat);
	auto data = std::make_shared<glsl::VectorVariable>("data", ndata, glsl::ShaderVariableType::eFloat);
	auto bsplit = std::make_shared<glsl::VectorVariable>("bsplit", 2, glsl::ShaderVariableType::eInt);
	auto weights = std::make_shared<glsl::VectorVariable>("weights", ndata, glsl::ShaderVariableType::eFloat);
	auto lambda = std::make_shared<glsl::SingleVariable>("lambda", glsl::ShaderVariableType::eFloat, std::nullopt);
	auto step_type = std::make_shared<glsl::SingleVariable>("step_type", glsl::ShaderVariableType::eInt, std::nullopt);
	auto nlstep = std::make_shared<glsl::VectorVariable>("nlstep", 4, glsl::ShaderVariableType::eFloat);
	auto error = std::make_shared<glsl::SingleVariable>("error", glsl::ShaderVariableType::eFloat, std::nullopt);
	auto new_error = std::make_shared<glsl::SingleVariable>("new_error", glsl::ShaderVariableType::eFloat, std::nullopt);
	auto residuals = std::make_shared<glsl::VectorVariable>("residuals", ndata, glsl::ShaderVariableType::eFloat);
	auto jacobian = std::make_shared<glsl::MatrixVariable>("jacobian", ndata, 4, glsl::ShaderVariableType::eFloat);
	auto hessian = std::make_shared<glsl::MatrixVariable>("hessian", 4, 4, glsl::ShaderVariableType::eFloat);
	auto lambda_hessian = std::make_shared<glsl::MatrixVariable>("lambda_hessian", 4, 4, glsl::ShaderVariableType::eFloat);
	auto upper_bound = std::make_shared<glsl::VectorVariable>("upper_bound", 4, glsl::ShaderVariableType::eFloat);
	auto lower_bound = std::make_shared<glsl::VectorVariable>("lower_bound", 4, glsl::ShaderVariableType::eFloat);

	auto pShader1 = glsl::qmri::ivim_guess_shader(ndata, true);
	auto shaderStr1 = pShader1->compile();
	std::cout << shaderStr1 << std::endl;
	std::cout << "\n\n\n" << std::endl;

	auto pShader2 = glsl::qmri::ivim_partial_nlsq_shader(ndata, true);
	auto shaderStr2 = pShader2->compile();
	std::cout << shaderStr2 << std::endl;
	std::cout << "\n\n\n" << std::endl;

	auto pShader3 = glsl::qmri::ivim_full_nlsq_shader(ndata, true);
	auto shaderStr3 = pShader3->compile();
	std::cout << shaderStr3 << std::endl;
	std::cout << "\n\n\n" << std::endl;

	namespace fs = std::filesystem;
	auto d_path = fs::current_path() / "data" / "ivim_data.vcdat";
	auto c_path = fs::current_path() / "data" / "ivim_bvals.vcdat";

	size_t nelem = fs::file_size(d_path) / ndata / sizeof(float);

	auto mgr = std::make_shared<kp::Manager>();

	auto kp_params = glsl::kp_tensor_from_vector(mgr, params, nelem);
	auto kp_consts = glsl::kp_tensor_from_file(mgr, glsl::ShaderVariableType::eFloat, c_path);
	auto kp_data = glsl::kp_tensor_from_file(mgr, glsl::ShaderVariableType::eFloat, d_path);
	auto kp_bsplit = glsl::kp_tensor_from_vector(mgr, bsplit, 1);
	std::vector<int32_t> data_bsplit(2); data_bsplit[0] = 11; data_bsplit[1] = 15;
	std::memcpy(kp_bsplit->data<int32_t>(), data_bsplit.data(), sizeof(int32_t) * data_bsplit.size());

	auto kp_weights = glsl::kp_tensor_from_vector(mgr, weights, 1);
	std::vector<float> data_weights(21, 1.0f);
	std::memcpy(kp_weights->data<float>(), data_weights.data(), sizeof(float) * data_weights.size());

	auto kp_lambda = glsl::kp_tensor_from_single(mgr, lambda, nelem);
	std::vector<float> data_lambda(nelem, 1.0f);
	std::memcpy(kp_lambda->data<float>(), data_lambda.data(), data_lambda.size() * sizeof(float));

	auto kp_step_type = glsl::kp_tensor_from_single(mgr, step_type, nelem);
	auto kp_nlstep = glsl::kp_tensor_from_vector(mgr, nlstep, nelem);
	auto kp_error = glsl::kp_tensor_from_single(mgr, error, nelem);
	auto kp_new_error = glsl::kp_tensor_from_single(mgr, new_error, nelem);
	auto kp_residuals = glsl::kp_tensor_from_vector(mgr, residuals, nelem);
	auto kp_jacobian = glsl::kp_tensor_from_matrix(mgr, jacobian, nelem);
	auto kp_hessian = glsl::kp_tensor_from_matrix(mgr, hessian, nelem);


	auto spirv1 = glsl::compileSource(shaderStr1);
	auto spirv2 = glsl::compileSource(shaderStr2);
	auto spirv3 = glsl::compileSource(shaderStr3);

	std::vector<std::shared_ptr<kp::Tensor>> shader_inputs = {
		kp_params, kp_consts, kp_data, kp_bsplit, kp_weights, kp_lambda, kp_step_type,
		kp_nlstep, kp_error, kp_new_error, kp_residuals, kp_jacobian, kp_hessian
	};

	kp::Workgroup wg{ (size_t)nelem, 1, 1 };
	
	std::shared_ptr<kp::Algorithm> algo1 = mgr->algorithm(shader_inputs, spirv1, wg);
	std::shared_ptr<kp::Algorithm> algo2 = mgr->algorithm(shader_inputs, spirv2, wg);
	std::shared_ptr<kp::Algorithm> algo3 = mgr->algorithm(shader_inputs, spirv3, wg);

	auto start = std::chrono::steady_clock::now();

	auto seq = mgr->sequence()
		->record<kp::OpTensorSyncDevice>(shader_inputs)
		->record<kp::OpAlgoDispatch>(algo1)
		->record<kp::OpMemoryBarrier>(shader_inputs, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
			vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader)
		->record<kp::OpAlgoDispatch>(algo2)
		->record<kp::OpMemoryBarrier>(shader_inputs, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
			vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader)
		->record<kp::OpAlgoDispatch>(algo3)
		->record<kp::OpTensorSyncLocal>(shader_inputs)
		->eval();

	auto end = std::chrono::steady_clock::now();

	std::cout << "run time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	bool print = true;
	if (print) {
		std::cout.precision(10);
		std::cout << "CONST TYPES: " << std::endl;
		std::cout << "bsplit: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_bsplit, bsplit, 0) << std::endl;
		std::cout << "weights: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_weights, weights, 0) << std::endl;
		std::cout << "consts: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_consts, consts, 0) << std::endl;


		std::cout << "FIRST ELEMENTS" << std::endl;
		std::cout << "residuals: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_residuals, residuals, 0) << std::endl;
		std::cout << "jacobian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_jacobian, jacobian, 0) << std::endl;
		std::cout << "hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_hessian, hessian, 0) << std::endl;
		std::cout << "data: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_data, data, 0) << std::endl;
		std::cout << "params: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_params, params, 0) << std::endl;
		std::cout << "lambda: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda, lambda, 0) << std::endl;
		std::cout << "step_type: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_step_type, step_type, 0) << std::endl;
		std::cout << "step: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_nlstep, nlstep, 0) << std::endl;
	

		std::cout << "MID ELEMENTS" << std::endl;
		std::cout << "residuals: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_residuals, residuals, nelem / 2) << std::endl;
		std::cout << "jacobian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_jacobian, jacobian, nelem / 2) << std::endl;
		std::cout << "hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_hessian, hessian, nelem / 2) << std::endl;
		std::cout << "data: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_data, data, nelem / 2) << std::endl;
		std::cout << "params: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_params, params, nelem / 2) << std::endl;
		std::cout << "lambda: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda, lambda, nelem / 2) << std::endl;
		std::cout << "step_type: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_step_type, step_type, nelem / 2) << std::endl;
		std::cout << "step: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_nlstep, nlstep, nelem / 2) << std::endl;

		
		std::cout << "LAST ELEMENTS" << std::endl;
		std::cout << "residuals: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_residuals, residuals, nelem - 1) << std::endl;
		std::cout << "jacobian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_jacobian, jacobian, nelem - 1) << std::endl;
		std::cout << "hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_hessian, hessian, nelem - 1) << std::endl;
		std::cout << "data: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_data, data, nelem - 1) << std::endl;
		std::cout << "params: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_params, params, nelem - 1) << std::endl;
		std::cout << "lambda: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda, lambda, nelem - 1) << std::endl;
		std::cout << "step_type: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_step_type, step_type, nelem - 1) << std::endl;
		std::cout << "step: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_nlstep, nlstep, nelem - 1) << std::endl;
	}

	auto p_path = fs::current_path() / "data" / "ivim_params.vcdat";
	glsl::kp_tensor_to_file(kp_params, glsl::ShaderVariableType::eFloat, p_path);

}

int main() {

	//std::cout << "test_avk: " << std::endl;
	//return test_avk();

	run_qmri_ivim();

	//run_compute_and_render_app();

	//return 0;
}



