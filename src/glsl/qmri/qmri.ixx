module;

export module qmri;

import <string>;
import <memory>;
import <vector>;
import <functional>;
import <optional>;
import <stdexcept>;

import vc;
import util;

import glsl;
import variable;
import function;
import lsq;

import func_factory;
import shader;

import expr;
import nlsq;

namespace glsl {
namespace qmri {

	using namespace vc;

	using vecptrfunc = std::vector<std::shared_ptr<Function>>;
	using refvecptrfunc = refw<std::vector<std::shared_ptr<Function>>>;

	// S = S0 * (f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D))

	export std::string ivim_guess_uniqueid(vc::ui16 ndata, bool single_precision)
	{
		return std::to_string(ndata) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<glsl::Function> ivim_guess(vc::ui16 ndata, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void ivim_guess_UNIQUEID(inout float params[4], in float bvals[ndata], in float data[ndata], in int bsplit[2])
{
	float log_data[ndata];
	for (int i = 0; i < ndata; ++i) {
		log_data[i] = log(data[i]);
	}
	
	float lin_param1[2]; // S0_prime, D
	lsq_linear2_upper_LL2UID(bvals, log_data, bsplit[1], lin_param1);
	lin_param1[0] = exp(lin_param1[0]);
	
	float lin_param2[2]; // S0, D_star_prime
	lsq_linear2_lower_LL2LID(bvals, log_data, bsplit[0], lin_param2);
	lin_param2[0] = exp(lin_param2[0]);	
	
	params[0] = abs(lin_param2[0]);
	params[1] = 1.0 - (lin_param1[0] / lin_param2[0]);
	params[1] = clamp(params[1], 0.0, 1.0);

	params[2] = abs(lin_param2[1]);
	params[3] = abs(lin_param1[1]);
	
	//params[2] = max(params[2], 1.2*params[3]);
}
)glsl";

		std::string uniqueid = ivim_guess_uniqueid(ndata, single_precision);

		std::function<std::string()> code_func = [ndata, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "LL2UID", lsq::lsq_linear2_upper_uniqueid(ndata, single_precision));
			util::replace_all(temp, "LL2LID", lsq::lsq_linear2_lower_uniqueid(ndata, single_precision));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"ivim_guess_" + uniqueid,
			std::vector<size_t>{ size_t(ndata), size_t(single_precision) },
			code_func,
			std::make_optional<vecptrfunc>({
				lsq::lsq_linear2_upper(ndata, single_precision),
				lsq::lsq_linear2_lower(ndata, single_precision)
				})
			);
	}
	
	export FunctionApplier ivim_guess(
		const std::shared_ptr<VectorVariable>& params,
		const std::shared_ptr<MatrixVariable>& consts,
		const std::shared_ptr<VectorVariable>& data,
		const std::shared_ptr<VectorVariable>& bsplit) 
	{
		// type and dims check
		{
			if (consts->getNDim1() != data->getNDim())
			{
				throw std::runtime_error("consts dim1 must agree with data dim");
			}

			if (!((ui16)params->getType() &
				(ui16)consts->getType() &
				(ui16)data->getType()))
			{
				throw std::runtime_error("All inputs must have same type");
			}

			if (!((params->getType() == ShaderVariableType::eFloat) ||
				(params->getType() == ShaderVariableType::eDouble))) {
				throw std::runtime_error("Inputs must have float or double type");
			}

			if (bsplit->getType() != ShaderVariableType::eInt) {
				throw std::runtime_error("bsplit must have integer type");
			}

		}

		ui16 ndim = data->getNDim();

		bool single_precision = true;
		if (params->getType() == ShaderVariableType::eDouble)
			single_precision = false;

		auto func = ivim_guess(ndim, single_precision);
		auto uniqueid = ivim_guess_uniqueid(ndim, single_precision);

		return FunctionApplier{ func, nullptr, {params, consts, data, bsplit}, uniqueid };
	}
	


	export std::shared_ptr<glsl::AutogenShader> ivim_guess_shader(vc::ui16 ndata, bool single_precision)
	{
		std::shared_ptr<AutogenShader> pShader = std::make_shared<AutogenShader>();

		auto params = std::make_shared<glsl::VectorVariable>("params", 4, ShaderVariableType::eFloat);
		auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, 1, ShaderVariableType::eFloat);
		auto data = std::make_shared<glsl::VectorVariable>("data", ndata, ShaderVariableType::eFloat);
		auto bsplit = std::make_shared<glsl::VectorVariable>("bsplit", 2, ShaderVariableType::eInt);
		auto step_type = std::make_shared<glsl::SingleVariable>("step_type", ShaderVariableType::eInt, std::nullopt);
		auto upper_bound = std::make_shared<glsl::VectorVariable>("upper_bound", 4, ShaderVariableType::eFloat);
		auto lower_bound = std::make_shared<glsl::VectorVariable>("lower_bound", 4, ShaderVariableType::eFloat);



		pShader->addVector(params, 0, IOShaderVariableType::INPUT_OUTPUT_TYPE);
		pShader->addMatrix(consts, 1, IOShaderVariableType::CONST_TYPE);
		pShader->addVector(data, 2, IOShaderVariableType::INPUT_TYPE);
		pShader->addVector(bsplit, 3, IOShaderVariableType::CONST_TYPE);
		pShader->addSingle(step_type, 6, IOShaderVariableType::OUTPUT_TYPE);
		pShader->addVector(lower_bound, 7, IOShaderVariableType::OUTPUT_TYPE);
		pShader->addVector(upper_bound, 8, IOShaderVariableType::OUTPUT_TYPE);

		pShader->apply(ivim_guess(params, consts, data, bsplit));

		std::string begin_str =
R"glsl(
step_type = -10;

lower_bound[0] = 0.0;
upper_bound[0] = abs(data[0] * 20.0);

lower_bound[1] = 0.0;
upper_bound[1] = 1.0;

lower_bound[2] = abs(params[2] / 100.0);
upper_bound[2] = abs(params[2] * 100.0);

lower_bound[3] = abs(params[3] / 100.0);
upper_bound[3] = abs(params[3] * 100.0);
)glsl";

		pShader->apply_scope(glsl::TextedScope::make(begin_str));

		return std::move(pShader);
	}
	
	export std::shared_ptr<glsl::AutogenShader> ivim_partial_nlsq_shader(vc::ui16 ndata, bool single_precision)
	{
		using namespace nlsq;

		std::shared_ptr<AutogenShader> pShader = std::make_shared<AutogenShader>();

		auto params = std::make_shared<glsl::VectorVariable>("params", 4, ShaderVariableType::eFloat);
		auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, 1, ShaderVariableType::eFloat);
		auto data = std::make_shared<glsl::VectorVariable>("data", ndata, ShaderVariableType::eFloat);
		auto weights = std::make_shared<glsl::VectorVariable>("weights", ndata, ShaderVariableType::eFloat);
		auto lambda = std::make_shared<glsl::SingleVariable>("lambda", ShaderVariableType::eFloat, std::nullopt);
		auto step_type = std::make_shared<glsl::SingleVariable>("step_type", ShaderVariableType::eInt, std::nullopt);
		auto upper_bound = std::make_shared<glsl::VectorVariable>("upper_bound", 4, ShaderVariableType::eFloat);
		auto lower_bound = std::make_shared<glsl::VectorVariable>("lower_bound", 4, ShaderVariableType::eFloat);

		auto mu = std::make_shared<glsl::SingleVariable>("mu", ShaderVariableType::eFloat, "0.25");
		auto eta = std::make_shared<glsl::SingleVariable>("eta", ShaderVariableType::eFloat, "0.75");
		auto acc = std::make_shared<glsl::SingleVariable>("acc", ShaderVariableType::eFloat, "0.2");
		auto dec = std::make_shared<glsl::SingleVariable>("dec", ShaderVariableType::eFloat, "5.0");

		auto local_params = std::make_shared<glsl::VectorVariable>("local_params", 2, ShaderVariableType::eFloat);
		auto local_consts = std::make_shared<glsl::MatrixVariable>("local_consts", ndata, 3, ShaderVariableType::eFloat);
		auto nlstep = std::make_shared<glsl::VectorVariable>("nlstep", 2, ShaderVariableType::eFloat);
		auto error = std::make_shared<glsl::SingleVariable>("error", ShaderVariableType::eFloat, std::nullopt);
		auto new_error = std::make_shared<glsl::SingleVariable>("new_error", ShaderVariableType::eFloat, std::nullopt);
		auto residuals = std::make_shared<glsl::VectorVariable>("residuals", ndata, ShaderVariableType::eFloat);
		auto gradient = std::make_shared<glsl::VectorVariable>("gradient", 2, ShaderVariableType::eFloat);
		auto jacobian = std::make_shared<glsl::MatrixVariable>("jacobian", ndata, 2, ShaderVariableType::eFloat);
		auto hessian = std::make_shared<glsl::MatrixVariable>("hessian", 2, 2, ShaderVariableType::eFloat);
		auto lambda_hessian = std::make_shared<glsl::MatrixVariable>("lambda_hessian", 2, 2, ShaderVariableType::eFloat);

		auto local_upper_bound = std::make_shared<glsl::VectorVariable>("local_upper_bound", 2, ShaderVariableType::eFloat);
		auto local_lower_bound = std::make_shared<glsl::VectorVariable>("local_lower_bound", 2, ShaderVariableType::eFloat);

		pShader->addVector(params, 0, IOShaderVariableType::INPUT_OUTPUT_TYPE);
		pShader->addMatrix(consts, 1, IOShaderVariableType::CONST_TYPE);
		pShader->addVector(data, 2, IOShaderVariableType::INPUT_TYPE);
		pShader->addVector(weights, 4, IOShaderVariableType::CONST_TYPE);
		pShader->addSingle(lambda, 5, IOShaderVariableType::INPUT_OUTPUT_TYPE);
		pShader->addSingle(step_type, 6, IOShaderVariableType::INPUT_OUTPUT_TYPE);
		pShader->addVector(lower_bound, 7, IOShaderVariableType::INPUT_OUTPUT_TYPE);
		pShader->addVector(upper_bound, 8, IOShaderVariableType::INPUT_OUTPUT_TYPE);

		std::vector<std::string> vars = { "s0","f","d1","d2","b" };
		std::string expresh = "s0*(f*exp(-b*d1)+(1-f)*exp(-b*d2))";
		expression::Expression expr(expresh, vars);
		SymbolicContext context;

		context.insert_const(std::make_pair("b", 0));
		context.insert_const(std::make_pair("s0", 1));
		context.insert_const(std::make_pair("d2", 2));
		context.insert_param(std::make_pair("f", 0));
		context.insert_param(std::make_pair("d1", 1));

		std::string begin_str =
R"glsl(
if (step_type == -10) {
	lambda = 1.0;
}

local_upper_bound[0] = upper_bound[1];
local_upper_bound[1] = upper_bound[2];

local_lower_bound[0] = lower_bound[1];
local_lower_bound[1] = lower_bound[2];

for (int i = 0; i < ndata; ++i) {
	local_consts[i*3 + 0] = consts[i];
	local_consts[i*3 + 1] = params[0];
	local_consts[i*3 + 2] = params[3];
}

local_params[0] = params[1];
local_params[1] = params[2];
)glsl";
		util::replace_all(begin_str, "ndata", std::to_string(ndata));

		pShader->apply_scope(glsl::TextedScope::make(begin_str));

		auto& for_scope = pShader->apply_scope(ForScope::make("int i = 0; i < 4; ++i"));

		for_scope.apply(nlsq::nlsq_slmh_w_step(
			expr, context,
			local_params, local_consts, data, weights,
			lambda, step_type, mu, eta, acc, dec,
			nlstep, error, new_error,
			residuals, gradient, jacobian, hessian, lambda_hessian
		));

		auto was_clamped = std::make_shared<glsl::SingleVariable>("was_clamped", ShaderVariableType::eInt, std::nullopt);

		for_scope.apply(nlsq::nlsq_clamping(
			was_clamped,
			local_params,
			local_upper_bound,
			local_lower_bound));

		auto& if_scope = for_scope.apply_scope(glsl::IfScope::make("was_clamped == 1"));
		if_scope.apply_scope(glsl::TextedScope::make(R"glsl(lambda *= 1.0;)glsl"));

		pShader->apply_scope(glsl::TextedScope::make(
R"glsl(
params[1] = local_params[0];
params[2] = local_params[1];

step_type = -20;
)glsl"));

		return pShader;
	}

	export std::shared_ptr<glsl::AutogenShader> ivim_full_nlsq_shader(vc::ui16 ndata, bool single_precision)
	{
		using namespace nlsq;

		std::shared_ptr<AutogenShader> pShader = std::make_shared<AutogenShader>();

		auto params = std::make_shared<glsl::VectorVariable>("params", 4, ShaderVariableType::eFloat);
		auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, 1, ShaderVariableType::eFloat);
		auto data = std::make_shared<glsl::VectorVariable>("data", ndata, ShaderVariableType::eFloat);
		auto weights = std::make_shared<glsl::VectorVariable>("weights", ndata, ShaderVariableType::eFloat);
		auto lambda = std::make_shared<glsl::SingleVariable>("lambda", ShaderVariableType::eFloat, std::nullopt);
		auto step_type = std::make_shared<glsl::SingleVariable>("step_type", ShaderVariableType::eInt, std::nullopt);

		auto mu = std::make_shared<glsl::SingleVariable>("mu", ShaderVariableType::eFloat, "0.25");
		auto eta = std::make_shared<glsl::SingleVariable>("eta", ShaderVariableType::eFloat, "0.75");
		auto acc = std::make_shared<glsl::SingleVariable>("acc", ShaderVariableType::eFloat, "0.2");
		auto dec = std::make_shared<glsl::SingleVariable>("dec", ShaderVariableType::eFloat, "5.0");

		auto nlstep = std::make_shared<glsl::VectorVariable>("nlstep", 4, ShaderVariableType::eFloat);
		auto error = std::make_shared<glsl::SingleVariable>("error", ShaderVariableType::eFloat, std::nullopt);
		auto new_error = std::make_shared<glsl::SingleVariable>("new_error", ShaderVariableType::eFloat, std::nullopt);
		auto residuals = std::make_shared<glsl::VectorVariable>("residuals", ndata, ShaderVariableType::eFloat);
		auto gradient = std::make_shared<glsl::VectorVariable>("gradient", 4, ShaderVariableType::eFloat);
		auto jacobian = std::make_shared<glsl::MatrixVariable>("jacobian", ndata, 4, ShaderVariableType::eFloat);
		auto hessian = std::make_shared<glsl::MatrixVariable>("hessian", 4, 4, ShaderVariableType::eFloat);
		auto lambda_hessian = std::make_shared<glsl::MatrixVariable>("lambda_hessian", 4, 4, ShaderVariableType::eFloat);
		auto upper_bound = std::make_shared<glsl::VectorVariable>("upper_bound", 4, ShaderVariableType::eFloat);
		auto lower_bound = std::make_shared<glsl::VectorVariable>("lower_bound", 4, ShaderVariableType::eFloat);

		pShader->addVector(params, 0, IOShaderVariableType::INPUT_OUTPUT_TYPE);
		pShader->addMatrix(consts, 1, IOShaderVariableType::CONST_TYPE);
		pShader->addVector(data, 2, IOShaderVariableType::INPUT_TYPE);
		pShader->addVector(weights, 4, IOShaderVariableType::CONST_TYPE);
		pShader->addSingle(lambda, 5, IOShaderVariableType::INPUT_OUTPUT_TYPE);
		pShader->addSingle(step_type, 6, IOShaderVariableType::OUTPUT_TYPE);
		pShader->addVector(lower_bound, 7, IOShaderVariableType::INPUT_OUTPUT_TYPE);
		pShader->addVector(upper_bound, 8, IOShaderVariableType::INPUT_OUTPUT_TYPE);

		/*
		pShader->addVector(nlstep, std::nullopt, IOShaderVariableType::LOCAL_TYPE);
		pShader->addSingle(error, std::nullopt, IOShaderVariableType::LOCAL_TYPE);
		pShader->addSingle(new_error, std::nullopt, IOShaderVariableType::LOCAL_TYPE);
		pShader->addVector(residuals, std::nullopt, IOShaderVariableType::LOCAL_TYPE);
		pShader->addVector(gradient, std::nullopt, IOShaderVariableType::LOCAL_TYPE);
		pShader->addMatrix(jacobian, std::nullopt, IOShaderVariableType::LOCAL_TYPE);
		pShader->addMatrix(hessian, std::nullopt, IOShaderVariableType::LOCAL_TYPE);
		pShader->addMatrix(lambda_hessian, std::nullopt, IOShaderVariableType::LOCAL_TYPE);
		pShader->addVector(upper_bound, 15, IOShaderVariableType::LOCAL_TYPE);
		pShader->addVector(lower_bound, 16, IOShaderVariableType::LOCAL_TYPE);
		*/

		std::vector<std::string> vars = { "s0","f","d1","d2","b" };
		std::string expresh = "s0*(f*exp(-b*d1)+(1-f)*exp(-b*d2))";
		expression::Expression expr(expresh, vars);
		SymbolicContext context;

		context.insert_const(std::make_pair("b", 0));
		context.insert_param(std::make_pair("s0", 0));
		context.insert_param(std::make_pair("f", 1));
		context.insert_param(std::make_pair("d1", 2));
		context.insert_param(std::make_pair("d2", 3));

		/*
		pShader->apply_scope(glsl::TextedScope::make(
R"glsl(
lower_bound[0] = 0;
upper_bound[0] = params[0] * 20;

lower_bound[1] = 0.0;
upper_bound[1] = 1.0;

lower_bound[2] = 0.0;
upper_bound[2] = min(params[2] * 100, 1.0);

lower_bound[3] = 0.0;
upper_bound[3] = min(params[3] * 10, 1.0);
)glsl"
));
		*/

		pShader->apply_scope(glsl::TextedScope::make(
R"glsl(
if (step_type == -20) {
	lambda = 1.0;
}
)glsl"));

		auto& for_scope = pShader->apply_scope(ForScope::make("int i = 0; i < 4; ++i"));

		for_scope.apply(nlsq::nlsq_slmh_w_step(
				expr, context,
				params, consts, data, weights,
				lambda, step_type, mu, eta, acc, dec,
				nlstep, error, new_error,
				residuals, gradient, jacobian, hessian, lambda_hessian
			));

		auto was_clamped = std::make_shared<glsl::SingleVariable>("was_clamped", ShaderVariableType::eInt, std::nullopt);

		for_scope.apply(nlsq::nlsq_clamping(
			was_clamped,
			params,
			upper_bound,
			lower_bound));

		auto& if_scope = for_scope.apply_scope(glsl::IfScope::make("was_clamped == 1"));
		if_scope.apply_scope(glsl::TextedScope::make(R"glsl(lambda *= 1.0;)glsl"));

		return pShader;
	}


}
}