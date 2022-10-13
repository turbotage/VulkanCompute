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
	
	params[0] = lin_param2[0];
	params[1] = 1.0 - (lin_param1[0] / lin_param2[0]);
	params[2] = -lin_param2[1];
	params[3] = -lin_param1[1];
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

			if (!((params->getType() == ShaderVariableType::FLOAT) ||
				(params->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}

			if (bsplit->getType() != ShaderVariableType::INT) {
				throw std::runtime_error("bsplit must have integer type");
			}

		}

		ui16 ndim = data->getNDim();

		bool single_precision = true;
		if (params->getType() == ShaderVariableType::DOUBLE)
			single_precision = false;

		auto func = ivim_guess(ndim, single_precision);
		auto uniqueid = ivim_guess_uniqueid(ndim, single_precision);

		return FunctionApplier{ func, nullptr, {params, consts, data, bsplit}, uniqueid };
	}
	


	export std::shared_ptr<glsl::AutogenShader> ivim_guess_shader(vc::ui16 ndata, bool single_precision)
	{
		std::shared_ptr<AutogenShader> pShader = std::make_shared<AutogenShader>();

		auto params = std::make_shared<glsl::VectorVariable>("params", 4, ShaderVariableType::FLOAT);
		auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, 1, ShaderVariableType::FLOAT);
		auto data = std::make_shared<glsl::VectorVariable>("data", ndata, ShaderVariableType::FLOAT);
		auto bsplit = std::make_shared<glsl::VectorVariable>("bsplit", 2, ShaderVariableType::INT);

		pShader->addVector(params, 0, IOShaderVariableType::INPUT_OUTPUT_TYPE);
		pShader->addMatrix(consts, 1, IOShaderVariableType::CONST_TYPE);
		pShader->addVector(data, 2, IOShaderVariableType::INPUT_TYPE);
		pShader->addVector(bsplit, 3, IOShaderVariableType::CONST_TYPE);

		pShader->apply(ivim_guess(params, consts, data, bsplit));

		return std::move(pShader);
	}
	
	export std::shared_ptr<glsl::AutogenShader> ivim_nlsq_shader(vc::ui16 ndata, bool single_precision)
	{
		using namespace nlsq;

		std::shared_ptr<AutogenShader> pShader = std::make_shared<AutogenShader>();

		auto params = std::make_shared<glsl::VectorVariable>("params", 4, ShaderVariableType::FLOAT);
		auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, 1, ShaderVariableType::FLOAT);
		auto data = std::make_shared<glsl::VectorVariable>("data", ndata, ShaderVariableType::FLOAT);
		auto weights = std::make_shared<glsl::VectorVariable>("weights", ndata, ShaderVariableType::FLOAT);
		auto lambda = std::make_shared<glsl::SingleVariable>("lambda", ShaderVariableType::FLOAT, std::nullopt);
		auto step_type = std::make_shared<glsl::SingleVariable>("step_type", ShaderVariableType::INT, std::nullopt);

		auto mu = std::make_shared<glsl::SingleVariable>("mu", ShaderVariableType::FLOAT, "0.25");
		auto eta = std::make_shared<glsl::SingleVariable>("eta", ShaderVariableType::FLOAT, "0.75");
		auto acc = std::make_shared<glsl::SingleVariable>("acc", ShaderVariableType::FLOAT, "0.2");
		auto dec = std::make_shared<glsl::SingleVariable>("dec", ShaderVariableType::FLOAT, "5.0");

		auto nlstep = std::make_shared<glsl::VectorVariable>("nlstep", 4, ShaderVariableType::FLOAT);
		auto error = std::make_shared<glsl::SingleVariable>("error", ShaderVariableType::FLOAT, std::nullopt);
		auto new_error = std::make_shared<glsl::SingleVariable>("new_error", ShaderVariableType::FLOAT, std::nullopt);
		auto residuals = std::make_shared<glsl::VectorVariable>("residuals", ndata, ShaderVariableType::FLOAT);
		auto jacobian = std::make_shared<glsl::MatrixVariable>("jacobian", ndata, 4, ShaderVariableType::FLOAT);
		auto hessian = std::make_shared<glsl::MatrixVariable>("hessian", 4, 4, ShaderVariableType::FLOAT);
		auto lambda_hessian = std::make_shared<glsl::MatrixVariable>("lambda_hessian", 4, 4, ShaderVariableType::FLOAT);
		auto upper_bound = std::make_shared<glsl::VectorVariable>("upper_bound", 4, ShaderVariableType::FLOAT);
		auto lower_bound = std::make_shared<glsl::VectorVariable>("lower_bound", 4, ShaderVariableType::FLOAT);

		pShader->addVector(params, 0, IOShaderVariableType::INPUT_OUTPUT_TYPE);
		pShader->addMatrix(consts, 1, IOShaderVariableType::CONST_TYPE);
		pShader->addVector(data, 2, IOShaderVariableType::INPUT_TYPE);
		pShader->addVector(weights, 4, IOShaderVariableType::CONST_TYPE);
		pShader->addSingle(lambda, 5, IOShaderVariableType::INPUT_OUTPUT_TYPE);
		pShader->addSingle(step_type, 6, IOShaderVariableType::INPUT_OUTPUT_TYPE);
		pShader->addVector(nlstep, 7, IOShaderVariableType::OUTPUT_TYPE);
		pShader->addSingle(error, 8, IOShaderVariableType::OUTPUT_TYPE);
		pShader->addSingle(new_error, 9, IOShaderVariableType::OUTPUT_TYPE);
		pShader->addVector(residuals, 10, IOShaderVariableType::OUTPUT_TYPE);
		pShader->addMatrix(jacobian, 11, IOShaderVariableType::OUTPUT_TYPE);
		pShader->addMatrix(hessian, 12, IOShaderVariableType::OUTPUT_TYPE);
		pShader->addMatrix(lambda_hessian, 13, IOShaderVariableType::LOCAL_TYPE);
		pShader->addVector(upper_bound, 14, IOShaderVariableType::LOCAL_TYPE);
		pShader->addVector(lower_bound, 15, IOShaderVariableType::LOCAL_TYPE);

		std::vector<std::string> vars = { "s0","f","d1","d2","b" };
		std::string expresh = "s0*(f*exp(-b*d1)+(1-f)*exp(-b*d2))";
		expression::Expression expr(expresh, vars);
		SymbolicContext context;
		context.insert_const(std::make_pair("b", 0));
		context.insert_param(std::make_pair("s0", 0));
		context.insert_param(std::make_pair("f", 1));
		context.insert_param(std::make_pair("d1", 2));
		context.insert_param(std::make_pair("d2", 3));

		pShader->apply_scope(ForScope::make("int i = 0; i < 5; ++i"))
			.apply(nlsq::nlsq_slmh_w_step(
				expr, context,
				params, consts, data, weights,
				lambda, step_type, mu, eta, acc, dec,
				nlstep, error, new_error,
				residuals, jacobian, hessian, lambda_hessian
			));

		return pShader;
	}


}
}