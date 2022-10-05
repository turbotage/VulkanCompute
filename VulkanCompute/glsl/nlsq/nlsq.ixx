module;

export module nlsq;

import <string>;
import <optional>;
import <functional>;
import <memory>;
import <stdexcept>;

import vc;
import glsl;
import util;

import linalg;
import symm;
import solver;

import variable;
import function;
import func_factory;

export import symbolic;
export import nlsq_symbolic;

namespace glsl {
namespace nlsq {

	using namespace vc;

	using vecptrfunc = std::vector<std::shared_ptr<Function>>;
	using refvecptrfunc = refw<std::vector<std::shared_ptr<Function>>>;

	export std::string nlsq_gain_ratio_uniqueid(ui16 nparam, bool single_precission)
	{
		return std::to_string(nparam) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> nlsq_gain_ratio(ui16 nparam, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
float nlsq_gain_UNIQUEID(in float step[nparam], in float neg_gradient[nparam], in float hessian[nparam*nparam], float obj_err, float new_obj_err) {
	return (obj_err - new_obj_err) / (inner_prod_IPID(step, neg_gradient) - weighted_vec_norm2_WVN2ID(hessian, step));
}
)glsl";

		std::string uniqueid = nlsq_gain_ratio_uniqueid(nparam, single_precission);

		std::function<std::string()> code_func = [nparam, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nparam", std::to_string(nparam));
			util::replace_all(temp, "IPID", linalg::inner_prod_uniqueid(nparam, single_precission));
			util::replace_all(temp, "WVN2ID", linalg::weighted_vec_norm2_uniqueid(nparam, single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"nlsq_gain_ratio_" + uniqueid,
			std::vector<size_t>{ size_t(nparam), size_t(single_precission) },
			code_func,
			std::make_optional<vecptrfunc>({
				linalg::inner_prod(nparam, single_precission),
				linalg::weighted_vec_norm2(nparam, single_precission)
				})
		);
	}

	export FunctionApplier nlsq_gain_ratio(
		const std::shared_ptr<SingleVariable>& ratio,
		const std::shared_ptr<VectorVariable>& step,
		const std::shared_ptr<VectorVariable>& gradient,
		const std::shared_ptr<MatrixVariable>& hessian,
		const std::shared_ptr<SingleVariable>& error,
		const std::shared_ptr<SingleVariable>& new_error)
	{
		// type and dimension checks
		{
			if (step->getNDim() != gradient->getNDim()) {
				throw std::runtime_error("step dim and gradient dim must agree");
			}
			if (hessian->getNDim1() != hessian->getNDim2()) {
				throw std::runtime_error("hessian must be square");
			}
			if (hessian->getNDim1() != step->getNDim()) {
				throw std::runtime_error("hessian dim1 must agree with step dim");
			}

			if ((ui16)step->getType() &
				(ui16)gradient->getType() &
				(ui16)hessian->getType() &
				(ui16)error->getType() &
				(ui16)new_error->getType())
			{
				throw std::runtime_error("All inputs must have same type");
			}
			if (!((ratio->getType() == ShaderVariableType::FLOAT) ||
				(ratio->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
			if (!((step->getType() == ShaderVariableType::FLOAT) ||
				(step->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 ndim = step->getNDim();

		bool single_precission = true;
		if (step->getType() == ShaderVariableType::DOUBLE)
			single_precission = false;

		auto func = nlsq_gain_ratio(ndim, single_precission);
		auto uniqueid = nlsq_gain_ratio_uniqueid(ndim, single_precission);

		return FunctionApplier{ func, ratio, {step, gradient, hessian, error, new_error }, uniqueid };
	}


	export std::string nlsq_error_uniqueid(ui16 ndata, bool single_precission)
	{
		return std::to_string(ndata) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> nlsq_error(ui16 ndata, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
float nlsq_error_UNIQUEID(in float res[ndata]) {
	return 0.5 * vec_norm2_VN2ID(res);
}
)glsl";

		std::string uniqueid = nlsq_error_uniqueid(ndata, single_precission);

		std::function<std::string()> code_func = [ndata, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "VN2ID", linalg::vec_norm2_uniqueid(ndata, single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"nlsq_error_" + uniqueid,
			std::vector<size_t>{ size_t(ndata), size_t(single_precission) },
			code_func,
			std::make_optional<vecptrfunc>({
				linalg::vec_norm2(ndata, single_precission)
				})
		);
	}

	export FunctionApplier nlsq_error(
		const std::shared_ptr<SingleVariable>& error,
		const std::shared_ptr<VectorVariable>& res)
	{
		// type and dims check
		{
			if (!((res->getType() == ShaderVariableType::FLOAT) ||
				(res->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 ndim = res->getNDim();

		bool single_precission = true;
		if (res->getType() == ShaderVariableType::DOUBLE)
			single_precission = false;

		auto func = nlsq_error(ndim, single_precission);
		auto uniqueid = nlsq_error_uniqueid(ndim, single_precission);

		return FunctionApplier{ func, nullptr, {res}, uniqueid };
	}

	/*
	export std::string nlsq_slm_step_uniqueid(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export std::shared_ptr<::glsl::Function> nlsq_slm_step(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void nlsq_slm_step_UNIQUEID(
	inout float params[nparam], in float consts[ndata*nconst], in float data[ndata],
	inout float lambda, inout int step_type, float mu, float eta, float inc, float dec, float tol) 
{
	float residuals[ndata];
	float jacobian[ndata*nparam];

	nlsq_residuals_jacobian_NRJID(params, consts, data, residuals, jacobian);
	
	float hessian[nparam*nparam];

	mul_transpose_mat_MTMID(jacobian, hessian);

	float lambda_hessian[nparam*nparam];

	mat_add_ldiag_out_MALOID(hessian, lambda, lambda_hessian);

	gmw81_GID(lambda_hessian);

	float gradient[nparam];
	
	mul_transpose_vec_MTVID(jacobian, residuals, gradient);
	vec_neg_VNEGID(gradient);
	
	float step[nparam];

	ldl_solve_LSID(lambda_hessian, gradient, step);

	float new_params[nparam];

	add_vec_vec_AVVID(params, step, new_params);
	
	float error = nlsq_error_NEID(residuals);	
	
	nlsq_residuals_NRID(new_params, consts, data, residuals);

	float new_error = nlsq_error_NEID(residuals);

	float gain_ratio = nlsq_gain_NGID(step, gradient, hessian, error, new_error);
	
	step_type = 0;
	if (new_error < error || gain_ratio > eta) {
		copy_vec(new_params, params);
		step_type = 10;
	}
	if (gain_ratio > eta) {
		lambda *= inc;
		step_type += 1;
	}
	if (gain_ratio < mu) {
		lambda *= dec;
		step -= 1;
	}
	
	// convergence
	mul_mat_vec_MMVID(jacobian, step, new_params);
	float jp_norm = vec_norm_VNORMID(new_params);

	if (jp_norm < tol * (1 + sqrt(2*error))) {
		step += 10;
	}
}
)glsl";

		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		std::string uniqueid = nlsq_slm_step_uniqueid(expr, context, nparam, ndata, nconst, single_precission);

		std::function<std::string()> code_func = [ndata, nparam, nconst, single_precission, &expr, &context, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "NRJID", nlsq_residuals_jacobian_uniqueid(expr, context, ndata, nparam, nconst, single_precission));
			util::replace_all(temp, "MTMID", linalg::mul_transpose_mat_uniqueid(ndata, nparam, single_precission));
			util::replace_all(temp, "MALOID", linalg::mat_add_ldiag_out_uniqueid(nparam, single_precission));
			util::replace_all(temp, "GID", linalg::gmw81_uniqueid(nparam, single_precission));
			util::replace_all(temp, "MTVID", linalg::mul_transpose_vec_uniqueid(ndata, nparam, single_precission));
			util::replace_all(temp, "VNEGID", linalg::vec_neg_uniqueid(nparam, single_precission));
			util::replace_all(temp, "LSID", linalg::ldl_solve_uniqueid(nparam, single_precission));
			util::replace_all(temp, "AVVID", linalg::add_vec_vec_uniqueid(nparam, single_precission));
			util::replace_all(temp, "NEID", nlsq_error_uniqueid(ndata, single_precission));
			util::replace_all(temp, "NRID", nlsq_residuals_uniqueid(expr, context, ndata, nparam, nconst, single_precission));
			util::replace_all(temp, "NGID", nlsq_gain_ratio_uniqueid(nparam, single_precission));
			util::replace_all(temp, "MMVID", linalg::mul_mat_vec_uniqueid(ndata, nparam, single_precission));
			util::replace_all(temp, "VNORMID", linalg::vec_norm_uniqueid(ndata, single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"nlsq_slm_step_" + uniqueid,
			std::vector<size_t>{ hashed_expr, size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::make_optional<vecptrfunc>({ 
				nlsq_residuals_jacobian(expr, context, ndata, nparam, nconst, single_precission),
				linalg::mul_transpose_mat(ndata, nparam, single_precission),
				linalg::mat_add_ldiag_out(nparam, single_precission),
				linalg::gmw81(nparam, single_precission),
				linalg::mul_transpose_vec(ndata, nparam, single_precission),
				linalg::vec_neg(nparam, single_precission),
				linalg::ldl_solve(nparam, single_precission),
				linalg::add_vec_vec(nparam, single_precission),
				nlsq_error(ndata, single_precission),
				nlsq_residuals(expr, context, ndata, nparam, nconst, single_precission),
				nlsq_gain_ratio(nparam, single_precission),
				linalg::mul_mat_vec(ndata, nparam, single_precission),
				linalg::vec_norm(ndata, single_precission)
				})
		);
	}
	*/

	export enum class StepType {
		NO_STEP,
		NO_STEP_DAMPING_INCREASED,
		STEP_DAMPING_UNCHANGED,
		STEP_DAMPING_DECREASED
	};

	export FunctionApplier nlsq_slmh_step(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		const std::shared_ptr<VectorVariable>& params,
		const std::shared_ptr<MatrixVariable>& consts,
		const std::shared_ptr<VectorVariable>& data,
		const std::shared_ptr<VectorVariable>& residuals, 
		const std::shared_ptr<MatrixVariable>& jacobian,
		const std::shared_ptr<MatrixVariable>& hessian,
		const std::shared_ptr<SingleVariable>& lambda,
		const std::shared_ptr<SingleVariable>& mu,
		const std::shared_ptr<SingleVariable>& eta,
		const std::shared_ptr<VectorVariable>& step,
		const std::shared_ptr<SingleVariable>& step_type)
	{
		// type and dimension checks
		/*
		nlsq_residuals_jacobian_hessian makes all the checks for residuals, jacobian, hessian, data, params, 
		*/

		glsl::FunctionFactory factory("nlsq_slmh_step", ShaderVariableType::VOID);

		factory.addVector(residuals, FunctionFactory::InputType::INOUT);
		factory.addMatrix(jacobian, FunctionFactory::InputType::INOUT);
		factory.addMatrix(hessian, FunctionFactory::InputType::INOUT);
		factory.addVector(data, FunctionFactory::InputType::IN);
		factory.addVector(params, FunctionFactory::InputType::INOUT);
		factory.addMatrix(consts, FunctionFactory::InputType::IN);
		factory.addSingle(lambda, FunctionFactory::InputType::INOUT);
		factory.addSingle(step_type, FunctionFactory::InputType::INOUT);

		auto& if_scope = factory.apply_scope(IfScope::make(
			"step_type > " + std::to_string((int)StepType::STEP_DAMPING_UNCHANGED)));
			
		if_scope.apply(nlsq_residuals_jacobian_hessian(expr, context,
			params, consts, data, residuals, jacobian, hessian));

		auto lambda_hessian = std::make_shared<glsl::MatrixVariable>(
			"lambda_hessian", hessian->getNDim1(), hessian->getNDim2(), hessian->getType());

		factory.apply(linalg::mul_transpose_mat(jacobian, lambda_hessian));

		factory.apply(linalg::mat_add_ldiag_out(hessian, lambda, lambda_hessian));

		factory.apply(linalg::gmw81(lambda_hessian));

		auto gradient = std::make_shared<VectorVariable>(
			"gradient", params->getNDim(), params->getType());

		factory.apply(linalg::mul_transpose_vec(jacobian, residuals, gradient));
		factory.apply(linalg::vec_neg(gradient));

		factory.apply(linalg::ldl_solve(lambda_hessian, gradient, step));

		auto new_params = std::make_shared<VectorVariable>(
			"new_params", params->getNDim(), params->getType());

		factory.apply(linalg::add_vec_vec(params, step, new_params));

		auto error = std::make_shared<SingleVariable>(
			"error", params->getType(), std::nullopt);

		factory.apply(nlsq_error(error, residuals));

		auto new_error = std::make_shared<SingleVariable>(
			"new_error", params->getType(), std::nullopt);

		factory.apply(nlsq_error(new_error, new_params));

		auto gain_ratio = std::make_shared<SingleVariable>(
			"gain_ratio", params->getType(), std::nullopt);

		factory.apply(nlsq_gain_ratio(gain_ratio, step, gradient, hessian, error, new_error));

		//auto& if_step = factory.apply_scope(IfScope::make("new_error < error || gain_ratio > eta"));

		return factory.build_applier();
	}
	

}
}

