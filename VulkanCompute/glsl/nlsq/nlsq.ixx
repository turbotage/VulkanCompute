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

import expr;

import linalg;
import symm;
import solver;
import permute;

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
float nlsq_gain_ratio_UNIQUEID(in float nlstep[nparam], in float neg_gradient[nparam], in float hessian[nparam*nparam], float obj_err, float new_obj_err) {
	return (obj_err - new_obj_err) / (inner_prod_IPID(nlstep, neg_gradient) - weighted_vec_norm2_WVN2ID(hessian, nlstep));
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

			if (!((ui16)step->getType() &
				(ui16)gradient->getType() &
				(ui16)hessian->getType() &
				(ui16)error->getType() &
				(ui16)new_error->getType()))
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

		return FunctionApplier{ func, error, {res}, uniqueid };
	}


	export enum class StepType {
		NO_STEP = 1,
		DAMPING_INCREASED = 2,
		DAMPING_DECREASED = 4,
		STEP = 8
	};



	export std::string nlsq_slmj_step_uniqueid(const expression::Expression& expr, const glsl::SymbolicContext& context,
		vc::ui16 ndata, vc::ui16 nparam, vc::ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export std::shared_ptr<::glsl::Function> nlsq_slmj_step(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		vc::ui16 ndata, vc::ui16 nparam, vc::ui16 nconst, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void nlsq_slmj_step_UNIQUEID(
	inout float params[nparam], in float consts[ndata*nconst], in float data[ndata],
	inout float lambda, inout int step_type, float mu, float eta, float acc, float dec,
	inout float nlstep[nparam], inout float error, inout float new_error,
	inout float residuals[ndata], inout float jacobian[ndata*nparam], inout float hessian[nparam*nparam]) 
{

	if (step_type < STEP_TYPE_STEP) {
		nlsq_residuals_jacobian_NRJID(params, consts, data, residuals, jacobian);
		mul_transpose_mat_MTMID(jacobian, hessian);
	}	

	float lambda_hessian[nparam*nparam] = hessian;
	mat_add_ldiag_MALID(lambda_hessian, lambda);

	int perm[nparam];
	diagonal_pivoting_DPID(lambda_hessian, perm);
	gmw81_G81ID(lambda_hessian);
	
	float gradient[nparam];
	mul_transpose_vec_MTVID(jacobian, residuals, gradient);
	vec_neg_VNID(gradient);
	
	float steplike[nparam];
	permute_vec_PVID(gradient, perm, steplike);
	
	ldl_solve_LSID(lambda_hessian, steplike, nlstep);

	permute_o_vec_POVID(nlstep, perm, nlstep);	

	add_vec_vec_AVVID(params, nlstep, steplike);
	
	error = nlsq_error_NEID(residuals);

	float new_residuals[ndata];
	nlsq_residuals_NRID(steplike, consts, data, new_residuals);

	new_error = nlsq_error_NEID(new_residuals);

	float gain_ratio = nlsq_gain_ratio_NGRID(nlstep, gradient, hessian, error, new_error);
	
	step_type = 0;
	if (new_error < error && gain_ratio > mu) {
		params = steplike;
		step_type += STEP_TYPE_STEP;

		if (gain_ratio > eta) {
			lambda *= acc;
			step_type += STEP_TYPE_DECREASED;
		}
	}
	else {
		step_type += STEP_TYPE_NOSTEP;
	}
	
	if (gain_ratio < mu) {
		lambda *= dec;
		step_type += STEP_TYPE_INCREASED;
	}
	
}
)glsl";

		using namespace glsl::linalg;
		using namespace glsl::nlsq;

		std::string uniqueid = nlsq_slmj_step_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());

		std::function<std::string()> code_func =
			[expr, context, ndata, nparam, nconst, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "nparam", std::to_string(nparam));
			util::replace_all(temp, "nconst", std::to_string(nconst));

			util::replace_all(temp, "STEP_TYPE_STEP", std::to_string(static_cast<int>(StepType::STEP)));
			util::replace_all(temp, "STEP_TYPE_DECREASED", std::to_string(static_cast<int>(StepType::DAMPING_DECREASED)));
			util::replace_all(temp, "STEP_TYPE_NOSTEP", std::to_string(static_cast<int>(StepType::NO_STEP)));
			util::replace_all(temp, "STEP_TYPE_INCREASED", std::to_string(static_cast<int>(StepType::DAMPING_INCREASED)));

			util::replace_all(temp, "NRJID", nlsq_residuals_jacobian_uniqueid(expr,
				context, ndata, nparam, nconst, single_precission));
			util::replace_all(temp, "MTMID", mul_transpose_mat_uniqueid(ndata, nparam, single_precission));
			util::replace_all(temp, "MALID", mat_add_ldiag_uniqueid(nparam, single_precission));
			util::replace_all(temp, "DPID", diagonal_pivoting_uniqueid(nparam, single_precission));
			util::replace_all(temp, "G81ID", gmw81_uniqueid(nparam, single_precission));
			util::replace_all(temp, "MTVID", mul_transpose_vec_uniqueid(ndata, nparam, single_precission));
			util::replace_all(temp, "VNID", vec_neg_uniqueid(nparam, single_precission));
			util::replace_all(temp, "PVID", permute_vec_uniqueid(nparam, single_precission));
			util::replace_all(temp, "LSID", ldl_solve_uniqueid(nparam, single_precission));
			util::replace_all(temp, "POVID", permute_o_vec_uniqueid(nparam, single_precission));
			util::replace_all(temp, "AVVID", add_vec_vec_uniqueid(nparam, single_precission));
			util::replace_all(temp, "NEID", nlsq_error_uniqueid(ndata, single_precission));
			util::replace_all(temp, "NRID", nlsq_residuals_uniqueid(expr,
				context, ndata, nparam, nconst, single_precission));
			util::replace_all(temp, "NGRID", nlsq_gain_ratio_uniqueid(nparam, single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"nlsq_slmj_step_" + uniqueid,
			std::vector<size_t>{ hashed_expr, size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::make_optional<vecptrfunc>({
				nlsq_residuals_jacobian(expr, context,
					ndata, nparam, nconst, single_precission),
				mul_transpose_mat(ndata, nparam, single_precission),
				mat_add_ldiag(nparam, single_precission),
				diagonal_pivoting(nparam, single_precission),
				gmw81(nparam, single_precission),
				mul_transpose_vec(ndata, nparam, single_precission),
				vec_neg(nparam, single_precission),
				permute_vec(nparam, single_precission),
				ldl_solve(nparam, single_precission),
				permute_o_vec(nparam, single_precission),
				add_vec_vec(nparam, single_precission),
				nlsq_error(ndata, single_precission),
				nlsq_residuals(expr, context,
					ndata, nparam, nconst, single_precission),
				nlsq_gain_ratio(nparam, single_precission)
				})
			);
	}

	export FunctionApplier nlsq_slmj_step(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		const std::shared_ptr<VectorVariable>& params,
		const std::shared_ptr<MatrixVariable>& consts,
		const std::shared_ptr<VectorVariable>& data,
		const std::shared_ptr<SingleVariable>& lambda,
		const std::shared_ptr<SingleVariable>& step_type,
		const std::shared_ptr<SingleVariable>& mu,
		const std::shared_ptr<SingleVariable>& eta,
		const std::shared_ptr<SingleVariable>& acc,
		const std::shared_ptr<SingleVariable>& dec,
		const std::shared_ptr<VectorVariable>& nlstep,
		const std::shared_ptr<SingleVariable>& error,
		const std::shared_ptr<SingleVariable>& new_error,
		const std::shared_ptr<VectorVariable>& residuals,
		const std::shared_ptr<MatrixVariable>& jacobian,
		const std::shared_ptr<MatrixVariable>& hessian)
	{
		// type and dimension checks
		{
			if (residuals->getNDim() != jacobian->getNDim1()) {
				throw std::runtime_error("residuals dim and jacobian dim1 must agree");
			}
			if (jacobian->getNDim2() != hessian->getNDim1()) {
				throw std::runtime_error("jacobian dim2 and hessian dim1 must agree");
			}
			if (!hessian->isSquare()) {
				throw std::runtime_error("hessian isn't square");
			}
			if (params->getNDim() != jacobian->getNDim2()) {
				throw std::runtime_error("params dim and jacobian dim2 must agree");
			}
			if (residuals->getNDim() != data->getNDim()) {
				throw std::runtime_error("residuals dim and data dim must agree");
			}
			if (residuals->getNDim() != consts->getNDim1()) {
				throw std::runtime_error("residuals dim and consts dim1 must agree");
			}
			if (nlstep->getNDim() != params->getNDim()) {
				throw std::runtime_error("step dim and params dim doesn't agree");
			}

			if (!((ui16)residuals->getType() &
				(ui16)jacobian->getType() &
				(ui16)hessian->getType() &
				(ui16)params->getType() &
				(ui16)data->getType() &
				(ui16)consts->getType() &
				(ui16)lambda->getType() &
				(ui16)nlstep->getType()
				))
			{
				throw std::runtime_error("All inputs must have same type");
			}
			if (!((params->getType() == ShaderVariableType::FLOAT) ||
				(params->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 ndata = residuals->getNDim();
		ui16 nparam = params->getNDim();
		ui16 nconst = consts->getNDim2();

		bool single_precission = true;
		if (residuals->getType() == ShaderVariableType::DOUBLE)
			single_precission = false;

		auto func = nlsq_slmj_step(expr, context, ndata, nparam, nconst, single_precission);

		auto uniqueid = nlsq_slmj_step_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		return FunctionApplier{ func, nullptr,
			{params, consts, data, lambda, step_type, mu, eta,
			acc, dec, nlstep, error, new_error, residuals,
			jacobian, hessian },
			uniqueid };
	}



	export std::string nlsq_slmh_step_uniqueid(const expression::Expression& expr, const glsl::SymbolicContext& context,
		vc::ui16 ndata, vc::ui16 nparam, vc::ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export std::shared_ptr<::glsl::Function> nlsq_slmh_step(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		vc::ui16 ndata, vc::ui16 nparam, vc::ui16 nconst, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void nlsq_slmh_step_UNIQUEID(
	inout float params[nparam], in float consts[ndata*nconst], in float data[ndata],
	inout float lambda, inout int step_type, float mu, float eta, float acc, float dec,
	inout float nlstep[nparam], inout float error, inout float new_error,
	inout float residuals[ndata], inout float jacobian[ndata*nparam], 
	inout float hessian[nparam*nparam], inout float lambda_hessian[nparam*nparam]) 
{
	nlsq_residuals_jacobian_hessian_l_NRJHLID(params, consts, data, lambda, residuals, jacobian, hessian, lambda_hessian);
	int perm[nparam];
	diagonal_pivoting_DPID(lambda_hessian, perm);
	gmw81_G81ID(lambda_hessian);
	
	float gradient[nparam];
	mul_transpose_vec_MTVID(jacobian, residuals, gradient);
	vec_neg_VNID(gradient);
	
	float steplike[nparam];
	permute_vec_PVID(gradient, perm, steplike);
	
	ldl_solve_LSID(lambda_hessian, steplike, nlstep);

	permute_o_vec_POVID(nlstep, perm, nlstep);	

	add_vec_vec_AVVID(params, nlstep, steplike);
	
	error = nlsq_error_NEID(residuals);

	float new_residuals[ndata];
	nlsq_residuals_NRID(steplike, consts, data, new_residuals);

	new_error = nlsq_error_NEID(new_residuals);

	float gain_ratio = nlsq_gain_ratio_NGRID(nlstep, gradient, hessian, error, new_error);
	
	step_type = 0;
	if (new_error < error && gain_ratio > mu) {
		params = steplike;
		step_type += STEP_TYPE_STEP;

		if (gain_ratio > eta) {
			lambda *= acc;
			step_type += STEP_TYPE_DECREASED;
		}
	}
	else {
		step_type += STEP_TYPE_NOSTEP;
	}
	
	if (gain_ratio < mu) {
		lambda *= dec;
		step_type += STEP_TYPE_INCREASED;
	}
	
}
)glsl";

		using namespace glsl::linalg;
		using namespace glsl::nlsq;

		std::string uniqueid = nlsq_slmh_step_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());

		std::function<std::string()> code_func =
			[expr, context, ndata, nparam, nconst, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "nparam", std::to_string(nparam));
			util::replace_all(temp, "nconst", std::to_string(nconst));

			util::replace_all(temp, "STEP_TYPE_STEP", std::to_string(static_cast<int>(StepType::STEP)));
			util::replace_all(temp, "STEP_TYPE_DECREASED", std::to_string(static_cast<int>(StepType::DAMPING_DECREASED)));
			util::replace_all(temp, "STEP_TYPE_NOSTEP", std::to_string(static_cast<int>(StepType::NO_STEP)));
			util::replace_all(temp, "STEP_TYPE_INCREASED", std::to_string(static_cast<int>(StepType::DAMPING_INCREASED)));

			util::replace_all(temp, "NRJHLID", nlsq_residuals_jacobian_hessian_l_uniqueid(expr, 
				context, ndata, nparam, nconst, single_precission));
			util::replace_all(temp, "DPID", diagonal_pivoting_uniqueid(nparam, single_precission));
			util::replace_all(temp, "G81ID", gmw81_uniqueid(nparam, single_precission));
			util::replace_all(temp, "MTVID", mul_transpose_vec_uniqueid(ndata, nparam, single_precission));
			util::replace_all(temp, "VNID", vec_neg_uniqueid(nparam, single_precission));
			util::replace_all(temp, "PVID", permute_vec_uniqueid(nparam, single_precission));
			util::replace_all(temp, "LSID", ldl_solve_uniqueid(nparam, single_precission));
			util::replace_all(temp, "POVID", permute_o_vec_uniqueid(nparam, single_precission));
			util::replace_all(temp, "AVVID", add_vec_vec_uniqueid(nparam, single_precission));
			util::replace_all(temp, "NEID", nlsq_error_uniqueid(ndata, single_precission));
			util::replace_all(temp, "NRID", nlsq_residuals_uniqueid(expr,
				context, ndata, nparam, nconst, single_precission));
			util::replace_all(temp, "NGRID", nlsq_gain_ratio_uniqueid(nparam, single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"nlsq_slmh_step_" + uniqueid,
			std::vector<size_t>{ hashed_expr, size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::make_optional<vecptrfunc>({
				nlsq_residuals_jacobian_hessian_l(expr, context,
					ndata, nparam, nconst, single_precission),
				diagonal_pivoting(nparam, single_precission),
				gmw81(nparam, single_precission),
				mul_transpose_vec(ndata, nparam, single_precission),
				vec_neg(nparam, single_precission),
				permute_vec(nparam, single_precission),
				ldl_solve(nparam, single_precission),
				permute_o_vec(nparam, single_precission),
				add_vec_vec(nparam, single_precission),
				nlsq_error(ndata, single_precission),
				nlsq_residuals(expr, context, 
					ndata, nparam, nconst, single_precission),
				nlsq_gain_ratio(nparam, single_precission)
				})
			);
	}

	export FunctionApplier nlsq_slmh_step(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		const std::shared_ptr<VectorVariable>& params,
		const std::shared_ptr<MatrixVariable>& consts,
		const std::shared_ptr<VectorVariable>& data,
		const std::shared_ptr<SingleVariable>& lambda,
		const std::shared_ptr<SingleVariable>& step_type,
		const std::shared_ptr<SingleVariable>& mu,
		const std::shared_ptr<SingleVariable>& eta,
		const std::shared_ptr<SingleVariable>& acc,
		const std::shared_ptr<SingleVariable>& dec,
		const std::shared_ptr<VectorVariable>& nlstep,
		const std::shared_ptr<SingleVariable>& error,
		const std::shared_ptr<SingleVariable>& new_error,
		const std::shared_ptr<VectorVariable>& residuals,
		const std::shared_ptr<MatrixVariable>& jacobian,
		const std::shared_ptr<MatrixVariable>& hessian,
		const std::shared_ptr<MatrixVariable>& lambda_hessian)
	{
		// type and dimension checks
		{
			if (residuals->getNDim() != jacobian->getNDim1()) {
				throw std::runtime_error("residuals dim and jacobian dim1 must agree");
			}
			if (jacobian->getNDim2() != hessian->getNDim1()) {
				throw std::runtime_error("jacobian dim2 and hessian dim1 must agree");
			}
			if (!hessian->isSquare()) {
				throw std::runtime_error("hessian isn't square");
			}
			if (params->getNDim() != jacobian->getNDim2()) {
				throw std::runtime_error("params dim and jacobian dim2 must agree");
			}
			if (residuals->getNDim() != data->getNDim()) {
				throw std::runtime_error("residuals dim and data dim must agree");
			}
			if (residuals->getNDim() != consts->getNDim1()) {
				throw std::runtime_error("residuals dim and consts dim1 must agree");
			}
			if (lambda_hessian->getNDim1() != hessian->getNDim1()) {
				throw std::runtime_error("hessian and lambda_hessian must have the same dimension");
			}
			if (!lambda_hessian->isSquare()) {
				throw std::runtime_error("lambda_hessian isn't square");
			}
			if (nlstep->getNDim() != params->getNDim()) {
				throw std::runtime_error("step dim and params dim doesn't agree");
			}

			if (!((ui16)residuals->getType() &
				(ui16)jacobian->getType() &
				(ui16)hessian->getType() &
				(ui16)params->getType() &
				(ui16)data->getType() &
				(ui16)consts->getType() &
				(ui16)lambda->getType() &
				(ui16)lambda_hessian->getType() &
				(ui16)nlstep->getType()
				))
			{
				throw std::runtime_error("All inputs must have same type");
			}
			if (!((params->getType() == ShaderVariableType::FLOAT) ||
				(params->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 ndata = residuals->getNDim();
		ui16 nparam = params->getNDim();
		ui16 nconst = consts->getNDim2();

		bool single_precission = true;
		if (residuals->getType() == ShaderVariableType::DOUBLE)
			single_precission = false;

		auto func = nlsq_slmh_step(expr, context, ndata, nparam, nconst, single_precission);

		auto uniqueid = nlsq_slmh_step_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		return FunctionApplier{ func, nullptr,
			{params, consts, data, lambda, step_type, mu, eta,
			acc, dec, nlstep, error, new_error, residuals,
			jacobian, hessian, lambda_hessian }, 
			uniqueid };
	}

}
}

