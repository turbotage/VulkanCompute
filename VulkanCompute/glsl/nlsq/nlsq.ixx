module;

#include <string>
#include <optional>

export module nlsq;

import vc;
import glsl;
import util;

import linalg;
import symm;
import solver;

export import symbolic;

namespace glsl {

	using namespace vc;

	export std::string nlsq_gain_ratio_uniqueid(ui16 nparam, bool single_precission)
	{
		return std::to_string(nparam) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function nlsq_gain_ratio(ui16 nparam, bool single_precission)
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

		return ::glsl::Function(
			"nlsq_gain_ratio_" + uniqueid,
			{ size_t(nparam), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({
				linalg::inner_prod(nparam, single_precission),
				linalg::weighted_vec_norm2(nparam, single_precission)
				})
		);
	}


	export std::string nlsq_error_uniqueid(ui16 ndata, bool single_precission)
	{
		return std::to_string(ndata) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function nlsq_error(ui16 ndata, bool single_precission)
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

		return ::glsl::Function(
			"nlsq_error_" + uniqueid,
			{ size_t(ndata), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({
				linalg::vec_norm2(ndata, single_precission)
				})
		);
	}


	export std::string nlsq_slm_step_uniqueid(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export ::glsl::Function nlsq_slm_step(
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
			util::replace_all(temp, "NRJID", nlsq::nlsq_residuals_jacobian_uniqueid(expr, context, ndata, nparam, nconst, single_precission));
			util::replace_all(temp, "MTMID", linalg::mul_transpose_mat_uniqueid(ndata, nparam, single_precission));
			util::replace_all(temp, "MALOID", linalg::mat_add_ldiag_out_uniqueid(nparam, single_precission));
			util::replace_all(temp, "GID", linalg::gmw81_uniqueid(nparam, single_precission));
			util::replace_all(temp, "MTVID", linalg::mul_transpose_vec_uniqueid(ndata, nparam, single_precission));
			util::replace_all(temp, "VNEGID", linalg::vec_neg_uniqueid(nparam, single_precission));
			util::replace_all(temp, "LSID", linalg::ldl_solve_uniqueid(nparam, single_precission));
			util::replace_all(temp, "AVVID", linalg::add_vec_vec_uniqueid(nparam, single_precission));
			util::replace_all(temp, "NEID", nlsq_error_uniqueid(ndata, single_precission));
			util::replace_all(temp, "NRID", nlsq::nlsq_residuals_uniqueid(expr, context, ndata, nparam, nconst, single_precission));
			util::replace_all(temp, "NGID", nlsq_gain_ratio_uniqueid(nparam, single_precission));
			util::replace_all(temp, "MMVID", linalg::mul_mat_vec_uniqueid(ndata, nparam, single_precission));
			util::replace_all(temp, "VNORMID", linalg::vec_norm_uniqueid(ndata, single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"nlsq_slm_step_" + uniqueid,
			{ hashed_expr, size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({ 
				nlsq::nlsq_residuals_jacobian(expr, context, ndata, nparam, nconst, single_precission),
				linalg::mul_transpose_mat(ndata, nparam, single_precission),
				linalg::mat_add_ldiag_out(nparam, single_precission),
				linalg::gmw81(nparam, single_precission),
				linalg::mul_transpose_vec(ndata, nparam, single_precission),
				linalg::vec_neg(nparam, single_precission),
				linalg::ldl_solve(nparam, single_precission),
				linalg::add_vec_vec(nparam, single_precission),
				nlsq_error(ndata, single_precission),
				nlsq::nlsq_residuals(expr, context, ndata, nparam, nconst, single_precission),
				nlsq_gain_ratio(nparam, single_precission),
				linalg::mul_mat_vec(ndata, nparam, single_precission),
				linalg::vec_norm(ndata, single_precission)
				})
		);
	}

}