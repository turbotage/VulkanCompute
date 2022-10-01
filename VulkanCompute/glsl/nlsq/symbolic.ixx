module;

#include <symengine/expression.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>

export module symbolic;

import <tuple>;
import <optional>;
import <string>;
import <functional>;
import <vector>;

import vc;
import util;
import linalg;
export import expr;
export import glsl;

namespace glsl {
namespace nlsq {

	using namespace vc;

	// residuals
	export std::string nlsq_residuals_uniqueid(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export ::glsl::Function nlsq_residuals(
		const expression::Expression& expr, const glsl::SymbolicContext& context, 
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void nlsq_residuals_UNIQUEID(in float params[nparam], in float consts[ndata*nconst], in float data[ndata], out float residuals[ndata]) {
	for (int i = 0; i < ndata; ++i) {
RESIDUAL_EXPRESSION
	}
}
)glsl";

		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());

		std::string uniqueid = nlsq_residuals_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		std::string resexpr = "\t\tresiduals[i] = " + expr.glsl_str(context) + " - data[i];";

		std::function<std::string()> code_func = 
			[ndata, nparam, nconst, resexpr, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "RESIDUAL_EXPRESSION", resexpr);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "nparam", std::to_string(nparam));
			util::replace_all(temp, "nconst", std::to_string(nconst));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"nlsq_residuals_" + uniqueid,
			{ hashed_expr, size_t(ndata), 
			size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	// residuals and jacobian
	export std::string nlsq_residuals_jacobian_uniqueid(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export ::glsl::Function nlsq_residuals_jacobian(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void nlsq_residuals_jacobian_UNIQUEID(
	in float params[nparam],
	in float consts[ndata*nconst],
	in float data[ndata],
	out float residuals[ndata],
	out float jacobian[ndata*nparam]) {
	
	for (int i = 0; i < ndata; ++i) {
		// eval
RESIDUAL_EXPRESSIONS
		// jacobian
JACOBIAN_EXPRESSIONS
	}	

}
)glsl";

		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());

		// residuals
		std::string resexpr = "\t\tresiduals[i] = " + expr.glsl_str(context) + " - data[i];";

		// jacobian
		std::string jacexpr = "";
		for (int i = 0; i < nparam; ++i) {
			std::string diff_symbol = context.get_params_name(i);
			std::string jexpr = expr.diff(diff_symbol)->glsl_str(context);
			std::string partj = "\t\tjacobian[i*" + std::to_string(nparam) + "+" + std::to_string(i) +
				"] = " + jexpr + ";\n";
			jacexpr += partj;
		}
	
		std::string uniqueid = nlsq_residuals_jacobian_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		std::function<std::string()> code_func =
			[ndata, nparam, nconst, resexpr, jacexpr, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "RESIDUAL_EXPRESSIONS", resexpr);
			util::replace_all(temp, "JACOBIAN_EXPRESSIONS", jacexpr);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "nparam", std::to_string(nparam));
			util::replace_all(temp, "nconst", std::to_string(nconst));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"nlsq_residuals_jacobian_" + uniqueid,
			{ hashed_expr, size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	// residuals, jacobian and hessian
	export std::string nlsq_residuals_jacobian_hessian_uniqueid(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export ::glsl::Function nlsq_residuals_jacobian_hessian(
		const expression::Expression& expr, const glsl::SymbolicContext& context, 
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void nlsq_residuals_jacobian_hessian_UNIQUEID(
	in float params[nparam],
	in float consts[ndata*nconst],
	in float data[ndata],
	out float residuals[ndata],
	out float jacobian[ndata*nparam], 
	out float hessian[nparam*nparam]) {
	
	for (int i = 0; i < ndata; ++i) {
		// eval
RESIDUAL_EXPRESSIONS
		// jacobian
JACOBIAN_EXPRESSIONS
		// second order part of hessian
HESSIAN_EXPRESSIONS
	}

	// copy to upper part
	for (int i = 1; i < nparam; ++i) {
		for (int j = 0; j < i; ++j) {
			hessian[j*nparam + i] = hessian[i*nparam + j];
		}
	}

	// add first order part of hessian
	mul_transpose_mat_add_MTMAID(jacobian, hessian);

}
)glsl";

		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());

		// residuals
		std::string resexpr = "\t\tresiduals[i] = " + expr.glsl_str(context) + " - data[i];";

		// jacobian
		std::string jacexpr = "";
		for (int i = 0; i < nparam; ++i) {
			std::string diff_symbol = context.get_params_name(i);
			std::string jexpr = expr.diff(diff_symbol)->glsl_str(context);
			std::string partj = "\t\tjacobian[i*" + std::to_string(nparam) + "+" + std::to_string(i) + 
				"] = " + jexpr + ";\n";
			jacexpr += partj;
		}

		// hessian
		std::string hesexpr = "";
		int k = 0;
		for (int i = 0; i < nparam; ++i) {
			auto diff_symbol1 = context.get_params_name(i);
			auto diff1 = expr.diff(diff_symbol1);
			for (int j = 0; j <= i; ++j) {
				std::string diff_symbol2 = context.get_params_name(j);

				std::string hexpr = diff1->diff(diff_symbol2)->glsl_str(context);
				std::string parth = "\t\thessian[" + std::to_string(i) + "*" + std::to_string(nparam) + "+" + 
					std::to_string(j) + "] += residuals[i] * " + hexpr + ";\n";
				hesexpr += parth;

				++k;
			}
		}

		std::string uniqueid = nlsq_residuals_jacobian_hessian_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		std::function<std::string()> code_func =
			[ndata, nparam, nconst, resexpr, jacexpr, hesexpr, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "RESIDUAL_EXPRESSIONS", resexpr);
			util::replace_all(temp, "JACOBIAN_EXPRESSIONS", jacexpr);
			util::replace_all(temp, "HESSIAN_EXPRESSIONS", hesexpr);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "nparam", std::to_string(nparam));
			util::replace_all(temp, "nconst", std::to_string(nconst));
			util::replace_all(temp, "MTMAID", linalg::mul_transpose_mat_add_uniqueid(ndata, nparam, single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};
		
		return ::glsl::Function(
			"nlsq_residuals_jacobian_hessian_" + uniqueid,
			{ hashed_expr, size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission)},
			code_func,
			std::make_optional<std::vector<Function>>({
				linalg::mul_transpose_mat_add(ndata, nparam, single_precission),
			})
		);

	}

	// residuals, jacobian, hessian and lambda-mult
	export std::string nlsq_residuals_jacobian_hessian_l_uniqueid(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export ::glsl::Function nlsq_residuals_jacobian_hessian_l(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void nlsq_residuals_jacobian_hessian_l_UNIQUEID(
	in float params[nparam],
	in float consts[ndata*nconst],
	in float data[ndata],
	float lambda,
	out float residuals[ndata],
	out float jacobian[ndata*nparam], 
	out float hessian[nparam*nparam]) {
	
	for (int i = 0; i < ndata; ++i) {
		// eval
RESIDUAL_EXPRESSIONS
		// jacobian
JACOBIAN_EXPRESSIONS
		// second order part of hessian
HESSIAN_EXPRESSIONS
	}

	// copy to upper part
	for (int i = 1; i < nparam; ++i) {
		for (int j = 0; j < i; ++j) {
			hessian[j*nparam + i] = hessian[i*nparam + j];
		}
	}

	// add first order part of hessian and add lambda diagonal
	mul_transpose_mat_add_MTMAID(jacobian, hessian);
	mat_add_ldiag_MALID(hessian, lambda);

}
)glsl";

		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());

		// residuals
		std::string resexpr = "\t\tresiduals[i] = " + expr.glsl_str(context) + " - data[i];";

		// jacobian
		std::string jacexpr = "";
		for (int i = 0; i < nparam; ++i) {
			std::string diff_symbol = context.get_params_name(i);
			std::string jexpr = expr.diff(diff_symbol)->glsl_str(context);
			std::string partj = "\t\tjacobian[i*" + std::to_string(nparam) + "+" + std::to_string(i) +
				"] = " + jexpr + ";\n";
			jacexpr += partj;
		}

		// hessian
		std::string hesexpr = "";
		int k = 0;
		for (int i = 0; i < nparam; ++i) {
			auto diff_symbol1 = context.get_params_name(i);
			auto diff1 = expr.diff(diff_symbol1);
			for (int j = 0; j <= i; ++j) {
				std::string diff_symbol2 = context.get_params_name(j);

				std::string hexpr = diff1->diff(diff_symbol2)->glsl_str(context);
				std::string parth = "\t\thessian[" + std::to_string(i) + "*" + std::to_string(nparam) + "+" +
					std::to_string(j) + "] += residuals[i] * " + hexpr + ";\n";
				hesexpr += parth;

				++k;
			}
		}

		std::string uniqueid = nlsq_residuals_jacobian_hessian_l_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		std::function<std::string()> code_func =
			[ndata, nparam, nconst, resexpr, jacexpr, hesexpr, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "RESIDUAL_EXPRESSIONS", resexpr);
			util::replace_all(temp, "JACOBIAN_EXPRESSIONS", jacexpr);
			util::replace_all(temp, "HESSIAN_EXPRESSIONS", hesexpr);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "nparam", std::to_string(nparam));
			util::replace_all(temp, "nconst", std::to_string(nconst));
			util::replace_all(temp, "MTMAID", linalg::mul_transpose_mat_add_uniqueid(ndata, nparam, single_precission));
			util::replace_all(temp, "MALID", linalg::mat_add_ldiag_uniqueid(nparam, single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"nlsq_residuals_jacobian_hessian_l_" + uniqueid,
			{ hashed_expr, size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({
				linalg::mul_transpose_mat_add(ndata, nparam, single_precission),
				linalg::mat_add_ldiag(nparam, single_precission)
				})
				);

	}

	// residuals, jacobian, hessian and scaled lambda-mult
	export std::string nlsq_residuals_jacobian_hessian_sl_uniqueid(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export ::glsl::Function nlsq_residuals_jacobian_hessian_sl(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void nlsq_residuals_jacobian_hessian_sl_UNIQUEID(
	in float params[nparam],
	in float consts[ndata*nconst],
	in float data[ndata],
	float lambda,
	out float residuals[ndata],
	out float jacobian[ndata*nparam], 
	out float hessian[nparam*nparam]) {
	
	for (int i = 0; i < ndata; ++i) {
		// eval
RESIDUAL_EXPRESSIONS
		// jacobian
JACOBIAN_EXPRESSIONS
		// second order part of hessian
HESSIAN_EXPRESSIONS
	}

	// copy to upper part
	for (int i = 1; i < nparam; ++i) {
		for (int j = 0; j < i; ++j) {
			hessian[j*nparam + i] = hessian[i*nparam + j];
		}
	}

	// add first order part of hessian and add lambda diagonal
	mul_transpose_mat_add_ldiag_MTMAID(jacobian, hessian);

}
)glsl";

		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());

		// residuals
		std::string resexpr = "\t\tresiduals[i] = " + expr.glsl_str(context) + " - data[i];";

		// jacobian
		std::string jacexpr = "";
		for (int i = 0; i < nparam; ++i) {
			std::string diff_symbol = context.get_params_name(i);
			std::string jexpr = expr.diff(diff_symbol)->glsl_str(context);
			std::string partj = "\t\tjacobian[i*" + std::to_string(nparam) + "+" + std::to_string(i) +
				"] = " + jexpr + ";\n";
			jacexpr += partj;
		}

		// hessian
		std::string hesexpr = "";
		int k = 0;
		for (int i = 0; i < nparam; ++i) {
			auto diff_symbol1 = context.get_params_name(i);
			auto diff1 = expr.diff(diff_symbol1);
			for (int j = 0; j <= i; ++j) {
				std::string diff_symbol2 = context.get_params_name(j);

				std::string hexpr = diff1->diff(diff_symbol2)->glsl_str(context);
				std::string parth = "\t\thessian[" + std::to_string(i) + "*" + std::to_string(nparam) + "+" +
					std::to_string(j) + "] += residuals[i] * " + hexpr + ";\n";
				hesexpr += parth;

				++k;
			}
		}

		std::string uniqueid = nlsq_residuals_jacobian_hessian_sl_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		std::function<std::string()> code_func =
			[ndata, nparam, nconst, resexpr, jacexpr, hesexpr, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "RESIDUAL_EXPRESSIONS", resexpr);
			util::replace_all(temp, "JACOBIAN_EXPRESSIONS", jacexpr);
			util::replace_all(temp, "HESSIAN_EXPRESSIONS", hesexpr);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "nparam", std::to_string(nparam));
			util::replace_all(temp, "nconst", std::to_string(nconst));
			util::replace_all(temp, "MTMAID", linalg::mul_transpose_mat_add_ldiag_uniqueid(ndata, nparam, single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"nlsq_residuals_jacobian_hessian_sl_" + uniqueid,
			{ hashed_expr, size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({
				linalg::mul_transpose_mat_add_ldiag(ndata, nparam, single_precission),
				})
				);

	}


}
}