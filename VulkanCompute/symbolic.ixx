module;

#include <tuple>
#include <optional>
#include <symengine/expression.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>

export module symbolic;

import util;
import glsl;
import linalg;
import expr;

namespace glsl {
namespace symbolic {
namespace nlsq {

	export ::glsl::Function nlsq_residual(const expression::Expression& expr, int ndata, int nparam, int nconst, bool single_precission = true);

	export ::glsl::Function nlsq_residual_jacobian_hessian(std::string expression_name, std::string expression, int ndata, int nparam, int nconst, bool single_precission = true);

}
}
}

namespace glsl {
namespace symbolic {
namespace nlsq {
	


	std::tuple<std::string, std::vector<std::string>, std::vector<std::string>> expr_diff_diff2(std::string expression, int nparam) 
	{
		auto expr = SymEngine::parse(expression);


		std::vector <SymEngine::RCP <const SymEngine::Basic>> diffs_sym(nparam);
		std::vector<std::string> diffs(nparam);
		for (int i = 0; i < nparam; ++i) {
			 auto sym = SymEngine::symbol("x" + std::to_string(i));
			 diffs_sym[i] = expr->diff(sym);
			 diffs[i] = sym_to_str(diffs_sym[i]);
		}

		std::vector<std::string> diff2s(nparam*(nparam + 1)/2);
		
		int k = 0;
		for (int i = 0; i < nparam; ++i) {
			auto diff = diffs_sym[i];
			for (int j = 0; j <= i; ++j) {
				auto sym = SymEngine::symbol("x" + std::to_string(j));
				diff2s[k] = sym_to_str(diff->diff(sym));
				++k;
			}
		}

		return std::make_tuple(sym_to_str(expr), diffs, diff2s);
	}

	// IMPL

	::glsl::Function nlsq_residual(std::string expression_name, std::string expression, int ndata, int nparam, int nconst, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void NAME_nlsq_residual(in float params[nparam], in float consts[ndata*nconst], in float data[ndata], out float residuals[ndata]) {
	for (int i = 0; i < ndata; ++i) {
RESIDUAL_EXPRESSION
	}
}
)glsl";

		auto rexpr = change_params_to_array_params(expression, nparam, nconst);
		std::string resexpr = "\t\tresiduals[i] = " + rexpr + " - data[i];";

		std::function<std::string()> code_func = 
			[ndata, nparam, nconst, expression_name, resexpr, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "NAME_nlsq_residual", expression_name + "_nlsq_residual");
			util::replace_all(temp, "RESIDUAL_EXPRESSION", resexpr);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "nparam", std::to_string(nparam));
			util::replace_all(temp, "nconst", std::to_string(nconst));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		std::hash<std::string> hasher;

		return ::glsl::Function(
			expression_name + "_nlsq_residual",
			{ hasher(expression), size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function nlsq_residual_jacobian_hessian(std::string expression_name, std::string expression, int ndata, int nparam, int nconst, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void NAME_nlsq_residual_jacobian_hessian(
	in float params[nparam],
	in float consts[ndata*nconst],
	in float data[ndata],
	out float residuals[ndata],
	out float jacobian[ndata*nparam], 
	out float hessian[nparam*nparam]) {
	
	// eval
	for (int i = 0; i < ndata; ++i) {
RESIDUAL_EXPRESSIONS
	}

	// jacobian
	for (int i = 0; i < ndata; ++i) {
JACOBIAN_EXPRESSIONS
	}	
	
	// zero hessian
	for (int i = 0; i < nparam; ++i) {
		for (int j = 0; j < nparam; ++j) {
			hessian[i*nparam + j] = 0;
		}
	}

	// second order part of hessian
	for (int i = 0; i < ndata; ++i) {
HESSIAN_EXPRESSIONS
	}

	// copy to upper part
	for (int i = 1; i < nparam; ++i) {
		for (int j = 0; j < i; ++j) {
			hessian[j*nparam + i] = hessian[i*nparam + j];
		}
	}

	// add first order part of hessian
	mul_transpose_mat_add(jacobian, hessian);
	
}
)glsl";

		std::string expr;
		std::vector<std::string> diffs;
		std::vector<std::string> diff2s;

		std::tie(expr, diffs, diff2s) = expr_diff_diff2(expression, nparam);

		// residuals
		auto rexpr = change_params_to_array_params(expression, nparam, nconst);
		std::string resexpr = "\t\tresiduals[i] = " + rexpr + " - data[i];";

		// jacobian
		std::string jacexpr = "";
		for (int i = 0; i < nparam; ++i) {
			auto jexpr = change_params_to_array_params(diffs[i], nparam, nconst);
			jacexpr += "\t\tjacobian[i*" + std::to_string(nparam) + "+" + std::to_string(i) + 
				"] = " + jexpr + ";\n";
		}

		// hessian
		std::string hesexpr = "";
		int k = 0;
		for (int i = 0; i < nparam; ++i) {
			for (int j = 0; j <= i; ++j) {
				auto hexpr = change_params_to_array_params(diff2s[k], nparam, nconst);
				hesexpr += "\t\thessian[" + std::to_string(i) + "*" + std::to_string(nparam) + "+" + 
					std::to_string(j) + "] += residuals[i] * " + hexpr + ";\n";
				++k;
			}
		}


		std::function<std::string()> code_func =
			[ndata, nparam, nconst, expression_name, resexpr, jacexpr, hesexpr, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "NAME_nlsq_residual_jacobian_hessian", expression_name + "_nlsq_residual_jacobian_hessian");
			util::replace_all(temp, "RESIDUAL_EXPRESSIONS", resexpr);
			util::replace_all(temp, "JACOBIAN_EXPRESSIONS", jacexpr);
			util::replace_all(temp, "HESSIAN_EXPRESSIONS", hesexpr);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "nparam", std::to_string(nparam));
			util::replace_all(temp, "nconst", std::to_string(nconst));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		std::hash<std::string> hasher;
		
		return ::glsl::Function(
			expression_name + "_nlsq_residual_jacobian_hessian",
			{ hasher(expression), size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission)},
			code_func,
			std::make_optional<std::vector<Function>>({
				linalg::mul_transpose_mat_add(ndata, nparam, single_precission),
			})
		);

	}

}
}
}
