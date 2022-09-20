module;

#include <tuple>
#include <optional>
#include <string>
#include <functional>
#include <vector>


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

	export ::glsl::Function nlsq_residual(const std::string& id,
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		int ndata, int nparam, int nconst, bool single_precission = true);

	export ::glsl::Function nlsq_residual_jacobian_hessian(const std::string& id,
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		int ndata, int nparam, int nconst, bool single_precission = true);

}
}
}

namespace glsl {
namespace symbolic {
namespace nlsq {

	// IMPL

	::glsl::Function nlsq_residual(const std::string& id,
		const expression::Expression& expr, const glsl::SymbolicContext& context, 
		int ndata, int nparam, int nconst, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void NAME_nlsq_residual(in float params[nparam], in float consts[ndata*nconst], in float data[ndata], out float residuals[ndata]) {
	for (int i = 0; i < ndata; ++i) {
RESIDUAL_EXPRESSION
	}
}
)glsl";

		std::string resexpr = "\t\tresiduals[i] = " + expr.glsl_str(context) + " - data[i];";

		std::function<std::string()> code_func = 
			[ndata, nparam, nconst, id, resexpr, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "NAME_nlsq_residual", id + "_nlsq_residual");
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
			id + "_nlsq_residual",
			{ hasher(id), size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function nlsq_residual_jacobian_hessian(const std::string& id,
		const expression::Expression& expr, const glsl::SymbolicContext& context, 
		int ndata, int nparam, int nconst, bool single_precission)
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

		// residuals
		std::string resexpr = "\t\tresiduals[i] = " + expr.glsl_str(context) + " - data[i];";

		// jacobian
		std::string jacexpr = "";
		for (int i = 0; i < nparam; ++i) {
			auto diff_symbol = context.get_params_name(i);
			auto jexpr = expr.diff(diff_symbol)->glsl_str(context);
			jacexpr += "\t\tjacobian[i*" + std::to_string(nparam) + "+" + std::to_string(i) + 
				"] = " + jexpr + ";\n";
		}

		// hessian
		std::string hesexpr = "";
		int k = 0;
		for (int i = 0; i < nparam; ++i) {
			auto diff_symbol1 = context.get_params_name(i);
			auto diff1 = expr.diff(diff_symbol1);
			for (int j = 0; j <= i; ++j) {
				auto diff_symbol2 = context.get_params_name(j);

				auto hexpr = diff1->diff(diff_symbol2)->glsl_str(context);
				hesexpr += "\t\thessian[" + std::to_string(i) + "*" + std::to_string(nparam) + "+" + 
					std::to_string(j) + "] += residuals[i] * " + hexpr + ";\n";
				++k;
			}
		}


		std::function<std::string()> code_func =
			[ndata, nparam, nconst, id, resexpr, jacexpr, hesexpr, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "NAME_nlsq_residual_jacobian_hessian", id + "_nlsq_residual_jacobian_hessian");
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
			id + "_nlsq_residual_jacobian_hessian",
			{ hasher(id), size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission)},
			code_func,
			std::make_optional<std::vector<Function>>({
				linalg::mul_transpose_mat_add(ndata, nparam, single_precission),
			})
		);

	}

}
}
}
