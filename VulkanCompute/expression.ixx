module;

#include <optional>
#include <symengine/expression.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>

export module expression;

import util;
import glsl;
import linalg;

namespace glsl {
namespace expression {

	export ::glsl::Function residual(std::string expression_name, std::string expression, int ndata, int nparam, int nconst, bool single_precission = true);

	export ::glsl::Function residual_hessian_jacobian(std::string expression_name, std::string expression, int ndata, int nparam, int nconst, bool single_precission = true);

}
}

namespace glsl {
namespace expression {


	std::string change_params_to_array_params(const std::string& expression, int nparam, int nconst) {
		std::string temp_expression = expression;
		// make param replacements
		for (int i = 0; i < nparam; ++i) {
			auto istr = std::to_string(i);
			int nreplacements = util::replace_all(temp_expression, "x" + istr, "x[" + istr + "]");
		}

		for (int i = 0; i < nconst; ++i) {
			auto istr = std::to_string(i);
			int nreplacements = util::replace_all(temp_expression, "y" + istr, "y[i*" + std::to_string(nconst) + " + " + istr + "]");
		}

		return temp_expression;
	}

	::glsl::Function residual(std::string expression_name, std::string expression, int ndata, int nparam, int nconst, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void NAME_residual(
	in float param[ncol],
	in float consts[nrow*nconsts],
	in float data[nrow],
	out float residuals[nrow],
) {
	
	// eval
	for (int i = 0; i < nrow; ++i) {
RESIDUAL_EXPRESSIONS
	}

}
)glsl";

		auto rexpr = change_params_to_array_params(expression, nparam, nconst);
		std::string resexpr = "\t\tresiduals[i] = " + rexpr + " - data[i]";

		std::function<std::string()> code_func = [resexpr, single_precission]() -> std::string
		{
			std::string temp = code;

			util::replace_all(temp, "RESIDUAL_EXPRESSIONS", resexpr);
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		std::hash<std::string> hasher;

		return ::glsl::Function(
			expression_name + "_residual",
			{ hasher(expression), size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function residual_hessian_jacobian(std::string expression_name, std::string expression, int ndata, int nparam, int nconst, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void NAME_residual_hessian_jacobian(
	in float param[ncol],
	in float consts[nrow*nconsts],
	in float data[nrow],
	out float residuals[nrow],
	out float jacobian[nrow*ncol], 
	out float hessian[ncol*ncol]
) {
	
	// eval
	for (int i = 0; i < nrow; ++i) {
RESIDUAL_EXPRESSIONS
	}
	

	// jacobian
	for (int i = 0; i < nrow; ++i) {
JACOBIAN_EXPRESSIONS
	}	
	
	// second order part of hessian
SECOND_ORDER_HESSIAN

	// add first order part of hessian
	mul_transpose_mat_add(jacobian, hessian);
	

}
)glsl";
		
		auto residual_expr = change_params_to_array_params(expression, nparam, nconst);
		

		std::function<std::string()> code_func = [expression, single_precission]() -> std::string
		{
			std::string temp = code;


			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		std::hash<std::string> hasher;

		return ::glsl::Function(
			expression_name + "_eval_hessian_jacobian",
			{ hasher(expression), size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission)},
			code_func,
			std::make_optional<std::vector<Function>>({
				linalg::mul_transpose_mat(ndata, nparam, single_precission),
			})
		);

	}

}
}
