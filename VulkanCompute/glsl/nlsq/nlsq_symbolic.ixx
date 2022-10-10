module;

#include <symengine/expression.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>

export module nlsq_symbolic;

import <tuple>;
import <optional>;
import <string>;
import <functional>;
import <vector>;
import <memory>;

import vc;
import util;
import linalg;
import symbolic;
export import expr;
export import glsl;
export import variable;
export import function;

namespace glsl {
namespace nlsq {

	using namespace vc;

	using vecptrfunc = std::vector<std::shared_ptr<Function>>;
	using refvecptrfunc = refw<std::vector<std::shared_ptr<Function>>>;

	// residuals
	export std::string nlsq_residuals_uniqueid(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export std::shared_ptr<::glsl::Function> nlsq_residuals(
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

		std::string resexpr = "\t\tresiduals[i] = " + expr.glsl_str(context) + " - data[i];\n";

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

		return std::make_shared<Function>(
			"nlsq_residuals_" + uniqueid,
			std::vector<size_t>{ hashed_expr, size_t(ndata), 
			size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::FunctionApplier nlsq_residuals(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		const std::shared_ptr<glsl::VectorVariable>& params,
		const std::shared_ptr<glsl::MatrixVariable>& consts,
		const std::shared_ptr<glsl::VectorVariable>& data,
		const std::shared_ptr<glsl::VectorVariable>& residuals)
	{
		// type and dimension checks
		{
			if (residuals->getNDim() != data->getNDim()) {
				throw std::runtime_error("residuals dim and data dim must agree");
			}
			if (residuals->getNDim() != consts->getNDim1()) {
				throw std::runtime_error("residuals dim and consts dim1 must agree");
			}

			if (!((ui16)residuals->getType() &
				(ui16)params->getType() &
				(ui16)data->getType() &
				(ui16)consts->getType()))
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

		auto func = nlsq_residuals(expr, context, ndata, nparam, nconst, single_precission);

		auto uniqueid = nlsq_residuals_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		return FunctionApplier{ func, nullptr,
			{params, consts, data, residuals }, uniqueid };

	}


	// residuals and jacobian
	export std::string nlsq_residuals_jacobian_uniqueid(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export std::shared_ptr<::glsl::Function> nlsq_residuals_jacobian(
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
		std::string resexpr = "\t\tresiduals[i] = " + expr.glsl_str(context) + " - data[i];\n";

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

		return std::make_shared<Function>(
			"nlsq_residuals_jacobian_" + uniqueid,
			std::vector<size_t>{ hashed_expr, size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::FunctionApplier nlsq_residuals_jacobian(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		const std::shared_ptr<glsl::VectorVariable>& params,
		const std::shared_ptr<glsl::MatrixVariable>& consts,
		const std::shared_ptr<glsl::VectorVariable>& data,
		const std::shared_ptr<glsl::VectorVariable>& residuals,
		const std::shared_ptr<glsl::MatrixVariable>& jacobian)
	{
		// type and dimension checks
		{
			if (residuals->getNDim() == jacobian->getNDim1()) {
				throw std::runtime_error("residuals dim and jacobian dim1 must agree");
			}
			if (params->getNDim() == jacobian->getNDim2()) {
				throw std::runtime_error("params dim and jacobian dim2 must agree");
			}
			if (residuals->getNDim() == data->getNDim()) {
				throw std::runtime_error("residuals dim and data dim must agree");
			}
			if (residuals->getNDim() == consts->getNDim1()) {
				throw std::runtime_error("residuals dim and consts dim1 must agree");
			}

			if ((ui16)residuals->getType() &
				(ui16)jacobian->getType() &
				(ui16)params->getType() &
				(ui16)data->getType() &
				(ui16)consts->getType())
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

		auto func = nlsq_residuals_jacobian(expr, context, ndata, nparam, nconst, single_precission);

		auto uniqueid = nlsq_residuals_jacobian_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		return FunctionApplier{ func, nullptr,
			{params, consts, data, residuals, jacobian}, uniqueid };

	}


	// residuals, jacobian and hessian
	export std::string nlsq_residuals_jacobian_hessian_uniqueid(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export std::shared_ptr<::glsl::Function> nlsq_residuals_jacobian_hessian(
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
		std::string resexpr = "\t\tresiduals[i] = " + expr.glsl_str(context) + " - data[i];\n";

		// jacobian
		std::string jacexpr = "";
		for (int i = 0; i < nparam; ++i) {
			std::string diff_symbol = context.get_params_name(i);
			auto dexpr = expr.diff(diff_symbol);
			bool is_zero = dexpr->is_zero();
			if (is_zero)
				continue;
			std::string jexpr = dexpr->glsl_str(context);
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
				auto dhexpr = diff1->diff(diff_symbol2);
				bool is_zero = dhexpr->is_zero();
				if (is_zero)
					continue;
				std::string hexpr = dhexpr->glsl_str(context);
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
		
		return std::make_shared<Function>(
			"nlsq_residuals_jacobian_hessian_" + uniqueid,
			std::vector<size_t>{ hashed_expr, size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission)},
			code_func,
			std::make_optional<vecptrfunc>({
				linalg::mul_transpose_mat_add(ndata, nparam, single_precission),
			})
		);

	}

	export ::glsl::FunctionApplier nlsq_residuals_jacobian_hessian(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		const std::shared_ptr<glsl::VectorVariable>& params,
		const std::shared_ptr<glsl::MatrixVariable>& consts,
		const std::shared_ptr<glsl::VectorVariable>& data,
		const std::shared_ptr<glsl::VectorVariable>& residuals,
		const std::shared_ptr<glsl::MatrixVariable>& jacobian,
		const std::shared_ptr<glsl::MatrixVariable>& hessian)
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
				throw std::runtime_error("hessian must be square square");
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

			if (!((ui16)residuals->getType() &
				(ui16)jacobian->getType() &
				(ui16)hessian->getType() &
				(ui16)params->getType() &
				(ui16)data->getType() &
				(ui16)consts->getType()))
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

		auto func = nlsq_residuals_jacobian_hessian(expr, context, ndata, nparam, nconst, single_precission);

		auto uniqueid = nlsq_residuals_jacobian_hessian_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		return FunctionApplier{ func, nullptr,
			{params, consts, data, residuals, jacobian, hessian}, uniqueid };

	}


	// residuals, jacobian, hessian and lambda-mult
	export std::string nlsq_residuals_jacobian_hessian_l_uniqueid(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export std::shared_ptr<::glsl::Function> nlsq_residuals_jacobian_hessian_l(
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
	out float hessian[nparam*nparam],
	out float lambda_hessian[nparam*nparam]) {
	
	mat_set_zero_MSZ(hessian);

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

	// store J^T @ J inside lambda_hessian, second order terms are in hessian
	mul_transpose_mat_MTMID(jacobian, lambda_hessian);
	add_mat_mat_ldiag_AMML(hessian, lambda, lambda_hessian);
}
)glsl";

		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());

		// residuals
		std::string resexpr = "\t\tresiduals[i] = " + expr.glsl_str(context) + " - data[i];\n";

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
			util::replace_all(temp, "MSZ", linalg::mat_set_zero_uniqueid(nparam, nparam, single_precission));
			util::replace_all(temp, "MTMID", linalg::mul_transpose_mat_uniqueid(ndata, nparam, single_precission));
			util::replace_all(temp, "AMML", linalg::add_mat_mat_ldiag_uniqueid(nparam, single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"nlsq_residuals_jacobian_hessian_l_" + uniqueid,
			std::vector<size_t>{ hashed_expr, size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::make_optional<vecptrfunc>({
				linalg::mat_set_zero(nparam, nparam, single_precission),
				linalg::mul_transpose_mat(ndata, nparam, single_precission),
				linalg::add_mat_mat_ldiag(nparam, single_precission)
				})
			);

	}

	export ::glsl::FunctionApplier nlsq_residuals_jacobian_hessian_l(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		const std::shared_ptr<glsl::VectorVariable>& params,
		const std::shared_ptr<glsl::MatrixVariable>& consts,
		const std::shared_ptr<glsl::VectorVariable>& data,
		const std::shared_ptr<glsl::SingleVariable>& lambda,
		const std::shared_ptr<glsl::VectorVariable>& residuals,
		const std::shared_ptr<glsl::MatrixVariable>& jacobian,
		const std::shared_ptr<glsl::MatrixVariable>& hessian,
		const std::shared_ptr<glsl::MatrixVariable>& lambda_hessian)
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

			if (!((ui16)residuals->getType() &
				(ui16)jacobian->getType() &
				(ui16)hessian->getType() &
				(ui16)params->getType() &
				(ui16)data->getType() &
				(ui16)consts->getType() &
				(ui16)lambda->getType() &
				(ui16)lambda_hessian->getType()))
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

		auto func = nlsq_residuals_jacobian_hessian_l(expr, context, ndata, nparam, nconst, single_precission);

		auto uniqueid = nlsq_residuals_jacobian_hessian_l_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		return FunctionApplier{ func, nullptr,
			{params, consts, data, lambda, residuals, jacobian, hessian, lambda_hessian }, uniqueid };

	}


	// residuals, jacobian, hessian and scaled lambda-mult
	export std::string nlsq_residuals_jacobian_hessian_sl_uniqueid(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		ui16 ndata, ui16 nparam, ui16 nconst, bool single_precission)
	{
		size_t hashed_expr = std::hash<std::string>()(expr.get_expression());
		return std::to_string(ndata) + "_" + std::to_string(nparam) + "_" + std::to_string(nconst) + "_" + util::stupid_compress(hashed_expr);
	}

	export std::shared_ptr<::glsl::Function> nlsq_residuals_jacobian_hessian_sl(
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
		std::string resexpr = "\t\tresiduals[i] = " + expr.glsl_str(context) + " - data[i];\n";

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

		return std::make_shared<Function>(
			"nlsq_residuals_jacobian_hessian_sl_" + uniqueid,
			std::vector<size_t>{ hashed_expr, size_t(ndata), size_t(nparam), size_t(nconst), size_t(single_precission) },
			code_func,
			std::make_optional<vecptrfunc>({
				linalg::mul_transpose_mat_add_ldiag(ndata, nparam, single_precission),
				})
				);

	}

	export ::glsl::FunctionApplier nlsq_residuals_jacobian_hessian_sl(
		const expression::Expression& expr, const glsl::SymbolicContext& context,
		const std::shared_ptr<glsl::VectorVariable>& params,
		const std::shared_ptr<glsl::MatrixVariable>& consts,
		const std::shared_ptr<glsl::VectorVariable>& data,
		const std::shared_ptr<glsl::VectorVariable>& residuals,
		const std::shared_ptr<glsl::MatrixVariable>& jacobian,
		const std::shared_ptr<glsl::MatrixVariable>& hessian)
	{
		// type and dimension checks
		{
			if (residuals->getNDim() == jacobian->getNDim1()) {
				throw std::runtime_error("residuals dim and jacobian dim1 must agree");
			}
			if (jacobian->getNDim2() == hessian->getNDim1()) {
				throw std::runtime_error("jacobian dim2 and hessian dim1 must agree");
			}
			if (hessian->isSquare()) {
				throw std::runtime_error("hessian is square");
			}
			if (params->getNDim() == jacobian->getNDim2()) {
				throw std::runtime_error("params dim and jacobian dim2 must agree");
			}
			if (residuals->getNDim() == data->getNDim()) {
				throw std::runtime_error("residuals dim and data dim must agree");
			}
			if (residuals->getNDim() == consts->getNDim1()) {
				throw std::runtime_error("residuals dim and consts dim1 must agree");
			}

			if ((ui16)residuals->getType() &
				(ui16)jacobian->getType() &
				(ui16)hessian->getType() &
				(ui16)params->getType() &
				(ui16)data->getType() &
				(ui16)consts->getType())
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

		auto func = nlsq_residuals_jacobian_hessian_sl(expr, context, ndata, nparam, nconst, single_precission);

		auto uniqueid = nlsq_residuals_jacobian_hessian_sl_uniqueid(expr, context, ndata, nparam, nconst, single_precission);

		return FunctionApplier{ func, nullptr,
			{params, consts, data, residuals, jacobian, hessian}, uniqueid };

	}


}
}
