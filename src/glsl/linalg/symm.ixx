module;

export module symm;

import <vector>;
import <string>;
import <optional>;
import <memory>;
import <functional>;
import <stdexcept>;

import vc;
import util;
import solver;
export import linalg;
import glsl;

import variable;
import function;


using namespace vc;

namespace glsl {
namespace linalg {

	using vecptrfunc = std::vector<std::shared_ptr<Function>>;
	using refvecptrfunc = refw<std::vector<std::shared_ptr<Function>>>;

	export std::string ldl_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<Function> ldl(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void ldl_UNIQUEID(inout float mat[ndim*ndim]) {
	float arr[ndim];

	for (int i = 0; i < ndim; ++i) {
		float d = mat[i*ndim + i];

		for (int j = i + 1; j < ndim; ++j) {
			arr[j] = mat[j*ndim + i];
			mat[j*ndim + i] /= d;
		}

		for (int j = i + 1; j < ndim; ++j) {
			float aj = arr[j];
			for (int k = j; k < ndim; ++k) {
				mat[k*ndim + j] -= aj * mat[k*ndim + i];
			}
		}
	}

}
)glsl";

		std::string uniqueid = ldl_uniqueid(ndim, single_precision);

		std::function<std::string()> code_func = [ndim, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"ldl_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);


	}

	export FunctionApplier ldl(const std::shared_ptr<MatrixVariable>& mat)
	{
		// type and dim checks
		{
			if (mat->getNDim1() != mat->getNDim2()) {
				throw std::runtime_error("matrices must be square");
			}

			if (!((mat->getType() == ShaderVariableType::eFloat) ||
				(mat->getType() == ShaderVariableType::eDouble))) {
				throw std::runtime_error("Inputs must have float or double type");
			}

		}

		ui16 ndim = mat->getNDim1();
		bool single_precision = true;
		if (mat->getType() == ShaderVariableType::eDouble)
			single_precision = false;

		auto func = ldl(ndim, single_precision);
		auto uniqueid = ldl_uniqueid(ndim, single_precision);

		return FunctionApplier{ func, nullptr, {mat}, uniqueid };
	}


	export std::string gmw81_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<Function> gmw81(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void gmw81_UNIQUEID(inout float mat[ndim*ndim]) {
	float m1 = 0.0;
	float m2 = 0.0;
	float beta2 = 0.0;
	float temp;
	float arr[ndim];

	for (int i = 0; i < ndim; ++i) {
		temp = abs(mat[i*ndim + i]);
		if (m1 < temp)
			m1 = temp;
	}

	if (beta2 < m1)
		beta2 = m1;

	for (int i = 1; i < ndim; ++i) {
		for (int j = 0; j < i; ++j) {
			temp = abs(mat[i*ndim + j]);
			if (m2 < temp)
				m2 = temp;
		}
	}

	if (ndim > 1)
		m2 /= float(sqrt(ndim*ndim - 1));

	if (beta2 < m2)
		beta2 = m2;

	for (int i = 0; i < ndim; ++i) {
		float d = abs(mat[i*ndim + i]);

		if (d < 5e-7)
			d = 5e-7;

		m2 = 0.0;
		for (int j = i + 1; j < ndim; ++j) {
			temp = abs(mat[j*ndim + i]);
			if (m2 < temp)
				m2 = temp;
		}
		
		m2 *= m2;

		if (m2 > d * beta2)
			d = m2 / beta2;

		mat[i*ndim + i] = d;

		for (int j = i + 1; j < ndim; ++j) {
			arr[j] = mat[j*ndim + i];
			mat[j*ndim + i] /= d;
		}

		for (int j = i + 1; j < ndim; ++j) {
			for (int k = j; k < ndim; ++k) {
				mat[k*ndim + j] -= arr[j] * mat[k*ndim + i];
			}
		}

	}

}
)glsl";

		std::string uniqueid = gmw81_uniqueid(ndim, single_precision);

		std::function<std::string()> code_func = [ndim, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"gmw81_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}

	export FunctionApplier gmw81(const std::shared_ptr<MatrixVariable>& mat)
	{
		// type and dim checks
		{
			if (mat->getNDim1() != mat->getNDim2()) {
				throw std::runtime_error("matrices must be square");
			}

			if (!((mat->getType() == ShaderVariableType::eFloat) ||
				(mat->getType() == ShaderVariableType::eDouble))) {
				throw std::runtime_error("Inputs must have float or double type");
			}

		}

		ui16 ndim = mat->getNDim1();
		bool single_precision = true;
		if (mat->getType() == ShaderVariableType::eDouble)
			single_precision = false;

		auto func = gmw81(ndim, single_precision);
		auto uniqueid = gmw81_uniqueid(ndim, single_precision);

		return FunctionApplier{ func, nullptr, {mat}, uniqueid };
	}


	export std::string ldl_solve_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<Function> ldl_solve(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void ldl_solve_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], inout float sol[ndim]) {
	float arr[ndim];
	forward_subs_unit_diaged_FSUDID(mat, rhs, arr);

	backward_subs_unit_t_BSUTID(mat, arr, sol);
}
)glsl";

		std::string uniqueid = ldl_solve_uniqueid(ndim, single_precision);

		std::function<std::string()> code_func = [ndim, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndim", std::to_string(ndim));
			util::replace_all(temp, "FSUDID", linalg::forward_subs_unit_diaged_uniqueid(ndim, single_precision));
			util::replace_all(temp, "BSUTID", linalg::backward_subs_unit_t_uniqueid(ndim, single_precision));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"ldl_solve_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::make_optional<vecptrfunc>({
				linalg::forward_subs_unit_diaged(ndim, single_precision),
				linalg::backward_subs_unit_t(ndim, single_precision)
			})
		);
	}

	export FunctionApplier ldl_solve(const std::shared_ptr<MatrixVariable>& mat,
		const std::shared_ptr<VectorVariable>& rhs, const std::shared_ptr<VectorVariable>& sol)
	{
		// type and dim checks
		{
			if (mat->getNDim1() != mat->getNDim2()) {
				throw std::runtime_error("matrix must be square");
			}
			if (rhs->getNDim() != mat->getNDim1()) {
				throw std::runtime_error("rhs dim must equal mat dim1");
			}
			if (sol->getNDim() != mat->getNDim1()) {
				throw std::runtime_error("sol dim must equal mat dim1");
			}

			if (!((ui16)mat->getType() &
				(ui16)rhs->getType() &
				(ui16)sol->getType()))
			{
				throw std::runtime_error("All inputs must have same type");
			}
			if (!((mat->getType() == ShaderVariableType::eFloat) ||
				(mat->getType() == ShaderVariableType::eDouble))) {
				throw std::runtime_error("Inputs must have float or double type");
			}

		}

		ui16 ndim = mat->getNDim1();
		bool single_precision = true;
		if (mat->getType() == ShaderVariableType::eDouble)
			single_precision = false;

		auto func = ldl_solve(ndim, single_precision);
		auto uniqueid = ldl_solve_uniqueid(ndim, single_precision);

		return FunctionApplier{ func, nullptr, {mat, rhs, sol}, uniqueid };
	}

}
}

