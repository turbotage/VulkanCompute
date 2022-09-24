module;

#include <vector>
#include <string>
#include <optional>

export module symm;

import util;
import solver;
export import linalg;
export import glsl;

namespace glsl {
namespace linalg {

	constexpr auto UNIQUE_ID = "UNIQUEID";

	::glsl::Function ldl(int ndim, bool single_precission = true)
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

		std::string uniqueid = std::to_string(ndim) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [ndim, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"ldl_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);


	}

	::glsl::Function gmw81(int ndim, bool single_precission = true)
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

		std::string uniqueid = std::to_string(ndim) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [ndim, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"gmw81_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function ldl_solve(int ndim, bool single_precission = true)
	{
		static const std::string code = // compute shader
R"glsl(
void ldl_solve_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], inout float sol[ndim]) {
	float diag[ndim];
	for (int i = 0; i < ndim; ++i) {
		diag[i] = mat[i*ndim + i];
	}

	float arr[ndim];
	forward_subs_unit_diag(mat, rhs, diag, arr);

	backward_subs_unit_t(mat, arr, sol);
}
)glsl";

		std::string uniqueid = std::to_string(ndim) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [ndim, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"ldl_solve_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({
				linalg::forward_subs_unit_diag(ndim, single_precission),
				linalg::backward_subs_unit_t(ndim, single_precission)
			})
		);
	}

}
}

