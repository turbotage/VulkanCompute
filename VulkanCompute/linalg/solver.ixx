module;

#include <string>
#include <vector>
#include <optional>

export module solver;

import util;
import glsl;
import linalg;

// INTERFACE
namespace glsl {
namespace linalg {
namespace solver {

	export ::glsl::Function forward_subs(int ndim, bool single_precission = true);

	export ::glsl::Function forward_subs_t(int ndim, bool single_precission = true);

	export ::glsl::Function forward_subs_unit(int ndim, bool single_precission = true);

	export ::glsl::Function forward_subs_unit_t(int ndim, bool single_precission = true);

	export ::glsl::Function backward_subs(int ndim, bool single_precission = true);

	export ::glsl::Function backward_subs_t(int ndim, bool single_precission = true);

	export ::glsl::Function backward_subs_unit(int ndim, bool single_precission = true);

	export ::glsl::Function backward_subs_unit_t(int ndim, bool single_precission = true);

	export ::glsl::Function lu(int ndim, bool single_precission = true);

}
}
}



// IMPLEMENTATION
namespace glsl {
namespace linalg {
namespace solver {

	::glsl::Function forward_subs(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void forward_subs(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= mat[i*ndim + j] * solution[j];
		}
		solution[i] /= mat[i*ndim + i];
	}
}
)glsl";

		std::function<std::string()> code_func = [ndim, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"forward_subs",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function forward_subs_t(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void forward_subs_t(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= mat[j*ndim + i] * solution[j];
		}
		solution[i] /= mat[i*ndim + i];
	}
}
)glsl";

		std::function<std::string()> code_func = [ndim, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"forward_subs_t",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function forward_subs_unit(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void forward_subs_unit(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= mat[i*ndim + j] * solution[j];
		}
	}
}
)glsl";

		std::function<std::string()> code_func = [ndim, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"forward_subs_unit",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function forward_subs_unit_t(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void forward_subs_unit_t(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= mat[j*ndim + i] * solution[j];
		}
	}
}
)glsl";

		std::function<std::string()> code_func = [ndim, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"forward_subs_unit_t",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function backward_subs(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void backward_subs(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= mat[i*ndim + j] * solution[j];
		}
		solution[i] /= mat[i*ndim + i];
	}
}
)glsl";

		std::function<std::string()> code_func = [ndim, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"backward_subs",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function backward_subs_t(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void backward_subs_t(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= mat[j*ndim + i] * solution[j];
		}
		solution[i] /= mat[i*ndim + i];
	}
}
)glsl";

		std::function<std::string()> code_func = [ndim, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"backward_subs_t",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function backward_subs_unit(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void backward_subs_unit(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= mat[i*ndim + j] * solution[j];
		}
	}
}
)glsl";

		std::function<std::string()> code_func = [ndim, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"backward_subs_unit",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function backward_subs_unit_t(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void backward_subs_unit_t(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= mat[j*ndim + i] * solution[j];
		}
	}
}
)glsl";

		std::function<std::string()> code_func = [ndim, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"backward_subs_unit_t",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function lu(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void lu(in float mat[ndim*ndim], out float pivot[ndim]) {
	float val;
	for (int k = 0; k < ndim - 2; ++k) {
		int max_row_idx;
		float max_row_value;
		max_mag_subrow(mat, k, max_row_idx, max_row_value);		
		row_interchange_i(mat, k, max_row_idx);
		pivot[k] = max_row_idx;
		
		val = A[k*ndim + k];
		if (val > machine_eps) {
			for (int i = k + 1; i < ndim; ++i) {
				mat[i*ndim + k] /= val; 
			}
			
			for (int i = k + 1; i < ndim; ++i) {
				for (int j = k + 1; j < ndim; ++j) {
					mat[i*ndim + j] -= mat[i*ndim + k]*mat[k*ndim + j]
				}
			}
		}
	}
}
)glsl";

		std::function<std::string()> code_func = [ndim, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
				util::replace_all(temp, "machine_eps", "1e-15");
			}
			else {
				util::replace_all(temp, "machine_eps", "1e-6");
			}
			return temp;
		};

		return ::glsl::Function(
			"lu",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({ max_mag_subrow(ndim, ndim, single_precission), row_interchange_i(ndim, ndim, single_precission) })
		);
	}

}
}
}

