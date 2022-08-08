module;

#include <string>
#include <optional>

export module linalg_glsl;

import glsl;
import util;

// INTERFACE
namespace glsl {
namespace linalg {
	
	export ::glsl::Function swap(bool single_precission = true);

	export ::glsl::Function copy_mat(int nrow, int ncol, bool single_precission = true);

	export ::glsl::Function copy_vec(int ndim, bool single_precission = true);

	export ::glsl::Function row_interchange_i(int nrow, int ncol, bool single_precission = true);

	export ::glsl::Function column_interchange_i(int nrow, int ncol, bool single_precission = true);

	export ::glsl::Function mul_mat_mat(int lnrow, int mid_dim, int rncol, bool single_precission = true);

	export ::glsl::Function mul_mat_vec(int nrow, int ncol, bool single_precission = true);

namespace square {

	export ::glsl::Function transpose_i(int ndim, bool single_precission = true);

	export ::glsl::Function mul_diag_vec_i(int ndim, bool single_precission = true);

	export ::glsl::Function mul_inv_diag_vec_i(int ndim, bool single_precission = true);

	export ::glsl::Function mul_mat_mat(int ndim, bool single_precission = true);

	export ::glsl::Function mul_mat_vec(int ndim, bool single_precission = true);

}

}
}


// IMPLEMENTATION
namespace glsl {
namespace linalg {

	::glsl::Function swap(bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void swap(inout float v1, inout float v2) {
	float temp = v1;
	v1 = v2;
	v2 = temp;
}
)glsl";

		std::function<std::string()> code_func = [single_precission]() -> std::string
		{
			std::string temp = code;
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"swap",
			{ size_t(single_precission) },
			code_func, 
			std::nullopt);
	}

	::glsl::Function copy_mat(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_mat(in float imat[nrow*ncol], out float omat[nrow*ncol]) {
	for (int i = 0; i < nrow; ++i) {
		for (int j = 0; j < ncol; ++j) {
			omat[i*ncol + j] = imat[i*ncol + j];
		}
	}
}
)glsl";

		std::function<std::string()> code_func = [nrow, ncol, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"copy_mat",
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	::glsl::Function copy_vec(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_vec(in float ivec[ndim], out float ovec[ndim]) {
	for (int i = 0; i < ndim; ++i) {
			ovec[i] = ivec[i];
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
			"copy_vec",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	::glsl::Function row_interchange_i(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void row_interchange_i(inout float mat[nrow*ncol], int ii, int jj) {
	for (int k = 0; k < ncol; ++k) {
		swap(mat[ii*ncol + k], mat[jj*ncol + k]);
	}
}
)glsl";

		std::function<std::string()> code_func = [nrow, ncol, single_precission]() -> std::string
		{
			std::string temp = code;
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
				util::replace_all(temp, "ncol", std::to_string(ncol));
				util::replace_all(temp, "nrow", std::to_string(nrow));
			}
			return temp;
		};

		return ::glsl::Function(
			"row_interchange_i",
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({ swap(single_precission) })
		);
	}

	::glsl::Function column_interchange_i(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void column_interchange_i(inout float mat[nrow*ncol], int ii, int jj) {
	for (int k = 0; k < nrow; ++k) {
		swap(mat[k*ncol + ii], mat[k*ncol + jj]);
	}
}
)glsl";

		std::function<std::string()> code_func = [nrow, ncol, single_precission]() -> std::string
		{
			std::string temp = code;
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
				util::replace_all(temp, "ncol", std::to_string(ncol));
				util::replace_all(temp, "nrow", std::to_string(nrow));
			}
			return temp;
		};

		return ::glsl::Function(
			"column_interchange_i",
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({ swap(single_precission) })
		);
	}

	::glsl::Function mul_mat_mat(int lnrow, int mid_dim, int rncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_mat_mat(in float lmat[lnrow*mid_dim], in float rmat[mid_dim*rncol], out float omat[lnrow*rncol]) {
	float entry;
	for (int i = 0; i < lnrow; ++i) {
		for (int j = 0; j < rncol; ++j) {
			entry = 0.0;
			for (int k = 0; k < mid_dim; ++k) {
				entry += lmat[i*mid_dim + k] * rmat[k*rncol + j];
			}
			omat[i*rncol + j] = entry;
		}
	}
}
)glsl";

		std::function<std::string()> code_func = [lnrow, mid_dim, rncol, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "lnrow", std::to_string(lnrow));
			util::replace_all(temp, "mid_dim", std::to_string(mid_dim));
			util::replace_all(temp, "rncol", std::to_string(rncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"mul_mat_mat",
			{ size_t(lnrow), size_t(mid_dim), size_t(rncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function mul_mat_vec(int nrow, int ncol, bool single_precission) 
	{
		static const std::string code = // compute shader
R"glsl(
void mul_mat_mat(in float lmat[nrow*ncol], in float rvec[ncol], out float ovec[nrow]) {
	float entry;
	for (int i = 0; i < nrow; ++i) {
		entry = 0.0;
		for (int j = 0; j < ncol; ++j) {
			entry += lmat[i*ncol + j] * rvec[j];
		}
		ovec[i] = entry;
	}
}
)glsl";
		
		std::function<std::string()> code_func = [nrow, ncol, single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"mul_mat_vec",
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


namespace square {
	
	::glsl::Function transpose_i(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void square_transpose_i(inout float mat[ndim*ndim]) {
	for (int i = 1; i < ndim; ++i) {
		for (int j = 0; j < i; ++j) {
			swap(mat[i*ndim + j], mat[j*n + i]);
		}
	}
}
)glsl";

		std::function<std::string()> code_func = [ndim, single_precission]() -> std::string
		{
			std::string temp = code;
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
				util::replace_all(temp, "ndim", std::to_string(ndim));
			}
			return temp;
		};

		return ::glsl::Function(
			"square_transpose_i",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({ swap(single_precission) })
		);

	}

	::glsl::Function mul_diag_vec_i(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_diag_vec_i(in float mat[ndim*ndim], inout float vec[ndim]) {
	for (int k = 0; k < ndim; ++k) {
		vec[k] *= mat[i*ndim + i];
	}
}
)glsl";

		std::function<std::string()> code_func = [ndim, single_precission]() -> std::string
		{
			std::string temp = code;
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
				util::replace_all(temp, "ndim", std::to_string(ndim));
			}
			return temp;
		};

		return ::glsl::Function(
			"mul_diag_vec_i",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function mul_inv_diag_vec_i(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_inv_diag_vec_i(in float mat[ndim*ndim], inout float vec[ndim]) {
	for (int k = 0; k < ndim; ++k) {
		vec[k] /= mat[i*ndim + i];
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
			"mul_inv_diag_vec_i",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function mul_mat_mat(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_mat_mat(in float lmat[ndim*ndim], in float rmat[ndim*ndim], inout float omat[ndim*ndim]) {
	float entry;
	for (int i = 0; i < ndim; ++i) {
		for (int j = 0; j < ndim; ++j) {
			entry = 0.0;
			for (int k = 0; k < ndim; ++k) {
				entry += lmat[i*ndim + k] * rmat[k*ndim + j];
			}
			omat[i*ndim + j] = entry;
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
			"mul_mat_mat",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	::glsl::Function mul_mat_vec(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_mat_vec(in float lmat[ndim*ndim], in float rvec[ndim], inout float ovec[ndim]) {
	float entry;
	for (int i = 0; i < ndim; ++i) {
		entry = 0.0;
		for (int j = 0; j < ndim; ++j) {
			entry += lmat[i*ndim + j] * rvec[j];
		}
		ovec[i] = entry;
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
			"mul_mat_vec",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

}

}
}
