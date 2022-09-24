module;

#include <string>
#include <optional>

export module linalg;

import glsl;
import util;

// IMPLEMENTATION
namespace glsl {
namespace linalg {

	constexpr auto UNIQUE_ID = "UNIQUEID";

	export ::glsl::Function swap(bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void swap_UNIQUEID(inout float v1, inout float v2) {
	float temp = v1;
	v1 = v2;
	v2 = temp;
}
)glsl";

		std::function<std::string()> code_func = [single_precission]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, single_precission ? "S" : "D");
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"swap_" + single_precission ? "S" : "D",
			{ size_t(single_precission) },
			code_func, 
			std::nullopt);
	}

	export ::glsl::Function copy_mat(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_mat_UNIQUEID(in float imat[nrow*ncol], out float omat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[i] = imat[i];
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"copy_mat_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::Function copy_mat_ostart(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_mat_ostart_UNIQUEID(in float imat[nrow*ncol], out float omat[nrow*ncol], uint start_index) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[start_index + i] = imat[i];
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"copy_mat_ostart_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::Function copy_mat_ostarted(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void copy_mat_ostarted_UNIQUEID(in float imat[nrow*ncol], out float omat[nrow*ncol]) {
	uint start_index = nrow*ncol*gl_GlobalInvocationID.x;
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[start_index + i] = imat[i];
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"copy_mat_ostarted_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::Function copy_mat_istart(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_mat_istart_UNIQUEID(in float imat[nrow*ncol], out float omat[nrow*ncol], uint start_index) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[i] = imat[start_index + i];
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"copy_mat_istart_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::Function copy_mat_istarted(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void copy_mat_istarted_UNIQUEID(in float imat[nrow*ncol], out float omat[nrow*ncol]) {
	uint start_index = nrow*ncol*gl_GlobalInvocationID.x;
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[i] = imat[start_index + i];
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"copy_mat_istarted_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::Function copy_vec(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_vec_UNIQUEID(in float ivec[ndim], out float ovec[ndim]) {
	for (int i = 0; i < ndim; ++i) {
			ovec[i] = ivec[i];
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
			"copy_vec_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::Function copy_vec_ostart(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_vec_ostart_UNIQUEID(in float ivec[ndim], out float ovec[ndim], uint start_index) {
	for (int i = 0; i < ndim; ++i) {
			ovec[start_index + i] = ivec[i];
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
			"copy_vec_ostart_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::Function copy_vec_ostarted(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void copy_vec_ostarted_UNIQUEID(in float ivec[ndim], out float ovec[ndim]) {
	uint start_index = ndim*gl_GlobalInvocationID.x;
	for (int i = 0; i < ndim; ++i) {
			ovec[start_index + i] = ivec[i];
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
			"copy_vec_ostarted_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::Function copy_vec_istart(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_vec_istart_UNIQUEID(in float ivec[ndim], out float ovec[ndim], uint start_index) {
	for (int i = 0; i < ndim; ++i) {
			ovec[i] = ivec[start_index + i];
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
			"copy_vec_istart_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::Function copy_vec_istarted(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void copy_vec_istarted_UNIQUEID(in float ivec[ndim], out float ovec[ndim]) {
	uint start_index = ndim*gl_GlobalInvocationID.x;
	for (int i = 0; i < ndim; ++i) {
			ovec[i] = ivec[start_index + i];
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
			"copy_vec_istarted_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::Function max_mag(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void max_mag_UNIQUEID(in float mat[nrow*ncol], out int max_row_idx, out int max_col_idx, out float max) {
	max_row_idx = 0;
	max_col_idx = 0;
	max = 0.0;
	float val;
	for (int i = 0; i < nrow; ++i) {
		for (int j = 0; j < ncol; ++j) {
			val = mat[i*ncol + j];
			if (val > max) {
				max_row_idx = i;
				max_col_idx = j;
				max = val;
			}
		}
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"max_mag_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::Function max_mag_subrow(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void max_mag_subrow_UNIQUEID(in float mat[nrow*ncol], int row, int start_col, out int max_idx, out float max) {
	max_idx = 0;
	max = 0.0;
	float val;
	for (int i = start_col; i < ncol; ++i) {
		val = mat[row*ncol + i];
		if (val > max) {
			max_idx = i;
			max = val;
		}
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"max_mag_subrow_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::Function max_mag_subcol(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void max_mag_subcol_UNIQUEID(in float mat[nrow*ncol], int col, int start_row, out int max_idx, out float max) {
	max_idx = 0;
	max = 0.0;
	float val;
	for (int i = start_row; i < nrow; ++i) {
		val = mat[i*ncol + col];
		if (val > max) {
			max_idx = i;
			max = val;
		}
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"max_mag_subcol_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}

	export ::glsl::Function row_interchange_i(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void row_interchange_i_UNIQUEID(inout float mat[nrow*ncol], int ii, int jj) {
	for (int k = 0; k < ncol; ++k) {
		swap(mat[ii*ncol + k], mat[jj*ncol + k]);
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ncol", std::to_string(ncol));
			util::replace_all(temp, "nrow", std::to_string(nrow));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"row_interchange_i_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({ swap(single_precission) })
		);
	}

	export ::glsl::Function subrow_interchange_i(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void subrow_interchange_i_UNIQUEID(inout float mat[nrow*ncol], int start_col, int ii, int jj) {
	for (int k = start_col; k < ncol; ++k) {
		swap(mat[ii*ncol + k], mat[jj*ncol + k]);
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ncol", std::to_string(ncol));
			util::replace_all(temp, "nrow", std::to_string(nrow));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"subrow_interchange_i_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({ swap(single_precission) })
		);
	}

	export ::glsl::Function col_interchange_i(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void col_interchange_i_UNIQUEID(inout float mat[nrow*ncol], int ii, int jj) {
	for (int k = 0; k < nrow; ++k) {
		swap(mat[k*ncol + ii], mat[k*ncol + jj]);
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ncol", std::to_string(ncol));
			util::replace_all(temp, "nrow", std::to_string(nrow));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"col_interchange_i_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({ swap(single_precission) })
		);
	}

	export ::glsl::Function subcol_interchange_i(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void col_interchange_i_UNIQUEID(inout float mat[nrow*ncol], int start_row, int ii, int jj) {
	for (int k = start_row; k < nrow; ++k) {
		swap(mat[k*ncol + ii], mat[k*ncol + jj]);
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ncol", std::to_string(ncol));
			util::replace_all(temp, "nrow", std::to_string(nrow));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"col_interchange_i_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({ swap(single_precission) })
		);
	}

	export ::glsl::Function mul_mat_mat(int lnrow, int mid_dim, int rncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_mat_mat_UNIQUEID(in float lmat[lnrow*mid_dim], in float rmat[mid_dim*rncol], out float omat[lnrow*rncol]) {
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

		std::string uniqueid = std::to_string(lnrow) + "_" + std::to_string(mid_dim) + "_" + std::to_string(rncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [lnrow, mid_dim, rncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "lnrow", std::to_string(lnrow));
			util::replace_all(temp, "mid_dim", std::to_string(mid_dim));
			util::replace_all(temp, "rncol", std::to_string(rncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"mul_mat_mat_" + uniqueid,
			{ size_t(lnrow), size_t(mid_dim), size_t(rncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::Function mul_mat_transpose(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_mat_transpose_UNIQUEID(in float mat[nrow*ncol], out float omat[nrow*nrow]) {
	float entry;
	for (int i = 0; i < nrow; ++i) {
		for (int j = 0; j <= i; ++j) {
			entry = 0.0;
			for (int k = 0; k < ncol; ++k) {
				entry += mat[i*ncol + k] * mat[j*ncol + k];
			}
			omat[i*nrow + j] = entry;
			if (i != j) {
				omat[j*nrow + i] = entry;
			}
		}
	}

}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"mul_mat_transpose_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::Function mul_mat_transpose_add(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void mul_mat_transpose_add_UNIQUEID(in float mat[nrow*ncol], inout float omat[nrow*nrow]) {
	float entry;
	for (int i = 0; i < nrow; ++i) {
		for (int j = 0; j <= i; ++j) {
			entry = 0.0;
			for (int k = 0; k < ncol; ++k) {
				entry += mat[i*ncol + k] * mat[j*ncol + k];
			}
			omat[i*nrow + j] += entry;
			if (i != j) {
				omat[j*nrow + i] += entry;
			}
		}
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"mul_mat_transpose_add_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::Function mul_transpose_mat(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void mul_transpose_mat_UNIQUEID(in float mat[nrow*ncol], out float omat[ncol*ncol]) {
	float entry;
	for (int i = 0; i < ncol; ++i) {
		for (int j = 0; j <= i; ++j) {
			entry = 0.0;
			for (int k = 0; k < nrow; ++k) {
				entry += mat[k*ncol + i] * mat[k*ncol + j];
			}
			omat[i*ncol + j] = entry;
			if (i != j) {
				omat[j*ncol + i] = entry;
			}
		}
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"mul_transpose_mat_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::Function mul_transpose_mat_add(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void mul_transpose_mat_add_UNIQUEID(in float mat[nrow*ncol], inout float omat[ncol*ncol]) {
	float entry;
	for (int i = 0; i < ncol; ++i) {
		for (int j = 0; j <= i; ++j) {
			entry = 0.0;
			for (int k = 0; k < nrow; ++k) {
				entry += mat[k*ncol + i] * mat[k*ncol + j];
			}
			omat[i*ncol + j] += entry;
			if (i != j) {
				omat[j*ncol + i] += entry;
			}
		}
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"mul_transpose_mat_add_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::Function mul_mat_vec(int nrow, int ncol, bool single_precission) 
	{
		static const std::string code = // compute shader
R"glsl(
void mul_mat_vec_UNIQUEID(in float lmat[nrow*ncol], in float rvec[ncol], out float ovec[nrow]) {
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
		
		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"mul_mat_vec_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}
	
	export ::glsl::Function mat_set_zero(int nrow, int ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mat_set_zero_UNIQUEID(inout float mat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		mat[i] = 0;
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"mat_set_zero_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::Function mat_set_ones(int nrow, int ncol, bool single_precission) {
		static const std::string code = // compute shader
R"glsl(
void mat_set_ones_UNIQUEID(inout float mat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		mat[i] = 1;
	}
}
)glsl";

		std::string uniqueid = std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"mat_set_ones_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	// SQUARE

	export ::glsl::Function transpose_square_i(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void transpose_square_i_UNIQUEID(inout float mat[ndim*ndim]) {
	for (int i = 1; i < ndim; ++i) {
		for (int j = 0; j < i; ++j) {
			swap(mat[i*ndim + j], mat[j*n + i]);
		}
	}
}
)glsl";

		std::string uniqueid = std::to_string(ndim) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [ndim, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
				util::replace_all(temp, "ndim", std::to_string(ndim));
			}
			return temp;
		};

		return ::glsl::Function(
			"transpose_square_i_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({ swap(single_precission) })
		);

	}

	export ::glsl::Function mul_diag_vec_square_i(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_diag_vec_square_i_UNIQUEID(in float mat[ndim*ndim], inout float vec[ndim]) {
	for (int k = 0; k < ndim; ++k) {
		vec[k] *= mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = std::to_string(ndim) + "_" + (single_precission ? "S" : "D");

		std::function<std::string()> code_func = [ndim, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			if (!single_precission) {
				util::replace_all(temp, UNIQUE_ID, uniqueid);
				util::replace_all(temp, "float", "double");
				util::replace_all(temp, "ndim", std::to_string(ndim));
			}
			return temp;
		};

		return ::glsl::Function(
			"mul_diag_vec_square_i_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::Function mul_inv_diag_vec_square_i(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_inv_diag_vec_square_i_UNIQUEID(in float mat[ndim*ndim], inout float vec[ndim]) {
	for (int k = 0; k < ndim; ++k) {
		vec[k] /= mat[i*ndim + i];
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
			"mul_inv_diag_vec_square_i_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::Function mul_mat_mat_square(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_mat_mat_square_UNIQUEID(in float lmat[ndim*ndim], in float rmat[ndim*ndim], inout float omat[ndim*ndim]) {
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
			"mul_mat_mat_square_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}
	
	export ::glsl::Function mul_unit_lower_upper_square(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_unit_lower_upper_square_UNIQUEID(in float lmat[ndim*ndim], in float rmat[ndim*ndim], inout float omat[ndim*ndim]) {
	float entry;
	int kmax;
	for (int i = 0; i < ndim; ++i) {
		for (int j = 0; j < ndim; ++j) {
			entry = 0.0;
			kmax = ((i < j) ? i : j) + 1;
			for (int k = 0; k < kmax; ++k) {
				entry += (i == k) ? rmat[k*ndim + j] : lmat[i*ndim + k] * rmat[k*ndim + j];
			}
			omat[i*ndim + j] = entry;
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
			"mul_unit_lower_upper_square_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::Function mul_mat_vec_square(int ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_mat_vec_square_UNIQUEID(in float lmat[ndim*ndim], in float rvec[ndim], inout float ovec[ndim]) {
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
			"mul_mat_vec_square_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

}
}
