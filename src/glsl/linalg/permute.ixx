module;

export module permute;

import <string>;
import <optional>;
import <memory>;
import <functional>;
import <optional>;

import vc;
import glsl;
import util;

export import function;

import copy;

using namespace vc;

namespace glsl {
namespace linalg {

	using vecptrfunc = std::vector<std::shared_ptr<Function>>;
	using refvecptrfunc = refw<std::vector<std::shared_ptr<Function>>>;

	export std::string max_mag_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> max_mag(ui16 nrow, ui16 ncol, bool single_precision)
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

		std::string uniqueid = max_mag_uniqueid(nrow, ncol, single_precision);

		std::function<std::string()> code_func = [nrow, ncol, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"max_mag_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt);
	}

	export std::string max_diagonal_abs_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> max_diagonal_abs(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
int max_diagonal_abs_UNIQUEID(in float mat[ndim*ndim], int offset) {
	float max_abs = -1.0;
	int max_index = 0;
	for (int i = offset; i < ndim; ++i) {
		if (abs(mat[i*ndim+i]) > max_abs) {
			max_index = i;
		}
	}
	return max_index;
}
)glsl";

		std::string uniqueid = max_diagonal_abs_uniqueid(ndim, single_precision);

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
			"max_diagonal_abs_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt);
	}


	export std::string max_mag_subrow_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> max_mag_subrow(ui16 nrow, ui16 ncol, bool single_precision)
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

		std::string uniqueid = max_mag_subrow_uniqueid(nrow, ncol, single_precision);

		std::function<std::string()> code_func = [nrow, ncol, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"max_mag_subrow_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt);
	}


	export std::string max_mag_subcol_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> max_mag_subcol(ui16 nrow, ui16 ncol, bool single_precision)
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

		std::string uniqueid = max_mag_subcol_uniqueid(nrow, ncol, single_precision);

		std::function<std::string()> code_func = [nrow, ncol, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "ncol", std::to_string(ncol));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"max_mag_subcol_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt);
	}


	export std::string row_interchange_i_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> row_interchange_i(ui16 nrow, ui16 ncol, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void row_interchange_i_UNIQUEID(inout float mat[nrow*ncol], int ii, int jj) {
	for (int k = 0; k < ncol; ++k) {
		swap_SWAPID(mat[ii*ncol + k], mat[jj*ncol + k]);
	}
}
)glsl";

		std::string uniqueid = row_interchange_i_uniqueid(nrow, ncol, single_precision);

		std::function<std::string()> code_func = [nrow, ncol, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ncol", std::to_string(ncol));
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "SWAPID", swap_uniqueid(single_precision));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"row_interchange_i_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::make_optional<vecptrfunc>({
				swap(single_precision)
				})
		);
	}


	export std::string subrow_interchange_i_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> subrow_interchange_i(ui16 nrow, ui16 ncol, bool single_precision)
	{
		static const std::string code = // compute shader
			R"glsl(
void subrow_interchange_i_UNIQUEID(inout float mat[nrow*ncol], int start_col, int ii, int jj) {
	for (int k = start_col; k < ncol; ++k) {
		swap_SWAPID(mat[ii*ncol + k], mat[jj*ncol + k]);
	}
}
)glsl";

		std::string uniqueid = subrow_interchange_i_uniqueid(nrow, ncol, single_precision);

		std::function<std::string()> code_func = [nrow, ncol, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ncol", std::to_string(ncol));
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "SWAPID", swap_uniqueid(single_precision));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"subrow_interchange_i_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::make_optional<vecptrfunc>({ swap(single_precision) })
		);
	}


	export std::string col_interchange_i_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> col_interchange_i(ui16 nrow, ui16 ncol, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void col_interchange_i_UNIQUEID(inout float mat[nrow*ncol], int ii, int jj) {
	for (int k = 0; k < nrow; ++k) {
		swap_SWAPID(mat[k*ncol + ii], mat[k*ncol + jj]);
	}
}
)glsl";

		std::string uniqueid = col_interchange_i_uniqueid(nrow, ncol, single_precision);

		std::function<std::string()> code_func = [nrow, ncol, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ncol", std::to_string(ncol));
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "SWAPID", swap_uniqueid(single_precision));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"col_interchange_i_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::make_optional<vecptrfunc>({ swap(single_precision) })
		);
	}


	export std::string subcol_interchange_i_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> subcol_interchange_i(ui16 nrow, ui16 ncol, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void col_interchange_i_UNIQUEID(inout float mat[nrow*ncol], int start_row, int ii, int jj) {
	for (int k = start_row; k < nrow; ++k) {
		swap_SWAPID(mat[k*ncol + ii], mat[k*ncol + jj]);
	}
}
)glsl";

		std::string uniqueid = subcol_interchange_i_uniqueid(nrow, ncol, single_precision);

		std::function<std::string()> code_func = [nrow, ncol, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ncol", std::to_string(ncol));
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "SWAPID", swap_uniqueid(single_precision));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"col_interchange_i_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::make_optional<vecptrfunc>({ swap(single_precision) })
		);
	}


	export std::string diagonal_pivoting_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> diagonal_pivoting(ui16 ndim, bool single_precision) {
		static const std::string code = // compute shader
R"glsl(
void diagonal_pivoting_UNIQUEID(inout float mat[ndim*ndim], inout int perm[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		perm[i] = i;
	}
	for (int i = 0; i < ndim; ++i) {
		int max_abs = max_diagonal_abs_MDAID(mat, i);
		row_interchange_i_RIIID(mat, i, max_abs);
		col_interchange_i_CIIID(mat, i, max_abs);
		int temp = perm[i];
		perm[i] = perm[max_abs];
		perm[max_abs] = temp;
	}
}
)glsl";

		std::string uniqueid = diagonal_pivoting_uniqueid(ndim, single_precision);

		std::function<std::string()> code_func = [ndim, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndim", std::to_string(ndim));
			util::replace_all(temp, "MDAID", linalg::max_diagonal_abs_uniqueid(ndim, single_precision));
			util::replace_all(temp, "RIIID", linalg::row_interchange_i_uniqueid(ndim, ndim, single_precision));
			util::replace_all(temp, "CIIID", linalg::col_interchange_i_uniqueid(ndim, ndim, single_precision));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"diagonal_pivoting_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::make_optional<vecptrfunc>({
				linalg::max_diagonal_abs(ndim, single_precision),
				linalg::row_interchange_i(ndim, ndim, single_precision),
				linalg::col_interchange_i(ndim, ndim, single_precision)
				})
			);
	}


	export std::string permute_vec_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> permute_vec(ui16 ndim, bool single_precision) {
		static const std::string code = // compute shader
R"glsl(
void permute_vec_UNIQUEID(in float vec[ndim], in int perm[ndim], out float ovec[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		ovec[i] = vec[perm[i]];
	}
}
)glsl";

		std::string uniqueid = permute_vec_uniqueid(ndim, single_precision);

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
			"permute_vec_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
			);
	}


	export std::string permute_o_vec_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> permute_o_vec(ui16 ndim, bool single_precision) {
		static const std::string code = // compute shader
R"glsl(
void permute_o_vec_UNIQUEID(in float vec[ndim], in int perm[ndim], out float ovec[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		ovec[perm[i]] = vec[i];
	}
}
)glsl";

		std::string uniqueid = permute_o_vec_uniqueid(ndim, single_precision);

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
			"permute_o_vec_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
			);
	}

	
}
}