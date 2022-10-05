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

	export std::string max_mag_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> max_mag(ui16 nrow, ui16 ncol, bool single_precission)
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

		std::string uniqueid = max_mag_uniqueid(nrow, ncol, single_precission);

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

		return std::make_shared<Function>(
			"max_mag_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}


	export std::string max_mag_subrow_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> max_mag_subrow(ui16 nrow, ui16 ncol, bool single_precission)
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

		std::string uniqueid = max_mag_subrow_uniqueid(nrow, ncol, single_precission);

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

		return std::make_shared<Function>(
			"max_mag_subrow_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}


	export std::string max_mag_subcol_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> max_mag_subcol(ui16 nrow, ui16 ncol, bool single_precission)
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

		std::string uniqueid = max_mag_subcol_uniqueid(nrow, ncol, single_precission);

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

		return std::make_shared<Function>(
			"max_mag_subcol_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}


	export std::string row_interchange_i_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> row_interchange_i(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void row_interchange_i_UNIQUEID(inout float mat[nrow*ncol], int ii, int jj) {
	for (int k = 0; k < ncol; ++k) {
		swap_SWAPID(mat[ii*ncol + k], mat[jj*ncol + k]);
	}
}
)glsl";

		std::string uniqueid = row_interchange_i_uniqueid(nrow, ncol, single_precission);

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ncol", std::to_string(ncol));
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "SWAPID", swap_uniqueid(single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"row_interchange_i_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::make_optional<vecptrfunc>({
				swap(single_precission)
				})
		);
	}


	export std::string subrow_interchange_i_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> subrow_interchange_i(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void subrow_interchange_i_UNIQUEID(inout float mat[nrow*ncol], int start_col, int ii, int jj) {
	for (int k = start_col; k < ncol; ++k) {
		swap_SWAPID(mat[ii*ncol + k], mat[jj*ncol + k]);
	}
}
)glsl";

		std::string uniqueid = subrow_interchange_i_uniqueid(nrow, ncol, single_precission);

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ncol", std::to_string(ncol));
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "SWAPID", swap_uniqueid(single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"subrow_interchange_i_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::make_optional<vecptrfunc>({ swap(single_precission) })
		);
	}


	export std::string col_interchange_i_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> col_interchange_i(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void col_interchange_i_UNIQUEID(inout float mat[nrow*ncol], int ii, int jj) {
	for (int k = 0; k < nrow; ++k) {
		swap_SWAPID(mat[k*ncol + ii], mat[k*ncol + jj]);
	}
}
)glsl";

		std::string uniqueid = col_interchange_i_uniqueid(nrow, ncol, single_precission);

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ncol", std::to_string(ncol));
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "SWAPID", swap_uniqueid(single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"col_interchange_i_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::make_optional<vecptrfunc>({ swap(single_precission) })
		);
	}


	export std::string subcol_interchange_i_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> subcol_interchange_i(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void col_interchange_i_UNIQUEID(inout float mat[nrow*ncol], int start_row, int ii, int jj) {
	for (int k = start_row; k < nrow; ++k) {
		swap_SWAPID(mat[k*ncol + ii], mat[k*ncol + jj]);
	}
}
)glsl";

		std::string uniqueid = subcol_interchange_i_uniqueid(nrow, ncol, single_precission);

		std::function<std::string()> code_func = [nrow, ncol, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ncol", std::to_string(ncol));
			util::replace_all(temp, "nrow", std::to_string(nrow));
			util::replace_all(temp, "SWAPID", swap_uniqueid(single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"col_interchange_i_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::make_optional<vecptrfunc>({ swap(single_precission) })
		);
	}


}
}