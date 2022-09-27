module;

#include <string>
#include <optional>

export module linalg;

import vc;
import glsl;
import util;

export import copy;
export import permute;

using namespace vc;

// IMPLEMENTATION
namespace glsl {
namespace linalg {

	export std::string mat_neg_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mat_neg(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void mag_neg_UNIQUEID(inout float mat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		mat[i] = -mat[i];
	}
}
)glsl";

		std::string uniqueid = mat_neg_uniqueid(nrow, ncol, single_precission);

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
			"mat_neg_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string vec_neg_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function vec_neg(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void vec_neg_UNIQUEID(inout float vec[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		vec[i] = -vec[i];
	}
}
)glsl";

		std::string uniqueid = vec_neg_uniqueid(ndim, single_precission);

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
			"vec_neg_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string dotprod_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function dotprod(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
float dotprod_UNIQUEID(in float vec1[ndim], in float vec2[ndim]) {
	float ret = 0;
	for (int i = 0; i < ndim; ++i) {
		ret += vec1[i] * vec2[i];
	}
	return ret;
}
)glsl";

		std::string uniqueid = dotprod_uniqueid(ndim, single_precission);

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
			"dotprod_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string vec_norm_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function vec_norm(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
float vec_norm_UNIQUEID(in float vec[ndim]) {
	float ret = 0;
	for (int i = 0; i < ndim; ++i) {
		ret += vec[i]*vec[i];
	}
	return sqrt(ret);
}
)glsl";

		std::string uniqueid = vec_norm_uniqueid(ndim, single_precission);

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
			"vec_norm_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string vec_norm2_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function vec_norm2(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
float vec_norm2_UNIQUEID(in float vec[ndim]) {
	float ret = 0;
	for (int i = 0; i < ndim; ++i) {
		ret += vec[i]*vec[i];
	}
	return ret;
}
)glsl";

		std::string uniqueid = vec_norm2_uniqueid(ndim, single_precission);

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
			"vec_norm2_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_mat_mat_uniqueid(ui16 lnrow, ui16 mid_dim, ui16 rncol, bool single_precission)
	{
		return std::to_string(lnrow) + "_" + std::to_string(mid_dim) + "_" + std::to_string(rncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_mat_mat(ui16 lnrow, ui16 mid_dim, ui16 rncol, bool single_precission)
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

		std::string uniqueid = mul_mat_mat_uniqueid(lnrow, mid_dim, rncol, single_precission);

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

	
	export std::string mul_mat_vec_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}
	
	export ::glsl::Function mul_mat_vec(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void mul_mat_vec_UNIQUEID(in float mat[nrow*ncol], in float vec[ncol], out float ovec[nrow]) {
	for (int i = 0; i < nrow; ++i) {
		ovec[i] = 0;
		for (int j = 0; j < ncol; ++j) {
			ovec[i] += mat[i*ncol + j] * vec[j];
		}
	}
}
)glsl";

		std::string uniqueid = mul_mat_vec_uniqueid(nrow, ncol, single_precission);

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


	export std::string mul_transpose_vec_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_transpose_vec(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void mul_transpose_vec_UNIQUEID(in float mat[nrow*ncol], in float vec[nrow], out float ovec[ncol]) {
	for (int i = 0; i < ncol; ++i) {
		ovec[i] = 0;
		for (int j = 0; j < nrow; ++j) {
			ovec[i] += mat[j*ncol + i] * vec[j];
		}
	}
}
)glsl";

		std::string uniqueid = mul_transpose_vec_uniqueid(nrow, ncol, single_precission);

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
			"mul_transpose_vec_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string add_mat_mat_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function add_mat_mat(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void add_mat_mat_UNIQUEID(in float lmat[nrow*ncol], in float rmat[nrow*ncol], out float omat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[i] = lmat[i] + rmat[i];
	}
}
)glsl";

		std::string uniqueid = add_mat_mat_uniqueid(nrow, ncol, single_precission);

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
			"add_mat_mat_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string add_mat_lmat_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function add_mat_lmat(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void add_mat_lmat_UNIQUEID(in float lmat[nrow*ncol], in float rmat[nrow*ncol], float lambda, out float omat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[i] = lmat[i] + lambda * rmat[i];
	}
}
)glsl";

		std::string uniqueid = add_mat_lmat_uniqueid(nrow, ncol, single_precission);

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
			"add_mat_lmat_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string sub_mat_mat_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function sub_mat_mat(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void sub_mat_mat_UNIQUEID(in float lmat[nrow*ncol], in float rmat[nrow*ncol], out float omat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[i] = lmat[i] - rmat[i];
	}
}
)glsl";

		std::string uniqueid = sub_mat_mat_uniqueid(nrow, ncol, single_precission);

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
			"sub_mat_lmat_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string sub_mat_lmat_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function sub_mat_lmat(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void sub_mat_lmat_UNIQUEID(in float lmat[nrow*ncol], in float rmat[nrow*ncol], float lambda, out float omat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[i] = lmat[i] - lambda*rmat[i];
	}
}
)glsl";

		std::string uniqueid = sub_mat_lmat_uniqueid(nrow, ncol, single_precission);

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
			"sub_mat_lmat_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string mat_add_ldiag_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mat_add_ldiag(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mat_add_ldiag_UNIQUEID(inout float mat[ndim*ndim], float lambda) {
	for (int i = 0; i < ndim; ++i) {
		mat[i*ndim + i] += lambda * mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = mat_add_ldiag_uniqueid(ndim, single_precission);

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
			"mat_add_ldiag_" + uniqueid,
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_mat_transpose_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_mat_transpose(ui16 nrow, ui16 ncol, bool single_precission)
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

		std::string uniqueid = mul_mat_transpose_uniqueid(nrow, ncol, single_precission);

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


	export std::string mul_mat_transpose_ldiag_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_mat_transpose_ldiag(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void mul_mat_transpose_ldiag_UNIQUEID(in float mat[nrow*ncol], float lambda, out float omat[nrow*nrow]) {
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
			else {
				omat[i*nrow + j] += lambda * entry;
			}
		}
	}

}
)glsl";

		std::string uniqueid = mul_mat_transpose_ldiag_uniqueid(nrow, ncol, single_precission);

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
			"mul_mat_transpose_ldiag_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_mat_transpose_add_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_mat_transpose_add(ui16 nrow, ui16 ncol, bool single_precission)
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

		std::string uniqueid = mul_mat_transpose_add_uniqueid(nrow, ncol, single_precission);

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


	export std::string mul_mat_transpose_add_ldiag_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_mat_transpose_add_ldiag(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void mul_mat_transpose_add_ldiag_UNIQUEID(in float mat[nrow*ncol], float lambda, inout float omat[nrow*nrow]) {
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
			else {
				omat[j*nrow + i] += lambda * entry;
			}
		}
	}
}
)glsl";

		std::string uniqueid = mul_mat_transpose_add_ldiag_uniqueid(nrow, ncol, single_precission);

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
			"mul_mat_transpose_add_ldiag_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	

	export std::string mul_transpose_mat_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_transpose_mat(ui16 nrow, ui16 ncol, bool single_precission)
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

		std::string uniqueid = mul_transpose_mat_uniqueid(nrow, ncol, single_precission);

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


	export std::string mul_transpose_mat_ldiag_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_transpose_mat_ldiag(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void mul_transpose_mat_ldiag_UNIQUEID(in float mat[nrow*ncol], float lambda, out float omat[ncol*ncol]) {
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
			else {
				omat[j*ncol + i] = lambda * entry;
			}
		}
	}
}
)glsl";

		std::string uniqueid = mul_transpose_mat_ldiag_uniqueid(nrow, ncol, single_precission);

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
			"mul_transpose_mat_ldiag_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_transpose_mat_add_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_transpose_mat_add(ui16 nrow, ui16 ncol, bool single_precission)
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

		std::string uniqueid = mul_transpose_mat_add_uniqueid(nrow, ncol, single_precission);

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


	export std::string mul_transpose_mat_add_ldiag_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_transpose_mat_add_ldiag(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
			R"glsl(
void mul_transpose_mat_add_ldiag_UNIQUEID(in float mat[nrow*ncol], float lambda, inout float omat[ncol*ncol]) {
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
			else {
				omat[j*ncol + i] += lambda * entry;
			}
		}
	}
}
)glsl";

		std::string uniqueid = mul_transpose_mat_add_ldiag_uniqueid(nrow, ncol, single_precission);

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
			"mul_transpose_mat_add_ldiag_" + uniqueid,
			{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}
	

	export std::string mat_set_zero_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mat_set_zero(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mat_set_zero_UNIQUEID(inout float mat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		mat[i] = 0;
	}
}
)glsl";

		std::string uniqueid = mat_set_zero_uniqueid(nrow, ncol, single_precission);

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


	export std::string mat_set_ones_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mat_set_ones(ui16 nrow, ui16 ncol, bool single_precission) {
		static const std::string code = // compute shader
R"glsl(
void mat_set_ones_UNIQUEID(inout float mat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		mat[i] = 1;
	}
}
)glsl";

		std::string uniqueid = mat_set_ones_uniqueid(nrow, ncol, single_precission);

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


	export std::string transpose_square_i_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function transpose_square_i(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void transpose_square_i_UNIQUEID(inout float mat[ndim*ndim]) {
	for (int i = 1; i < ndim; ++i) {
		for (int j = 0; j < i; ++j) {
			swap_SWAPID(mat[i*ndim + j], mat[j*n + i]);
		}
	}
}
)glsl";

		std::string uniqueid = transpose_square_i_uniqueid(ndim, single_precission);

		std::function<std::string()> code_func = [ndim, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "SWAPID", swap_uniqueid(single_precission));
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


	export std::string mul_diag_vec_square_i_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_diag_vec_square_i(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_diag_vec_square_i_UNIQUEID(in float mat[ndim*ndim], inout float vec[ndim]) {
	for (int k = 0; k < ndim; ++k) {
		vec[k] *= mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = mul_diag_vec_square_i_uniqueid(ndim, single_precission);

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


	export std::string mul_inv_diag_vec_square_i_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_inv_diag_vec_square_i(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_inv_diag_vec_square_i_UNIQUEID(in float mat[ndim*ndim], inout float vec[ndim]) {
	for (int k = 0; k < ndim; ++k) {
		vec[k] /= mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = mul_inv_diag_vec_square_i_uniqueid(ndim, single_precission);

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


	export std::string mul_mat_mat_square_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_mat_mat_square(ui16 ndim, bool single_precission)
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

		std::string uniqueid = mul_mat_mat_square_uniqueid(ndim, single_precission);

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
	

	export std::string mul_unit_lower_upper_square_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_unit_lower_upper_square(ui16 ndim, bool single_precission)
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

		std::string uniqueid = mul_unit_lower_upper_square_uniqueid(ndim, single_precission);

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


	export std::string mul_mat_vec_square_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export ::glsl::Function mul_mat_vec_square(ui16 ndim, bool single_precission)
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

		std::string uniqueid = mul_mat_vec_square_uniqueid(ndim, single_precission);

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
