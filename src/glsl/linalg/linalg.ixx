 module;

export module linalg;

import <string>;
import <optional>;
import <functional>;
import <memory>;
import <vector>;
import <stdexcept>;

import vc;
import glsl;
import util;

export import variable;
export import function;

export import copy;
export import permute;

using namespace vc;

// IMPLEMENTATION
namespace glsl {
namespace linalg {

	using vecptrfunc = std::vector<std::shared_ptr<Function>>;
	using refvecptrfunc = refw<std::vector<std::shared_ptr<Function>>>;

	export std::string mat_neg_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mat_neg(ui16 nrow, ui16 ncol, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void mag_neg_UNIQUEID(inout float mat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		mat[i] = -mat[i];
	}
}
)glsl";

		std::string uniqueid = mat_neg_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mat_neg_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::FunctionApplier mat_neg(const std::shared_ptr<glsl::MatrixVariable>& mat)
	{
		// type checks and dims
		{
			if (!((mat->getType() == ShaderVariableType::FLOAT) ||
				(mat->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 nrow = mat->getNDim1();
		ui16 ncol = mat->getNDim2();

		bool single_precision = true;
		if (mat->getType() == ShaderVariableType::DOUBLE)
			single_precision = false;

		auto func = mat_neg(nrow, ncol, single_precision);
		auto uniqueid = mat_neg_uniqueid(nrow, ncol, single_precision);

		return FunctionApplier{ func, nullptr, { mat }, uniqueid };
	}


	export std::string vec_neg_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> vec_neg(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void vec_neg_UNIQUEID(inout float vec[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		vec[i] = -vec[i];
	}
}
)glsl";

		std::string uniqueid = vec_neg_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"vec_neg_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::FunctionApplier vec_neg(const std::shared_ptr<glsl::VectorVariable>& vec)
	{
		// type checks and dims
		{
			if (!((vec->getType() == ShaderVariableType::FLOAT) ||
				(vec->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 ndim = vec->getNDim();

		bool single_precision = true;
		if (vec->getType() == ShaderVariableType::DOUBLE)
			single_precision = false;

		auto func = vec_neg(ndim, single_precision);
		auto uniqueid = vec_neg_uniqueid(ndim, single_precision);

		return FunctionApplier{ func, nullptr, { vec }, uniqueid };
	}


	export std::string inner_prod_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> inner_prod(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
float inner_prod_UNIQUEID(in float v1[ndim], in float v2[ndim]) {
	float ret = 0;
	for (int i = 0; i < ndim; ++i) {
		ret += v1[i] * v2[i];
	}
	return ret;
}
)glsl";

		std::string uniqueid = inner_prod_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"inner_prod_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string weighted_inner_prod_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> weighted_inner_prod(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
float weighted_inner_prod_UNIQUEID(in float mat[ndim*ndim], in float v1[ndim], in float v2[ndim]) {
	float ret = 0;
	for (int i = 0; i < ndim; ++i) {
		float temp = 0;
		for (int j = 0; j < ndim; ++j) {
			temp += mat[i*ndim + j] * v2[j];
		}
		ret += v1[i] * temp;
	}
	return ret;
}
)glsl";

		std::string uniqueid = weighted_inner_prod_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"weighted_inner_prod_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string diag_weighted_inner_prod_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> diag_weighted_inner_prod(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
float diag_weighted_inner_prod_UNIQUEID(in float diag[ndim], in float v1[ndim], in float v2[ndim]) {
	float ret = 0;
	for (int i = 0; i < ndim; ++i) {
		ret += v1[i] * diag[i] * v2[i];
	}
	return ret;
}
)glsl";

		std::string uniqueid = diag_weighted_inner_prod_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"diag_weighted_inner_prod_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
			);
	}



	export std::string weighted_vec_norm_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> weighted_vec_norm(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
float weighted_inner_prod_UNIQUEID(in float mat[ndim*ndim], in float vec[ndim]) {
	float ret = 0;
	for (int i = 0; i < ndim; ++i) {
		float temp = 0;
		for (int j = 0; j < ndim; ++j) {
			temp += mat[i*ndim + j] * vec[j];
		}
		ret += vec[i] * temp;
	}
	return sqrt(ret);
}
)glsl";

		std::string uniqueid = weighted_vec_norm_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"weighted_vec_norm_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string diag_weighted_vec_norm_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> diag_weighted_vec_norm(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
float diag_weighted_inner_prod_UNIQUEID(in float diag[ndim], in float vec[ndim]) {
	float ret = 0;
	for (int i = 0; i < ndim; ++i) {
		ret += vec[i] * diag[i] * vec[i];
	}
	return sqrt(ret);
}
)glsl";

		std::string uniqueid = diag_weighted_vec_norm_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"diag_weighted_vec_norm_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
			);
	}


	export std::string weighted_vec_norm2_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> weighted_vec_norm2(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
float weighted_vec_norm2_UNIQUEID(in float mat[ndim*ndim], in float vec[ndim]) {
	float ret = 0;
	for (int i = 0; i < ndim; ++i) {
		float temp = 0;
		for (int j = 0; j < ndim; ++j) {
			temp += mat[i*ndim + j] * vec[j];
		}
		ret += vec[i] * temp;
	}
	return ret;
}
)glsl";

		std::string uniqueid = weighted_vec_norm2_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"weighted_vec_norm2_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string diag_weighted_vec_norm2_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> diag_weighted_vec_norm2(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
			R"glsl(
float diag_weighted_vec_norm2_UNIQUEID(in float diag[ndim], in float vec[ndim]) {
	float ret = 0;
	for (int i = 0; i < ndim; ++i) {
		ret += vec[i] * diag[i] * vec[i];
	}
	return ret;
}
)glsl";

		std::string uniqueid = diag_weighted_vec_norm2_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"diag_weighted_vec_norm2_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
			);
	}


	export std::string vec_norm_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> vec_norm(ui16 ndim, bool single_precision)
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

		std::string uniqueid = vec_norm_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"vec_norm_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string vec_norm2_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> vec_norm2(ui16 ndim, bool single_precision)
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

		std::string uniqueid = vec_norm2_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"vec_norm2_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_mat_mat_uniqueid(ui16 lnrow, ui16 mid_dim, ui16 rncol, bool single_precision)
	{
		return std::to_string(lnrow) + "_" + std::to_string(mid_dim) + "_" + std::to_string(rncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_mat_mat(ui16 lnrow, ui16 mid_dim, ui16 rncol, bool single_precision)
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

		std::string uniqueid = mul_mat_mat_uniqueid(lnrow, mid_dim, rncol, single_precision);

		std::function<std::string()> code_func = [lnrow, mid_dim, rncol, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "lnrow", std::to_string(lnrow));
			util::replace_all(temp, "mid_dim", std::to_string(mid_dim));
			util::replace_all(temp, "rncol", std::to_string(rncol));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<::glsl::Function>(
			"mul_mat_mat_" + uniqueid,
			std::vector<size_t>{ size_t(lnrow), size_t(mid_dim), size_t(rncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}

	
	export std::string mul_mat_vec_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}
	
	export std::shared_ptr<::glsl::Function> mul_mat_vec(ui16 nrow, ui16 ncol, bool single_precision)
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

		std::string uniqueid = mul_mat_vec_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_mat_vec_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_transpose_vec_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_transpose_vec(ui16 nrow, ui16 ncol, bool single_precision)
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

		std::string uniqueid = mul_transpose_vec_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_transpose_vec_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::FunctionApplier mul_transpose_vec(
		const std::shared_ptr<glsl::MatrixVariable>& mat, const std::shared_ptr<glsl::VectorVariable>& vec,
		const std::shared_ptr<glsl::VectorVariable>& out)
	{
		// type checks and dims
		{
			if (mat->getNDim2() != out->getNDim()) {
				throw std::runtime_error("mat dim2 must equal out dim");
			}
			if (mat->getNDim1() != vec->getNDim()) {
				throw std::runtime_error("mat dim1 must equal vec dim");
			}

			if (!((ui16)mat->getType() &
				(ui16)vec->getType() &
				(ui16)out->getType()))
			{
				throw std::runtime_error("All inputs must have same type");
			}
			if (!((mat->getType() == ShaderVariableType::FLOAT) ||
				(mat->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 nrow = mat->getNDim1();
		ui16 ncol = mat->getNDim2();

		bool single_precision = true;
		if (mat->getType() == ShaderVariableType::DOUBLE)
			single_precision = false;

		auto func = mul_transpose_vec(nrow, ncol, single_precision);
		auto uniqueid = mul_transpose_vec_uniqueid(nrow, ncol, single_precision);

		return FunctionApplier{ func, nullptr, {mat, vec, out}, uniqueid };
	}


	export std::string mul_transpose_diag_vec_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_transpose_diag_vec(ui16 nrow, ui16 ncol, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_transpose_diag_vec_UNIQUEID(in float mat[nrow*ncol], in float diag[nrow], in float vec[nrow], out float ovec[ncol]) {
	for (int i = 0; i < ncol; ++i) {
		ovec[i] = 0;
		for (int j = 0; j < nrow; ++j) {
			ovec[i] += mat[j*ncol + i] * diag[j] * vec[j];
		}
	}
}
)glsl";

		std::string uniqueid = mul_transpose_diag_vec_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_transpose_vec_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
			);
	}

	export ::glsl::FunctionApplier mul_transpose_diag_vec(
		const std::shared_ptr<glsl::MatrixVariable>& mat, 
		const std::shared_ptr<glsl::VectorVariable>& diag,
		const std::shared_ptr<glsl::VectorVariable>& vec,
		const std::shared_ptr<glsl::VectorVariable>& out)
	{
		// type checks and dims
		{
			if (mat->getNDim2() != out->getNDim()) {
				throw std::runtime_error("mat dim2 must equal out dim");
			}
			if (mat->getNDim1() != vec->getNDim()) {
				throw std::runtime_error("mat dim1 must equal vec dim");
			}
			if (mat->getNDim1() != diag->getNDim()) {
				throw std::runtime_error("mat dim1 must equal diag dim");
			}

			if (!((ui16)mat->getType() &
				(ui16)diag->getType() &
				(ui16)vec->getType() &
				(ui16)out->getType()))
			{
				throw std::runtime_error("All inputs must have same type");
			}
			if (!((mat->getType() == ShaderVariableType::FLOAT) ||
				(mat->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 nrow = mat->getNDim1();
		ui16 ncol = mat->getNDim2();

		bool single_precision = true;
		if (mat->getType() == ShaderVariableType::DOUBLE)
			single_precision = false;

		auto func = mul_transpose_vec(nrow, ncol, single_precision);
		auto uniqueid = mul_transpose_vec_uniqueid(nrow, ncol, single_precision);

		return FunctionApplier{ func, nullptr, {mat, diag, vec, out}, uniqueid };
	}


	export std::string add_mat_mat_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> add_mat_mat(ui16 nrow, ui16 ncol, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void add_mat_mat_UNIQUEID(in float lmat[nrow*ncol], in float rmat[nrow*ncol], out float omat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[i] = lmat[i] + rmat[i];
	}
}
)glsl";

		std::string uniqueid = add_mat_mat_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"add_mat_mat_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string add_vec_vec_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<Function> add_vec_vec(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void add_vec_vec_UNIQUEID(in float lvec[ndim], in float rvec[ndim], out float ovec[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		ovec[i] = lvec[i] + rvec[i];
	}
}
)glsl";

		std::string uniqueid = add_vec_vec_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"add_vec_vec_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}

	export FunctionApplier add_vec_vec(const std::shared_ptr<VectorVariable>& lvec,
		const std::shared_ptr<VectorVariable>& rvec, const std::shared_ptr<VectorVariable>& ovec)
	{
		// types and dimensions
		{
			if (lvec->getNDim() != rvec->getNDim()) {
				throw std::runtime_error("lvec dim and rvec dim1 must agree");
			}
			if (rvec->getNDim() != ovec->getNDim()) {
				throw std::runtime_error("rvec dim and ovec dim1 must agree");
			}

			if (!((ui16)lvec->getType() &
				(ui16)rvec->getType() &
				(ui16)ovec->getType()))
			{
				throw std::runtime_error("All inputs must have same type");
			}
			if (!((lvec->getType() == ShaderVariableType::FLOAT) ||
				(lvec->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 ndim = lvec->getNDim();

		bool single_precision = true;
		if (lvec->getType() == ShaderVariableType::DOUBLE)
			single_precision = false;

		auto func = add_vec_vec(ndim, single_precision);

		auto uniqueid = add_vec_vec_uniqueid(ndim, single_precision);

		return FunctionApplier{ func, nullptr,
			{lvec, rvec, ovec }, uniqueid };

	}


	export std::string add_mat_lmat_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> add_mat_lmat(ui16 nrow, ui16 ncol, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void add_mat_lmat_UNIQUEID(in float lmat[nrow*ncol], in float rmat[nrow*ncol], float lambda, out float omat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[i] = lmat[i] + lambda * rmat[i];
	}
}
)glsl";

		std::string uniqueid = add_mat_lmat_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"add_mat_lmat_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string add_mat_mat_ldiag_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<Function> add_mat_mat_ldiag(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void add_mat_mat_ldiag_UNIQUEID(inout float mat[ndim*ndim], float lambda, inout float lmat[ndim*ndim]) {
	float entry1;	
	float entry2;
	for (int i = 0; i < ndim; ++i) {
		for (int j = 0; j < ndim; ++j) {
			entry1 = mat[i*ndim + j];
			entry2 = lmat[i*ndim + j];
			mat[i*ndim + j] += entry2;
			lmat[i*ndim + j] += entry1;
			if (i == j) {
				lmat[i*ndim+j] += lambda * entry2;
			}
		}
	}
}
)glsl";

		std::string uniqueid = add_mat_mat_ldiag_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"add_mat_mat_ldiag_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
			);
	}

	export ::glsl::FunctionApplier add_mat_mat_ldiag(const std::shared_ptr<glsl::MatrixVariable>& mat1,
		const std::shared_ptr<glsl::SingleVariable>& lambda, const std::shared_ptr<glsl::MatrixVariable>& mat2)
	{
		// type checks and dims
		{
			if (mat1->getNDim1() != mat2->getNDim1()) {
				throw std::runtime_error("lmat dim1 must equal rmat dim1");
			}
			if (mat1->getNDim2() != mat2->getNDim2()) {
				throw std::runtime_error("lmat dim1 must equal rmat dim2");
			}
			if (mat1->getNDim1() != mat1->getNDim2()) {
				throw std::runtime_error("lmat must be square");
			}

			if (!((ui16)mat1->getType() &
				(ui16)mat2->getType()))
			{
				throw std::runtime_error("All inputs must have same type");
			}
			if (!((mat1->getType() == ShaderVariableType::FLOAT) ||
				(mat1->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 ndim = mat1->getNDim1();

		bool single_precision = true;
		if (mat2->getType() == ShaderVariableType::DOUBLE)
			single_precision = false;

		auto func = add_mat_mat_ldiag(ndim, single_precision);
		auto uniqueid = add_mat_mat_ldiag_uniqueid(ndim, single_precision);

		return FunctionApplier{ func, nullptr, {mat1, lambda, mat2}, uniqueid };
	}


	export std::string sub_mat_mat_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> sub_mat_mat(ui16 nrow, ui16 ncol, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void sub_mat_mat_UNIQUEID(in float lmat[nrow*ncol], in float rmat[nrow*ncol], out float omat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[i] = lmat[i] - rmat[i];
	}
}
)glsl";

		std::string uniqueid = sub_mat_mat_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"sub_mat_lmat_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string sub_mat_lmat_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> sub_mat_lmat(ui16 nrow, ui16 ncol, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void sub_mat_lmat_UNIQUEID(in float lmat[nrow*ncol], in float rmat[nrow*ncol], float lambda, out float omat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[i] = lmat[i] - lambda*rmat[i];
	}
}
)glsl";

		std::string uniqueid = sub_mat_lmat_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"sub_mat_lmat_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string mat_add_ldiag_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mat_add_ldiag(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void mat_add_ldiag_UNIQUEID(inout float mat[ndim*ndim], float lambda) {
	for (int i = 0; i < ndim; ++i) {
		mat[i*ndim + i] += lambda * mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = mat_add_ldiag_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mat_add_ldiag_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::FunctionApplier mat_add_ldiag(const std::shared_ptr<glsl::MatrixVariable>& mat,
		const std::shared_ptr<glsl::SingleVariable>& lambda)
	{
		// type and dimension checks
		{
			if (!mat->isSquare()) {
				throw std::runtime_error("mat isn't square");
			}

			if (!((ui16)mat->getType() &
				(ui16)lambda->getType()))
			{
				throw std::runtime_error("All inputs must have same type");
			}
			if (!((mat->getType() == ShaderVariableType::FLOAT) ||
				(mat->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 ndim = mat->getNDim1();

		bool single_precision = true;
		if (mat->getType() == ShaderVariableType::DOUBLE)
			single_precision = false;

		auto func = mat_add_ldiag(ndim, single_precision);

		auto uniqueid = mat_add_ldiag_uniqueid(ndim, single_precision);

		return FunctionApplier{ func, nullptr, {mat, lambda}, uniqueid };
	}


	export std::string mat_add_ldiag_out_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mat_add_ldiag_out(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void mat_add_ldiag_out_UNIQUEID(in float mat[ndim*ndim], float lambda, out float omat[ndim*ndim]) {
	for (int i = 0; i < ndim*ndim; ++i) {
		omat[i] = mat[i];
	}
	for (int i = 0; i < ndim; ++i) {
		omat[i*ndim + i] += lambda * omat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = mat_add_ldiag_out_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mat_add_ldiag_out_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::FunctionApplier mat_add_ldiag_out(
		const std::shared_ptr<MatrixVariable>& in, const std::shared_ptr<SingleVariable>& lambda,
		const std::shared_ptr<MatrixVariable>& out)
	{
		// type and dimension checks
		{
			if (in->getNDim1() != out->getNDim1()) {
				throw std::runtime_error("in dim1 must equal out dim1");
			}
			if (out->getNDim2() != out->getNDim2()) {
				throw std::runtime_error("in dim2 must equal out dim2");
			}
			if (in->getNDim1() != in->getNDim2()) {
				throw std::runtime_error("matrices must be square");
			}

			if (!((ui16)in->getType() &
				(ui16)out->getType()))
			{
				throw std::runtime_error("All inputs must have same type");
			}
			if (!((in->getType() == ShaderVariableType::FLOAT) ||
				(in->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 ndim = in->getNDim1();

		bool single_precision = true;
		if (in->getType() == ShaderVariableType::DOUBLE)
			single_precision = false;
		
		auto func = mat_add_ldiag_out(ndim, single_precision);

		auto uniqueid = mat_add_ldiag_out_uniqueid(ndim, single_precision);

		return FunctionApplier{ func, nullptr, {in, lambda, out}, uniqueid };
	}


	export std::string mul_mat_transpose_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_mat_transpose(ui16 nrow, ui16 ncol, bool single_precision)
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

		std::string uniqueid = mul_mat_transpose_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_mat_transpose_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_mat_transpose_ldiag_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_mat_transpose_ldiag(ui16 nrow, ui16 ncol, bool single_precision)
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

		std::string uniqueid = mul_mat_transpose_ldiag_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_mat_transpose_ldiag_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_mat_transpose_add_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_mat_transpose_add(ui16 nrow, ui16 ncol, bool single_precision)
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

		std::string uniqueid = mul_mat_transpose_add_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_mat_transpose_add_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_mat_transpose_add_ldiag_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_mat_transpose_add_ldiag(ui16 nrow, ui16 ncol, bool single_precision)
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

		std::string uniqueid = mul_mat_transpose_add_ldiag_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_mat_transpose_add_ldiag_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}

	
	export std::string mul_transpose_mat_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_transpose_mat(ui16 nrow, ui16 ncol, bool single_precision)
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

		std::string uniqueid = mul_transpose_mat_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_transpose_mat_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}

	export ::glsl::FunctionApplier mul_transpose_mat(
		const std::shared_ptr<glsl::MatrixVariable>& in, const std::shared_ptr<glsl::MatrixVariable>& out)
	{
		// type and dimension checks
		{
			if (in->getNDim2() != out->getNDim1()) {
				throw std::runtime_error("Input matrix 2nd dim must be equal to out matrix dim1");
			}
			if (out->getNDim1() != out->getNDim2() ) {
				throw std::runtime_error("out dim1 must equal out dim2");
			}

			if (!((ui16)in->getType() &
				(ui16)out->getType()))
			{
				throw std::runtime_error("All inputs must have same type");
			}
			if (!((in->getType() == ShaderVariableType::FLOAT) ||
				(in->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 nrow = in->getNDim1();
		ui16 ncol = in->getNDim2();

		bool single_precision = true;
		if (in->getType() == ShaderVariableType::DOUBLE)
			single_precision = false;

		auto func = mul_transpose_mat(nrow, ncol, single_precision);

		auto uniqueid = mul_transpose_mat_uniqueid(nrow, ncol, single_precision);

		return FunctionApplier{ func, nullptr, { in, out }, uniqueid };
	}


	export std::string mul_transpose_diag_mat_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_transpose_diag_mat(ui16 nrow, ui16 ncol, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_transpose_diag_mat_UNIQUEID(in float mat[nrow*ncol], in float diag[nrow], out float omat[ncol*ncol]) {
	float entry;
	for (int i = 0; i < ncol; ++i) {
		for (int j = 0; j <= i; ++j) {
			entry = 0.0;
			for (int k = 0; k < nrow; ++k) {
				entry += mat[k*ncol + i] * diag[k] * mat[k*ncol + j];
			}
			omat[i*ncol + j] = entry;
			if (i != j) {
				omat[j*ncol + i] = entry;
			}
		}
	}
}
)glsl";

		std::string uniqueid = mul_transpose_mat_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_transpose_diag_mat_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
			);
	}

	export ::glsl::FunctionApplier mul_transpose_diag_mat(
		const std::shared_ptr<glsl::MatrixVariable>& in, 
		const std::shared_ptr<glsl::VectorVariable>& diag,
		const std::shared_ptr<glsl::MatrixVariable>& out)
	{
		// type and dimension checks
		{
			if (in->getNDim2() != out->getNDim1()) {
				throw std::runtime_error("Input matrix 2nd dim must be equal to out matrix dim1");
			}
			if (out->getNDim1() != out->getNDim2()) {
				throw std::runtime_error("out dim1 must equal out dim2");
			}
			if (in->getNDim1() != diag->getNDim()) {
				throw std::runtime_error("in dim1 must equal diag dim");
			}

			if (!((ui16)in->getType() &
				(ui16)diag->getType() &
				(ui16)out->getType()))
			{
				throw std::runtime_error("All inputs must have same type");
			}
			if (!((in->getType() == ShaderVariableType::FLOAT) ||
				(in->getType() == ShaderVariableType::DOUBLE))) {
				throw std::runtime_error("Inputs must have float or double type");
			}
		}

		ui16 nrow = in->getNDim1();
		ui16 ncol = in->getNDim2();

		bool single_precision = true;
		if (in->getType() == ShaderVariableType::DOUBLE)
			single_precision = false;

		auto func = mul_transpose_diag_mat(nrow, ncol, single_precision);

		auto uniqueid = mul_transpose_diag_mat_uniqueid(nrow, ncol, single_precision);

		return FunctionApplier{ func, nullptr, { in, diag, out }, uniqueid };
	}


	export std::string mul_transpose_mat_ldiag_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_transpose_mat_ldiag(ui16 nrow, ui16 ncol, bool single_precision)
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
				omat[j*ncol + i] += lambda * entry;
			}
		}
	}
}
)glsl";

		std::string uniqueid = mul_transpose_mat_ldiag_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_transpose_mat_ldiag_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_transpose_mat_add_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_transpose_mat_add(ui16 nrow, ui16 ncol, bool single_precision)
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

		std::string uniqueid = mul_transpose_mat_add_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_transpose_mat_add_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_transpose_mat_add_ldiag_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_transpose_mat_add_ldiag(ui16 nrow, ui16 ncol, bool single_precision)
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

		std::string uniqueid = mul_transpose_mat_add_ldiag_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_transpose_mat_add_ldiag_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}
	

	export std::string mat_set_zero_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mat_set_zero(ui16 nrow, ui16 ncol, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void mat_set_zero_UNIQUEID(inout float mat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		mat[i] = 0;
	}
}
)glsl";

		std::string uniqueid = mat_set_zero_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mat_set_zero_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string mat_set_ones_uniqueid(ui16 nrow, ui16 ncol, bool single_precision)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mat_set_ones(ui16 nrow, ui16 ncol, bool single_precision) {
		static const std::string code = // compute shader
R"glsl(
void mat_set_ones_UNIQUEID(inout float mat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		mat[i] = 1;
	}
}
)glsl";

		std::string uniqueid = mat_set_ones_uniqueid(nrow, ncol, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mat_set_ones_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string transpose_square_i_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> transpose_square_i(ui16 ndim, bool single_precision)
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

		std::string uniqueid = transpose_square_i_uniqueid(ndim, single_precision);

		std::function<std::string()> code_func = [ndim, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, "SWAPID", swap_uniqueid(single_precision));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
				util::replace_all(temp, "ndim", std::to_string(ndim));
			}
			return temp;
		};

		return std::make_shared<::glsl::Function>(
			"transpose_square_i_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::make_optional<vecptrfunc>({ swap(single_precision) })
		);

	}


	export std::string mul_diag_vec_square_i_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_diag_vec_square_i(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_diag_vec_square_i_UNIQUEID(in float mat[ndim*ndim], inout float vec[ndim]) {
	for (int k = 0; k < ndim; ++k) {
		vec[k] *= mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = mul_diag_vec_square_i_uniqueid(ndim, single_precision);

		std::function<std::string()> code_func = [ndim, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			if (!single_precision) {
				util::replace_all(temp, UNIQUE_ID, uniqueid);
				util::replace_all(temp, "float", "double");
				util::replace_all(temp, "ndim", std::to_string(ndim));
			}
			return temp;
		};

		return std::make_shared<::glsl::Function>(
			"mul_diag_vec_square_i_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_inv_diag_vec_square_i_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_inv_diag_vec_square_i(ui16 ndim, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void mul_inv_diag_vec_square_i_UNIQUEID(in float mat[ndim*ndim], inout float vec[ndim]) {
	for (int k = 0; k < ndim; ++k) {
		vec[k] /= mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = mul_inv_diag_vec_square_i_uniqueid(ndim, single_precision);

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
		
		return std::make_shared<::glsl::Function>(
			"mul_inv_diag_vec_square_i_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_mat_mat_square_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_mat_mat_square(ui16 ndim, bool single_precision)
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

		std::string uniqueid = mul_mat_mat_square_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_mat_mat_square_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}
	

	export std::string mul_unit_lower_upper_square_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_unit_lower_upper_square(ui16 ndim, bool single_precision)
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

		std::string uniqueid = mul_unit_lower_upper_square_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_unit_lower_upper_square_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}


	export std::string mul_mat_vec_square_uniqueid(ui16 ndim, bool single_precision)
	{
		return std::to_string(ndim) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> mul_mat_vec_square(ui16 ndim, bool single_precision)
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

		std::string uniqueid = mul_mat_vec_square_uniqueid(ndim, single_precision);

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

		return std::make_shared<::glsl::Function>(
			"mul_mat_vec_square_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precision) },
			code_func,
			std::nullopt
		);
	}

}
}
