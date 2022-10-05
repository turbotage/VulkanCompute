module;

export module copy;

import <string>;
import <optional>;
import <memory>;
import <vector>;
import <functional>;
import <initializer_list>;

import vc;
import glsl;
import util;

export import function;

using namespace vc;

// IMPLEMENTATION
namespace glsl {
namespace linalg {

	export std::string swap_uniqueid(bool single_precission)
	{
		return single_precission ? "S" : "D";
	}

	export std::shared_ptr<::glsl::Function> swap(bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void swap_UNIQUEID(inout float v1, inout float v2) {
	float temp = v1;
	v1 = v2;
	v2 = temp;
}
)glsl";

		auto uniqueid = swap_uniqueid(single_precission);

		std::function<std::string()> code_func = [single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"swap_" + uniqueid,
			std::vector<size_t>{ size_t(single_precission) },
			code_func,
			std::nullopt);
	}


	export std::string copy_mat_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> copy_mat(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_mat_UNIQUEID(in float imat[nrow*ncol], out float omat[nrow*ncol]) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[i] = imat[i];
	}
}
)glsl";

		std::string uniqueid = copy_mat_uniqueid(nrow, ncol, single_precission);

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
			std::string("copy_mat_" + uniqueid),
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}


	export std::string copy_mat_ostart_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> copy_mat_ostart(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_mat_ostart_UNIQUEID(in float imat[nrow*ncol], out float omat[nrow*ncol], uint start_index) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[start_index + i] = imat[i];
	}
}
)glsl";

		std::string uniqueid = copy_mat_ostart_uniqueid(nrow, ncol, single_precission);

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
			"copy_mat_ostart_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}


	export std::string copy_mat_ostarted_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> copy_mat_ostarted(ui16 nrow, ui16 ncol, bool single_precission)
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

		std::string uniqueid = copy_mat_ostarted_uniqueid(nrow, ncol, single_precission);

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
			"copy_mat_ostarted_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}


	export std::string copy_mat_istart_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> copy_mat_istart(ui16 nrow, ui16 ncol, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_mat_istart_UNIQUEID(in float imat[nrow*ncol], out float omat[nrow*ncol], uint start_index) {
	for (int i = 0; i < nrow*ncol; ++i) {
		omat[i] = imat[start_index + i];
	}
}
)glsl";

		std::string uniqueid = copy_mat_istart_uniqueid(nrow, ncol, single_precission);

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
			"copy_mat_istart_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}


	export std::string copy_mat_istarted_uniqueid(ui16 nrow, ui16 ncol, bool single_precission)
	{
		return std::to_string(nrow) + "_" + std::to_string(ncol) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> copy_mat_istarted(ui16 nrow, ui16 ncol, bool single_precission)
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

		std::string uniqueid = copy_mat_istarted_uniqueid(nrow, ncol, single_precission);

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
			"copy_mat_istarted_" + uniqueid,
			std::vector<size_t>{ size_t(nrow), size_t(ncol), size_t(single_precission) },
			code_func,
			std::nullopt);
	}


	export std::string copy_vec_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> copy_vec(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_vec_UNIQUEID(in float ivec[ndim], out float ovec[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		ovec[i] = ivec[i];
	}
}
)glsl";

		std::string uniqueid = copy_vec_uniqueid(ndim, single_precission);

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

		return std::make_shared<Function>(
			"copy_vec_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt);
	}


	export std::string copy_vec_ostart_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> copy_vec_ostart(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_vec_ostart_UNIQUEID(in float ivec[ndim], out float ovec[ndim], uint start_index) {
	for (int i = 0; i < ndim; ++i) {
		ovec[start_index + i] = ivec[i];
	}
}
)glsl";

		std::string uniqueid = copy_vec_ostart_uniqueid(ndim, single_precission);

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

		return std::make_shared<Function>(
			"copy_vec_ostart_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt);
	}


	export std::string copy_vec_ostarted_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> copy_vec_ostarted(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_vec_ostarted_UNIQUEID(in float ivec[ndim], out float ovec[ndim]) {
	uint start_index = ndim*gl_GlobalInvocationID.x;
	for (int i = 0; i < ndim; ++i) {
		ovec[start_index + i] = ivec[i];
	}
}
)glsl";

		std::string uniqueid = copy_vec_ostarted_uniqueid(ndim, single_precission);

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

		return std::make_shared<Function>(
			"copy_vec_ostarted_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt);
	}


	export std::string copy_vec_istart_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> copy_vec_istart(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_vec_istart_UNIQUEID(in float ivec[ndim], out float ovec[ndim], uint start_index) {
	for (int i = 0; i < ndim; ++i) {
		ovec[i] = ivec[start_index + i];
	}
}
)glsl";

		std::string uniqueid = copy_vec_istart_uniqueid(ndim, single_precission);

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

		return std::make_shared<Function>(
			"copy_vec_istart_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt);
	}


	export std::string copy_vec_istarted_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> copy_vec_istarted(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void copy_vec_istarted_UNIQUEID(in float ivec[ndim], out float ovec[ndim]) {
	uint start_index = ndim*gl_GlobalInvocationID.x;
	for (int i = 0; i < ndim; ++i) {
		ovec[i] = ivec[start_index + i];
	}
}
)glsl";

		std::string uniqueid = copy_vec_istarted_uniqueid(ndim, single_precission);

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

		return std::make_shared<Function>(
			"copy_vec_istarted_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt);
	}


}
}