module;

export module solver;

import <string>;
import <vector>;
import <optional>;
import <memory>;

import vc;
import util;
import glsl;
export import linalg;

using namespace vc;

namespace glsl {
namespace linalg {

	using vecptrfunc = std::vector<std::shared_ptr<Function>>;
	using refvecptrfunc = refw<std::vector<std::shared_ptr<Function>>>;

	// FORWARD SUBSTITUTION (Lx=y)

	export std::string forward_subs_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> forward_subs(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void forward_subs_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= mat[i*ndim + j] * solution[j];
		}
		solution[i] /= mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = forward_subs_uniqueid(ndim, single_precission);

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
			"forward_subs_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string forward_subs_t_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> forward_subs_t(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void forward_subs_t_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= mat[j*ndim + i] * solution[j];
		}
		solution[i] /= mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = forward_subs_t_uniqueid(ndim, single_precission);

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
			"forward_subs_t_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string forward_subs_unit_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> forward_subs_unit(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void forward_subs_unit_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= mat[i*ndim + j] * solution[j];
		}
	}
}
)glsl";

		std::string uniqueid = forward_subs_unit_uniqueid(ndim, single_precission);

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
			"forward_subs_unit_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string forward_subs_unit_t_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> forward_subs_unit_t(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void forward_subs_unit_t_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= mat[j*ndim + i] * solution[j];
		}
	}
}
)glsl";

		std::string uniqueid = forward_subs_unit_t_uniqueid(ndim, single_precission);

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
			"forward_subs_unit_t_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	// with diagonal (LDx=y)

	export std::string forward_subs_diag_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> forward_subs_diag(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void forward_subs_diag_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim] in float diag[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= (mat[i*ndim + j] * diag[j] * solution[j]);
		}
		solution[i] /= (mat[i*ndim + i] * diag[i]);
	}
}
)glsl";

		std::string uniqueid = forward_subs_diag_uniqueid(ndim, single_precission);

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
			"forward_subs_diag_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string forward_subs_t_diag_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> forward_subs_t_diag(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void forward_subs_t_diag_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], in float diag[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= (mat[j*ndim + i] * diag[j] * solution[j]);
		}
		solution[i] /= (mat[i*ndim + i] * diag[i]);
	}
}
)glsl";

		std::string uniqueid = forward_subs_t_diag_uniqueid(ndim, single_precission);

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
			"forward_subs_t_diag_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string forward_subs_unit_diag_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> forward_subs_unit_diag(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void forward_subs_unit_diag_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], in float diag[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= (mat[i*ndim + j] * diag[j] * solution[j]);
		}
		solution[i] /= diag[i];
	}
}
)glsl";

		std::string uniqueid = forward_subs_unit_diag_uniqueid(ndim, single_precission);

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
			"forward_subs_unit_diag_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string forward_subs_unit_t_diag_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> forward_subs_unit_t_diag(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void forward_subs_unit_t_diag_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], in float diag[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= (mat[j*ndim + i] * diag[j] * solution[j]);
		}
		solution[i] /= diag[i];
	}
}
)glsl";

		std::string uniqueid = forward_subs_unit_t_diag_uniqueid(ndim, single_precission);

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
			"forward_subs_unit_t_diag_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	// with diagonal along diagonal of matrix (LDx=y)

	export std::string forward_subs_unit_diaged_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> forward_subs_unit_diaged(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void forward_subs_unit_diaged_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= (mat[i*ndim + j] * mat[j*ndim + j] * solution[j]);
		}
		solution[i] /= mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = forward_subs_unit_diaged_uniqueid(ndim, single_precission);

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
			"forward_subs_unit_diaged_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string forward_subs_unit_t_diaged_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> forward_subs_unit_t_diaged(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void forward_subs_unit_t_diaged_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = 0; i < ndim; ++i) {
		solution[i] = rhs[i];
		for (int j = 0; j < i; ++j) {
			solution[i] -= (mat[j*ndim + i] * mat[j*ndim + j] * solution[j]);
		}
		solution[i] /= mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = forward_subs_unit_t_diaged_uniqueid(ndim, single_precission);

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
			"forward_subs_unit_t_diaged_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	// BACKWARD SUBSTITUTION (Ux=y)

	export std::string backward_subs_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> backward_subs(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void backward_subs_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= mat[i*ndim + j] * solution[j];
		}
		solution[i] /= mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = backward_subs_uniqueid(ndim, single_precission);

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
			"backward_subs_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string backward_subs_t_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> backward_subs_t(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void backward_subs_t_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= mat[j*ndim + i] * solution[j];
		}
		solution[i] /= mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = backward_subs_t_uniqueid(ndim, single_precission);

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
			"backward_subs_t_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string backward_subs_unit_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> backward_subs_unit(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void backward_subs_unit_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= mat[i*ndim + j] * solution[j];
		}
	}
}
)glsl";

		std::string uniqueid = backward_subs_unit_uniqueid(ndim, single_precission);

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
			"backward_subs_unit_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string backward_subs_unit_t_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> backward_subs_unit_t(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void backward_subs_unit_t_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= mat[j*ndim + i] * solution[j];
		}
	}
}
)glsl";

		std::string uniqueid = backward_subs_unit_t_uniqueid(ndim, single_precission);

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
			"backward_subs_unit_t_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	// with diagonal (DUx=y)

	export std::string backward_subs_diag_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> backward_subs_diag(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void backward_subs_diag_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], in float diag[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= (mat[i*ndim + j] * diag[i] * solution[j]);
		}
		solution[i] /= (mat[i*ndim + i] * diag[i]);
	}
}
)glsl";

		std::string uniqueid = backward_subs_diag_uniqueid(ndim, single_precission);

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
			"backward_subs_diag_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string backward_subs_t_diag_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> backward_subs_t_diag(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void backward_subs_t_diag_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], in float diag[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= (mat[j*ndim + i] * diag[i] * solution[j]);
		}
		solution[i] /= (mat[i*ndim + i] * diag[i]);
	}
}
)glsl";

		std::string uniqueid = backward_subs_t_diag_uniqueid(ndim, single_precission);

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
			"backward_subs_t_diag_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string backward_subs_unit_diag_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> backward_subs_unit_diag(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void backward_subs_unit_diag_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], in float diag[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= (mat[i*ndim + j] * diag[i] * solution[j]);
		}
		solution[i] /= diag[i];
	}
}
)glsl";

		std::string uniqueid = backward_subs_unit_diag_uniqueid(ndim, single_precission);

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
			"backward_subs_unit_diag_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string backward_subs_unit_t_diag_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> backward_subs_unit_t_diag(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void backward_subs_unit_t_diag_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], in float diag[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= (mat[j*ndim + i] * diag[i] * solution[j]);
		}
		solution[i] /= diag[i];
	}
}
)glsl";

		std::string uniqueid = backward_subs_unit_t_diag_uniqueid(ndim, single_precission);

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
			"backward_subs_unit_t_diag_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	// with diagonal along diagonal of matrix (DUx=y)

	export std::string backward_subs_unit_diaged_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> backward_subs_unit_diaged(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void backward_subs_unit_diaged_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= (mat[i*ndim + j] * mat[i*ndim + i] * solution[j]);
		}
		solution[i] /= mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = backward_subs_unit_diaged_uniqueid(ndim, single_precission);

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
			"backward_subs_unit_diaged_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}


	export std::string backward_subs_unit_t_diaged_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> backward_subs_unit_t_diaged(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void backward_subs_unit_t_diaged_UNIQUEID(in float mat[ndim*ndim], in float rhs[ndim], out float solution[ndim]) {
	for (int i = ndim - 1; i >= 0; --i) {
		solution[i] = rhs[i];
		for (int j = i + 1; j < ndim; ++j) {
			solution[i] -= (mat[j*ndim + i] * mat[i*ndim + i] * solution[j]);
		}
		solution[i] /= mat[i*ndim + i];
	}
}
)glsl";

		std::string uniqueid = backward_subs_unit_t_diaged_uniqueid(ndim, single_precission);

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
			"backward_subs_unit_t_diaged_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::nullopt
		);
	}

	// LU DECOMPOSITION

	export std::string lu_uniqueid(ui16 ndim, bool single_precission)
	{
		return std::to_string(ndim) + "_" + (single_precission ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> lu(ui16 ndim, bool single_precission)
	{
		static const std::string code = // compute shader
R"glsl(
void lu_UNIQUEID(inout float mat[ndim*ndim], out int pivot[ndim]) {
	float val;
	for (int k = 0; k < ndim - 1; ++k) {
		int max_row_idx;
		float max_row_value;
		max_mag_subrow_MMSID(mat, k, k, max_row_idx, max_row_value);		
		row_interchange_i_RIIID(mat, k, max_row_idx);
		pivot[k] = max_row_idx;
		
		val = mat[k*ndim + k];
		if (val > machine_eps) {
			for (int i = k + 1; i < ndim; ++i) {
				mat[i*ndim + k] /= val; 
			}
			
			for (int i = k + 1; i < ndim; ++i) {
				for (int j = k + 1; j < ndim; ++j) {
					mat[i*ndim + j] -= mat[i*ndim + k]*mat[k*ndim + j];
				}
			}
		}
	}
}
)glsl";

		std::string uniqueid = lu_uniqueid(ndim, single_precission);

		std::function<std::string()> code_func = [ndim, single_precission, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndim", std::to_string(ndim));
			util::replace_all(temp, "MMSID", max_mag_subrow_uniqueid(ndim, ndim, single_precission));
			util::replace_all(temp, "RIIID", row_interchange_i_uniqueid(ndim, ndim, single_precission));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
				util::replace_all(temp, "machine_eps", "1e-15");
			}
			else {
				util::replace_all(temp, "machine_eps", "1e-6");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"lu_" + uniqueid,
			std::vector<size_t>{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::make_optional<vecptrfunc>({ 
				max_mag_subrow(ndim, ndim, single_precission), 
				row_interchange_i(ndim, ndim, single_precission) 
			})
		);
	}

}
}

