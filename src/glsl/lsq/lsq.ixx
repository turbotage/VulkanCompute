module;

export module lsq;

import <string>;
import <memory>;
import <vector>;
import <optional>;
import <functional>;

import vc;
import util;

import glsl;
import function;

import symm;

namespace glsl {
namespace lsq {

	using vecptrfunc = std::vector<std::shared_ptr<Function>>;
	using refvecptrfunc = vc::refw<std::vector<std::shared_ptr<Function>>>;


	export std::string lsq_linear2_lower_uniqueid(vc::ui16 ndata, bool single_precision)
	{
		return std::to_string(ndata) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> lsq_linear2_lower(vc::ui16 ndata, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void lsq_linear2_lower_UNIQUEID(in float x[ndata], in float y[ndata], int upper_bound, out float param[2]) {
	float mat[2*2];
	mat[0] = upper_bound;
	mat[1] = 0.0;
	mat[2] = 0.0;
	mat[3] = 0.0;
		
	float rhs[2];
	rhs[0] = 0.0;
	rhs[1] = 0.0;

	for (int i = 0; i < upper_bound; ++i) {
		mat[1] += x[i];
		mat[3] += x[i] * x[i];

		rhs[0] += y[i];
		rhs[1] += x[i] * y[i];
	}
	mat[2] = mat[1];
	
	ldl_LID(mat);

	ldl_solve_LSID(mat, rhs, param);
} 
)glsl";

		std::string uniqueid = lsq_linear2_lower_uniqueid(ndata, single_precision);

		std::function<std::string()> code_func = [ndata, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "LID", linalg::ldl_uniqueid(2, single_precision));
			util::replace_all(temp, "LSID", linalg::ldl_solve_uniqueid(2, single_precision));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"lsq_linear2_lower_" + uniqueid,
			std::vector<size_t>{ size_t(ndata), size_t(single_precision) },
			code_func,
			std::make_optional<vecptrfunc>({
				linalg::ldl(2, single_precision),
				linalg::ldl_solve(2, single_precision)
				})
			);
	}


	export std::string lsq_linear2_upper_uniqueid(vc::ui16 ndata, bool single_precision)
	{
		return std::to_string(ndata) + "_" + (single_precision ? "S" : "D");
	}

	export std::shared_ptr<::glsl::Function> lsq_linear2_upper(vc::ui16 ndata, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void lsq_linear2_upper_UNIQUEID(in float x[ndata], in float y[ndata], int lower_bound, out float param[2]) {
	float mat[2*2];
	mat[0] = ndata - lower_bound;
	mat[1] = 0.0;
	mat[2] = 0.0;
	mat[3] = 0.0;
		
	float rhs[2];
	rhs[0] = 0.0;
	rhs[1] = 0.0;

	for (int i = lower_bound; i < ndata; ++i) {
		mat[1] += x[i];
		mat[3] += x[i] * x[i];

		rhs[0] += y[i];
		rhs[1] += x[i] * y[i];
	}
	mat[2] = mat[1];
	
	ldl_LID(mat);

	ldl_solve_LSID(mat, rhs, param);
} 
)glsl";

		std::string uniqueid = lsq_linear2_lower_uniqueid(ndata, single_precision);

		std::function<std::string()> code_func = [ndata, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "LID", linalg::ldl_uniqueid(2, single_precision));
			util::replace_all(temp, "LSID", linalg::ldl_solve_uniqueid(2, single_precision));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"lsq_linear2_upper_" + uniqueid,
			std::vector<size_t>{ size_t(ndata), size_t(single_precision) },
			code_func,
			std::make_optional<vecptrfunc>({
				linalg::ldl(2, single_precision),
				linalg::ldl_solve(2, single_precision)
				})
			);
	}


}
}
