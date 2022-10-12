module;

export module qmri;

import <string>;
import <memory>;
import <vector>;
import <functional>;
import <optional>;

import vc;
import util;

import glsl;
import function;
import lsq;

namespace glsl {
namespace qmri {

	using vecptrfunc = std::vector<std::shared_ptr<Function>>;
	using refvecptrfunc = vc::refw<std::vector<std::shared_ptr<Function>>>;

	export std::string ivim_guess_uniqueid(vc::ui16 ndata, bool single_precision)
	{
		return std::to_string(ndata) + "_" + (single_precision ? "S" : "D");
	}

	// S = S0 * (f * np.exp(-b * D_star) + (1 - f) * np.exp(-b * D))
	export std::shared_ptr<glsl::Function> ivim_guess(vc::ui16 ndata, bool single_precision)
	{
		static const std::string code = // compute shader
R"glsl(
void ivim_guess_UNIQUEID(inout float params[4], in float bvals[ndata], in float data[ndata], int lower_b, int upper_b)
{
	float log_data[ndata];
	for (int i = 0; i < ndata; ++i) {
		log_data[i] = log(data[i]);
	}
	
	float lin_param1[2]; // S0_prime, D
	lsq_linear2_upper_LL2UID(bvals, log_data, upper_b, lin_param1);
	lin_param1[0] = exp(lin_param1[0]);

	float lin_param2[2]; // S0, D_star_prime
	lsq_linear2_lower_LL2LID(bvals, log_data, lower_b, lin_param2);
	lin_param2[0] = exp(lin_param2[0]);	

	params[0] = lin_param2[0];
	params[1] = 1.0 - (lin_param1[0] / lin_param2[0]);
	params[2] = -lin_param2[1];
	params[3] = -lin_param1[1];
}
)glsl";

		std::string uniqueid = ivim_guess_uniqueid(ndata, single_precision);

		std::function<std::string()> code_func = [ndata, single_precision, uniqueid]() -> std::string
		{
			std::string temp = code;
			util::replace_all(temp, UNIQUE_ID, uniqueid);
			util::replace_all(temp, "ndata", std::to_string(ndata));
			util::replace_all(temp, "LL2UID", lsq::lsq_linear2_upper_uniqueid(ndata, single_precision));
			util::replace_all(temp, "LL2LID", lsq::lsq_linear2_lower_uniqueid(ndata, single_precision));
			if (!single_precision) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return std::make_shared<Function>(
			"ivim_guess_" + uniqueid,
			std::vector<size_t>{ size_t(ndata), size_t(single_precision) },
			code_func,
			std::make_optional<vecptrfunc>({
				lsq::lsq_linear2_upper(ndata, single_precision),
				lsq::lsq_linear2_lower(ndata, single_precision)
				})
			);
	}


}
}