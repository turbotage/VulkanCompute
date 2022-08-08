module;

#include <string>
#include <vector>
#include <optional>

export module solver_glsl;

import util;
import glsl;
import linalg_glsl;

// INTERFACE
namespace glsl {
namespace linalg {
namespace solver {

	export ::glsl::Function forward_subs(int ndim, bool single_precission = true);

	export ::glsl::Function forward_subs_t(int ndim, bool single_precission = true);

}
}
}



// IMPLEMENTATION
namespace glsl {
namespace linalg {
namespace solver {

	::glsl::Function forward_subs(int ndim, bool single_precission)
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
			util::replace_all(temp, "ndim", std::to_string(ndim));
			if (!single_precission) {
				util::replace_all(temp, "float", "double");
			}
			return temp;
		};

		return ::glsl::Function(
			"forward_subs",
			{ size_t(ndim), size_t(single_precission) },
			code_func,
			std::make_optional<std::vector<Function>>({ ::glsl::linalg::swap(single_precission) })
		);
	}

}
}
}

