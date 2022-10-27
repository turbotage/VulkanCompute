module;

export module glsl;

import <string>;
import <vector>;
import <optional>;

import util;
import vc;
import variable;

using namespace vc;

namespace glsl {

	export constexpr auto UNIQUE_ID = "UNIQUEID";

	export enum class OptimizationType {
		NO_OPTIMIZATION = 1,
		OPTIMIZE_FOR_SPEED = 2,
		OPTIMIZE_FOR_SIZE = 4,
		REMAP = 8
	};

	export std::vector<ui32> compileSource(const std::string& source, OptimizationType opt_type = 
		static_cast<OptimizationType>
			(
			static_cast<int>(OptimizationType::OPTIMIZE_FOR_SPEED) | 
			static_cast<int>(OptimizationType::REMAP)
			)
	);

	export std::string compileSourceToFile(const std::string& source, OptimizationType opt_type = 
		static_cast<OptimizationType>
			(
			static_cast<int>(OptimizationType::OPTIMIZE_FOR_SPEED) |
			static_cast<int>(OptimizationType::REMAP)
			)
	);

	export std::optional<std::string> decompileSPIRV(bool return_string = false);

}

