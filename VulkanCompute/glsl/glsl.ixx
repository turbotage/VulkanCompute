module;

#include <kompute/Kompute.hpp>

export module glsl;

import <string>;
import <optional>;
import <vector>;
import <functional>;
import <unordered_set>;
import <set>;
import <type_traits>;
import <memory>;

import util;
import vc;
import variable;

using namespace vc;

namespace glsl {

	export constexpr auto UNIQUE_ID = "UNIQUEID";

	export std::vector<ui32> compileSource(const std::string& source, bool optimize = true);

}

