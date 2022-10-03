module;

#include <kompute/Kompute.hpp>

module glsl;

import <memory>;
import <string>;

using namespace glsl;


import linalg;

import vc;
using namespace vc;

namespace {
	// most shaders are smaller than 30 kB
	constexpr auto DEFAULT_SHADER_SIZE = 30000;
}

std::vector<uint32_t> glsl::compileSource(const std::string& source, bool optimize)
{
	std::ofstream fileOut("tmpshader.comp");
	fileOut << source;
	fileOut.close();
	std::ifstream fileStream;
	if (system(std::string("glslangValidator -V tmpshader.comp -o tmpshader.comp.spv").c_str()))
	{
		throw std::runtime_error("Error running glslangValidator command");
	}

	if (optimize) {
		if (system(std::string("spirv-opt -O tmpshader.comp.spv -o tmpshader.comp.spv").c_str()))
		{
			throw std::runtime_error("Error running spirv-opt command");
		}

		try {
			system(std::string("mkdir tmp").c_str());
		}
		catch (...) {}

		if (system(std::string("spirv-remap --do-everything --input tmpshader.comp.spv --output tmp").c_str())) 
		{
			throw std::runtime_error("Error running spirv-opt command");
		}

		fileStream = std::ifstream("tmp/tmpshader.comp.spv", std::ios::binary);
	}
	else {
		fileStream = std::ifstream("tmpshader.comp.spv", std::ios::binary);
	}

	std::vector<char> buffer;
	buffer.insert(
		buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
	return { (uint32_t*)buffer.data(),
			 (uint32_t*)(buffer.data() + buffer.size()) };
}

