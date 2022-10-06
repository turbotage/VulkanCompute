module;

module glsl;

import <string>;
import <vector>;
import <stdexcept>;
import <fstream>;
import <ostream>;
import <optional>;

import vc;

using namespace vc;
using namespace glsl;

namespace {
	static std::string last_shader_name;
}

std::vector<ui32> glsl::compileSource(const std::string& source, OptimizationType opt_type)
{
	int opt_type_int = static_cast<int>(opt_type);

	std::ofstream fileOut("tmpshader.comp");
	fileOut << source;
	fileOut.close();
	std::ifstream fileStream;
	if (system(std::string("glslangValidator -V tmpshader.comp -o tmpshader.comp.spv").c_str()))
	{
		throw std::runtime_error("Error running glslangValidator command");
	}

	if (opt_type_int & static_cast<int>(OptimizationType::OPTIMIZE_FOR_SPEED))
	{
		if (system(std::string("spirv-opt -O tmpshader.comp.spv -o tmpshader.comp.spv").c_str()))
		{
			throw std::runtime_error("Error running spirv-opt command");
		}
	}
	
	if (opt_type_int & static_cast<int>(OptimizationType::OPTIMIZE_FOR_SIZE)) {
		if (system(std::string("spirv-opt -Os tmpshader.comp.spv -o tmpshader.comp.spv").c_str()))
		{
			throw std::runtime_error("Error running spirv-opt command");
		}
	}

	if (opt_type_int & static_cast<int>(OptimizationType::REMAP)) {

		try {
			system(std::string("mkdir tmp").c_str());
		}
		catch (...) {}

		if (system(std::string("spirv-remap --do-everything --input tmpshader.comp.spv --output tmp").c_str())) 
		{
			throw std::runtime_error("Error running spirv-opt command");
		}

		last_shader_name = "tmp/tmpshader.comp.spv";
		fileStream = std::ifstream(last_shader_name, std::ios::binary);
	}
	else {
		last_shader_name = "tmpshader.comp.spv";
		fileStream = std::ifstream(last_shader_name, std::ios::binary);
	}

	std::vector<char> buffer;
	buffer.insert(
		buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
	return { (uint32_t*)buffer.data(),
			 (uint32_t*)(buffer.data() + buffer.size()) };
}

std::optional<std::string> glsl::decompileSPIRV(bool return_string)
{
	std::string cmd = "spirv-cross \"" + last_shader_name + "\" -V --output" + " \"tmpshader_built.comp\"";
	if (system(cmd.c_str()))
	{
		throw std::runtime_error("Error running spirv-cross command");
	}

	if (return_string) {
		std::ifstream fileStream("tmpshader_built.comp", std::ios::binary);
		std::vector<char> buffer;
		buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
		return std::make_optional<std::string>(std::string(buffer.begin(), buffer.end()));
	}

	return std::nullopt;
}