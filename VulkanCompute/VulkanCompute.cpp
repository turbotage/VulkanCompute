// VulkanCompute.cpp : Defines the entry point for the application.
//
#include <kompute/Kompute.hpp>
#include <memory>
#include <optional>

import glsl;
import linalg;


static std::vector<uint32_t>
compileSource(const std::string& source)
{
	std::ofstream fileOut("tmp_kp_shader.comp");
	fileOut << source;
	fileOut.close();
	if (system(
		std::string(
			"glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv")
		.c_str()))
		throw std::runtime_error("Error running glslangValidator command");
	std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
	std::vector<char> buffer;
	buffer.insert(
		buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
	return { (uint32_t*)buffer.data(),
			 (uint32_t*)(buffer.data() + buffer.size()) };
}

int main() {

	glsl::Shader shader;
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 0, "float", "global_in1"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 1, "float", "global_in2"));

	static std::string code =
		R"glsl(
#define N 4

void main() {
	uint index = gl_GlobalInvocationID.x;
	swap(global_in1[N*index], global_in2[N*index]);
	swap(global_in1[N*index + 1], global_in2[N*index + 1]);
	swap(global_in1[N*index + 2], global_in2[N*index + 2]);
	swap(global_in1[N*index + 3], global_in2[N*index + 3]);
}
)glsl";

	std::function<std::string()> code_func = []() -> std::string
	{
		std::string temp = code;
		return temp;
	};

	glsl::Function func("main", {}, code_func, std::make_optional<std::vector<glsl::Function>>({ glsl::linalg::swap() }));

	shader.addFunction(func);

	std::string glsl_shader = shader.compile();

	std::cout << glsl_shader << std::endl;

	auto spirv = compileSource(glsl_shader);

	std::string spirv_str(spirv.begin(), spirv.end());
	
	kp::Manager mgr;

	auto tensor1 = mgr.tensor({
		/* work1 */ 1.0, 2.0, 3.0, 4.0,
		/* work2 */ 5.0, 6.0, 7.0, 8.0,
		/* work3 */ 9.0, 10.0, 11.0, 12.0 });

	auto tensor2 = mgr.tensor({
		/* work1 */ 10.0, 20.0, 30.0, 40.0,
		/* work2 */ 50.0, 60.0, 70.0, 80.0,
		/* work3 */ 90.0, 100.0, 110.0, 120.0 });

	std::vector<std::shared_ptr<kp::Tensor>> params = { tensor1, tensor2 };

	kp::Workgroup wg({ 3,1,1 });
	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv, wg);

	mgr.sequence()
		->record<kp::OpTensorSyncDevice>(params)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpTensorSyncLocal>(params)
		->eval();

	std::cout << "tensor1: { ";
	for (const float& elem : tensor1->vector()) {
		std::cout << elem << "  ";
	}
	std::cout << "}" << std::endl;

	std::cout << "tensor2: { ";
	for (const float& elem : tensor2->vector()) {
		std::cout << elem << "  ";
	}
	std::cout << "}" << std::endl;

	return 0;
}



