// VulkanCompute.cpp : Defines the entry point for the application.
//
#include <kompute/Kompute.hpp>
#include <memory>
#include <optional>

import glsl;
import linalg;
import solver;

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
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 0, "float", "global_in"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 1, "float", "global_out"));

	static std::string code =
		R"glsl(
#define N 3

float mat[N*N];
float omat[N*N];
int pivot[N];

void main() {
	uint index = gl_GlobalInvocationID.x;
	uint startindex = index*N*N;	

	// Copy global mat into temp matrix
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			mat[i*N + j] = global_in[startindex + i*N + j];
		}
	}

	// Perform LU decomp

	lu(mat, pivot);

	//mul_unit_lower_upper_square(mat, mat, omat);

	// Copy global mat into temp matrix
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			global_out[startindex + i*N + j] = mat[i*N + j];
		}
	}

}
)glsl";

	std::function<std::string()> code_func = []() -> std::string
	{
		std::string temp = code;
		return temp;
	};

	glsl::Function func("main", {}, code_func, 
		std::make_optional<std::vector<glsl::Function>>({ 
			glsl::linalg::solver::lu(3),
			//glsl::linalg::mul_unit_lower_upper_square(3),
		})
	);

	shader.addFunction(func);

	std::string glsl_shader = shader.compile();

	std::cout << glsl_shader << std::endl;

	auto spirv = compileSource(glsl_shader);

	std::string spirv_str(spirv.begin(), spirv.end());
	
	kp::Manager mgr;

	auto tensor1 = mgr.tensor({
		/* work1 - row 1 */ 1.0, 2.0, 3.0,
		/* work1 - row 2 */ 4.0, 5.0, 6.0,
		/* work1 - row 3 */ 7.0, 8.0, 9.0,
		
		/* work2 - row 1 */ 3.0, 1.0, 1.0,
		/* work2 - row 2 */ 1.0, 5.0, 1.0,
		/* work2 - row 3 */ 1.0, 1.0, 9.0,
		});

	auto tensor2 = mgr.tensor({
		/* work1 - row 1 */ 0.0, 0.0, 0.0,
		/* work1 - row 2 */ 0.0, 0.0, 0.0,
		/* work1 - row 3 */ 0.0, 0.0, 0.0,

		/* work2 - row 1 */ 0.0, 0.0, 0.0,
		/* work2 - row 2 */ 0.0, 0.0, 0.0,
		/* work2 - row 3 */ 0.0, 0.0, 0.0,
		});

	std::vector<std::shared_ptr<kp::Tensor>> params = { tensor1, tensor2 };

	kp::Workgroup wg({ 2,1,1 });
	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv, wg);

	mgr.sequence()
		->record<kp::OpTensorSyncDevice>(params)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpTensorSyncLocal>(params)
		->eval();

	std::cout << "tensor1: [ \n";
	auto v1 = tensor1->vector();
	for (int k = 0; k < 2; ++k) {
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				std::cout << v1[k*3*3 + i*3 + j] << "  ";
			}
			std::cout << "\n";
		}
	}
	std::cout << "]" << std::endl;

	std::cout << "tensor2: [ \n";
	v1 = tensor2->vector();
	for (int k = 0; k < 2; ++k) {
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				std::cout << v1[k * 3 * 3 + i * 3 + j] << "  ";
			}
			std::cout << "\n";
		}
	}
	std::cout << "]" << std::endl;

	return 0;
}



