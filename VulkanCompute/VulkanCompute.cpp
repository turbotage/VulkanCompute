// VulkanCompute.cpp : Defines the entry point for the application.
//

#include <vector>
#include <kompute/Kompute.hpp>
#include <memory>
#include <optional>
#include <random>
#include <chrono>
#include <symengine/expression.h>
#include <symengine/refine.h>
#include <symengine/simplify.h>
#include <symengine/parser.h>
#include <symengine/parser/parser.h>
#include <string>

import vc;
import util;
import glsl;
import linalg;
import solver;
import symbolic;
import expr;
import symm;

/*
void test_expr_res_glsl(bool print = true) {
	glsl::Shader shader;
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 0, "float", "global_param"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 1, "float", "global_const"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 2, "float", "global_data"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 3, "float", "global_residuals"));

	static std::string code =
R"glsl(

void main() {

	float param[nparam];
	float consts[ndata*nconst];
	float data[ndata];
	float residuals[ndata];

	uint index = gl_GlobalInvocationID.x;
	uint startindex = index*nparam;	

	// copy params
	for (int i = 0; i < nparam; ++i) {
		param[i] = global_param[startindex + i];
	}
	
	startindex = index*ndata*nconst;
	// copy consts
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < nconst; ++j) {
			consts[i*nconst + j] = global_const[startindex + i*nconst + j];
		}
	}

	startindex = index*ndata;
	// copy data
	for (int i = 0; i < ndata; ++i) {
		data[i] = global_data[startindex + i];
	}

	mod_nlsq_residuals(param, consts, data, residuals);

	startindex = index*ndata;
	// copy back residuals
	for (int i = 0; i < ndata; ++i) {
		global_residuals[startindex + i] = residuals[i];
	}

}
)glsl";

	int ndata = 4;
	int nparam = 2;
	int nconst = 1;

	std::function<std::string()> code_func = [ndata, nparam, nconst]() -> std::string
	{
		std::string temp = code;
		util::replace_all(temp, "ndata", std::to_string(ndata));
		util::replace_all(temp, "nparam", std::to_string(nparam));
		util::replace_all(temp, "nconst", std::to_string(nconst));
		return temp;
	};

	expression::Expression expr("s0*exp(-b*adc)", { "s0", "b", "adc" });
	glsl::SymbolicContext context;
	context.insert_const(std::make_pair("b", 0));
	context.insert_param(std::make_pair("s0", 0));
	context.insert_param(std::make_pair("adc", 1));


	glsl::Function func("main", {}, code_func,
		std::make_optional<std::vector<glsl::Function>>({
			glsl::symbolic::nlsq::nlsq_residuals("mod", expr, context, ndata, nparam, nconst, true)
		})
	);

	shader.addFunction(func);
	

	std::string glsl_shader = shader.compile();

	std::cout << glsl_shader << std::endl;

	auto spirv = glsl::compileSource(glsl_shader);
	
	std::string spirv_str(spirv.begin(), spirv.end());

	kp::Manager mgr;

	auto data = mgr.tensor({
		100.0,      81.87307531,	44.93289641,	20.1896518,		// work1
		2*100.0,    2*81.87307531,  2*44.93289641,  2*20.1896518	// work2
		});
	
	auto param = mgr.tensor({
		100.0, 0.002,	// work1
		200.0, 0.002	// work2
		});

	auto consts = mgr.tensor({
		0.0,	// work1
		100.0,
		400.0,
		800.0,
		0.0,	// work2
		100.0,
		400.0,
		800.0,
		});

	auto residuals = mgr.tensor({
		10.0, 10.0, 10.0, 10.0, // work1
		20.0, 20.0, 20.0, 20.0, // work2
		});

	std::vector<std::shared_ptr<kp::Tensor>> params = { param, consts, data, residuals };

	kp::Workgroup wg({ 2,1,1 });
	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv, wg);

	auto start = std::chrono::steady_clock::now();

	mgr.sequence()
		->record<kp::OpTensorSyncDevice>(params)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpTensorSyncLocal>(params)
		->eval();

	auto end = std::chrono::steady_clock::now();

	std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< std::endl;

	if (print) {
		auto res = residuals->vector();
		std::string printr = "residuals: \n";
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < ndata; ++j) {
				printr += std::to_string(res[i * ndata + j]) + "  ";
			}
			printr += "\n";
		}
		std::cout << printr;
	}
}

void test_expr_res_jac_glsl(bool print = true) {
	glsl::Shader shader;
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 0, "float", "global_param"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 1, "float", "global_const"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 2, "float", "global_data"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 3, "float", "global_residuals"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 4, "float", "global_jacobian"));

	static std::string code =
R"glsl(

void main() {

	float param[nparam];
	float consts[ndata*nconst];
	float data[ndata];
	float residuals[ndata];
	float jacobian[ndata*nparam];

	uint index = gl_GlobalInvocationID.x;

	// copy params
	uint startindex = index*nparam;	
	for (int i = 0; i < nparam; ++i) {
		param[i] = global_param[startindex + i];
	}
	
	// copy consts
	startindex = index*ndata*nconst;
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < nconst; ++j) {
			consts[i*nconst + j] = global_const[startindex + i*nconst + j];
		}
	}

	// copy data
	startindex = index*ndata;
	for (int i = 0; i < ndata; ++i) {
		data[i] = global_data[startindex + i];
	}

	mod_nlsq_residuals_jacobian(param, consts, data, residuals, jacobian);

	// copy back residuals
	startindex = index*ndata;
	for (int i = 0; i < ndata; ++i) {
		global_residuals[startindex + i] = residuals[i];
	}

	
	// copy back jacobian
	startindex = index*ndata*nparam;
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < nparam; ++j) {
			global_jacobian[startindex + i*nparam + j] = jacobian[i*nparam + j];
		}
	}

}
)glsl";

	int n = 2;

	int ndata = 3;
	int nparam = 4;
	int nconst = 1;

	std::function<std::string()> code_func = [ndata, nparam, nconst]() -> std::string
	{
		std::string temp = code;
		util::replace_all(temp, "ndata", std::to_string(ndata));
		util::replace_all(temp, "nparam", std::to_string(nparam));
		util::replace_all(temp, "nconst", std::to_string(nconst));
		return temp;
	};

	expression::Expression expr("s0*(f*exp(-b*d1)+(1-f)*exp(-b*d2))", { "s0", "f", "d1", "d2", "b" });
	glsl::SymbolicContext context;
	context.insert_param(std::make_pair("s0", 0));
	context.insert_param(std::make_pair("f", 1));
	context.insert_param(std::make_pair("d1", 2));
	context.insert_param(std::make_pair("d2", 3));
	context.insert_const(std::make_pair("b", 0));

	glsl::Function func("main", {}, code_func,
		std::make_optional<std::vector<glsl::Function>>({
			glsl::symbolic::nlsq::nlsq_residuals_jacobian("mod", expr, context, ndata, nparam, nconst, true)
			})
	);

	shader.addFunction(func);

	std::string glsl_shader = shader.compile();

	std::cout << glsl_shader << std::endl;

	auto spirv = glsl::compileSource(glsl_shader);

	std::string spirv_str(spirv.begin(), spirv.end());

	kp::Manager mgr;

	auto data = mgr.tensor({
		0.1793,  0.1655,  0.1487,		// work1
		0.3959,  0.3686,  0.3355		// work2
		});

	auto param = mgr.tensor({
		0.2041, 0.9306, 0.4506, 0.4654,	// work1
		0.4456, 0.6436, 0.5457, 0.1791	// work2
		});

	auto consts = mgr.tensor({
		0.2876,  0.4650,  0.7020,		// work1
		0.2876,  0.4650,  0.7020		// work2
		});

	auto residuals = mgr.tensor({
		10.0, 10.0, 10.0, 10.0,			// work1
		20.0, 20.0, 20.0, 20.0,			// work2
		});

	auto jacobian = mgr.tensor(std::vector<float>(n * ndata * nparam, 0.0f));

	std::vector<std::shared_ptr<kp::Tensor>> params = { param, consts, data, residuals, jacobian };

	kp::Workgroup wg({ (size_t)n ,1,1 });
	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv, wg);

	auto start = std::chrono::steady_clock::now();

	mgr.sequence()
		->record<kp::OpTensorSyncDevice>(params)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpTensorSyncLocal>(params)
		->eval();

	auto end = std::chrono::steady_clock::now();

	std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< std::endl;


	if (print) {
		auto res = residuals->vector();
		auto jac = jacobian->vector();
		std::string printr = "";
		for (int i = 0; i < n; ++i) {
			printr += "residuals: \n";
			for (int j = 0; j < ndata; ++j) {
				printr += std::to_string(res[i * ndata + j]) + "  ";
			}
			printr += "\n jacobian: \n";
			for (int j = 0; j < ndata; ++j) {
				for (int k = 0; k < nparam; ++k) {
					printr += std::to_string(jac[i * ndata * nparam + j * nparam + k]) + "  ";
				}
				printr += "\n";
			}
		}

		std::cout << printr;
	}
}

void test_expr_res_jac_hes_glsl(bool print = true) {
	glsl::Shader shader;
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 0, "float", "global_param"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 1, "float", "global_const"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 2, "float", "global_data"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 3, "float", "global_residuals"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 4, "float", "global_jacobian"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 5, "float", "global_hessian"));

	static std::string code =
		R"glsl(

void main() {

	float param[nparam];
	float consts[ndata*nconst];
	float data[ndata];
	float residuals[ndata];
	float jacobian[ndata*nparam];
	float hessian[nparam*nparam];

	uint index = gl_GlobalInvocationID.x;

	// copy params
	uint startindex = index*nparam;	
	for (int i = 0; i < nparam; ++i) {
		param[i] = global_param[startindex + i];
	}
	
	// copy consts
	startindex = index*ndata*nconst;
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < nconst; ++j) {
			consts[i*nconst + j] = global_const[startindex + i*nconst + j];
		}
	}

	// copy data
	startindex = index*ndata;
	for (int i = 0; i < ndata; ++i) {
		data[i] = global_data[startindex + i];
	}

	mod_nlsq_residuals_jacobian_hessian(param, consts, data, residuals, jacobian, hessian);

	// copy back residuals
	startindex = index*ndata;
	for (int i = 0; i < ndata; ++i) {
		global_residuals[startindex + i] = residuals[i];
	}

	
	// copy back jacobian
	startindex = index*ndata*nparam;
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < nparam; ++j) {
			global_jacobian[startindex + i*nparam + j] = jacobian[i*nparam + j];
		}
	}

	// copy back hessian
	startindex = index*nparam*nparam;
	for (int i = 0; i < nparam; ++i) {
		for (int j = 0; j < nparam; ++j) {
			global_hessian[startindex + i*nparam + j] = hessian[i*nparam + j];
		}
	}

}
)glsl";

	int n = 2;

	int ndata = 3;
	int nparam = 4;
	int nconst = 1;

	std::function<std::string()> code_func = [ndata, nparam, nconst]() -> std::string
	{
		std::string temp = code;
		util::replace_all(temp, "ndata", std::to_string(ndata));
		util::replace_all(temp, "nparam", std::to_string(nparam));
		util::replace_all(temp, "nconst", std::to_string(nconst));
		return temp;
	};

	expression::Expression expr("s0*(f*exp(-b*d1)+(1-f)*exp(-b*d2))", { "s0", "f", "d1", "d2", "b" });
	glsl::SymbolicContext context;
	context.insert_param(std::make_pair("s0", 0));
	context.insert_param(std::make_pair("f", 1));
	context.insert_param(std::make_pair("d1", 2));
	context.insert_param(std::make_pair("d2", 3));
	context.insert_const(std::make_pair("b", 0));

	glsl::Function func("main", {}, code_func,
		std::make_optional<std::vector<glsl::Function>>({
			glsl::symbolic::nlsq::nlsq_residuals_jacobian_hessian("mod", expr, context, ndata, nparam, nconst, true)
			})
	);

	shader.addFunction(func);


	std::string glsl_shader = shader.compile();

	std::cout << glsl_shader << std::endl;

	auto spirv = glsl::compileSource(glsl_shader);

	std::string spirv_str(spirv.begin(), spirv.end());

	kp::Manager mgr;

	auto data = mgr.tensor({
		0.1793,  0.1655,  0.1487,		// work1
		0.3959,  0.3686,  0.3355		// work2
		});

	auto param = mgr.tensor({
		0.2041, 0.9306, 0.4506, 0.4654,	// work1
		0.4456, 0.6436, 0.5457, 0.1791	// work2
		});

	auto consts = mgr.tensor({
		0.2876,  0.4650,  0.7020,		// work1
		0.2876,  0.4650,  0.7020		// work2
		});

	auto residuals = mgr.tensor({
		10.0, 10.0, 10.0, 10.0,			// work1
		20.0, 20.0, 20.0, 20.0,			// work2
		});

	auto jacobian = mgr.tensor(std::vector<float>(n * ndata * nparam, 0.0f));
	auto hessian = mgr.tensor(std::vector<float>(n * nparam * nparam, 0.0f));

	std::vector<std::shared_ptr<kp::Tensor>> params = { param, consts, data, residuals, jacobian, hessian };

	kp::Workgroup wg({ (size_t)n ,1,1 });
	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv, wg);

	auto start = std::chrono::steady_clock::now();

	mgr.sequence()
		->record<kp::OpTensorSyncDevice>(params)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpTensorSyncLocal>(params)
		->eval();

	auto end = std::chrono::steady_clock::now();

	std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< std::endl;


	if (print) {
		auto res = residuals->vector();
		auto jac = jacobian->vector();
		auto hes = hessian->vector();
		std::string printr = ""; 
		for (int i = 0; i < n; ++i) {
			printr += "residuals: \n";
			for (int j = 0; j < ndata; ++j) {
				printr += std::to_string(res[i * ndata + j]) + "  ";
			}
			printr += "\n jacobian: \n";
			for (int j = 0; j < ndata; ++j) {
				for (int k = 0; k < nparam; ++k) {
					printr += std::to_string(jac[i*ndata*nparam + j*nparam + k]) + "  ";
				}
				printr += "\n";
			}
			printr += "\n hessian: \n";
			for (int j = 0; j < nparam; ++j) {
				for (int k = 0; k < nparam; ++k) {
					printr += std::to_string(hes[i*nparam*nparam + j*nparam + k]) + "  ";
				}
				printr += "\n";
			}
		}

		std::cout << printr;
	}
}

void test_expr_res_jac_hes_glsl_time_many(int n = 1000000, int runs = 5, bool print = false) {
	glsl::Shader shader;
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 0, "float", "global_param"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 1, "float", "global_const"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 2, "float", "global_data"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 3, "float", "global_residuals"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 4, "float", "global_jacobian"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 5, "float", "global_hessian"));

	static std::string code =
		R"glsl(

void main() {

	float param[nparam];
	float consts[ndata*nconst];
	float data[ndata];
	float residuals[ndata];
	float jacobian[ndata*nparam];
	float hessian[nparam*nparam];

	uint index = gl_GlobalInvocationID.x;

	// copy params
	uint startindex = index*nparam;	
	for (int i = 0; i < nparam; ++i) {
		param[i] = global_param[startindex + i];
	}
	
	// copy consts
	startindex = index*ndata*nconst;
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < nconst; ++j) {
			consts[i*nconst + j] = global_const[startindex + i*nconst + j];
		}
	}

	// copy data
	startindex = index*ndata;
	for (int i = 0; i < ndata; ++i) {
		data[i] = global_data[startindex + i];
	}

	mod_nlsq_residuals_jacobian_hessian(param, consts, data, residuals, jacobian, hessian);

	// copy back residuals
	startindex = index*ndata;
	for (int i = 0; i < ndata; ++i) {
		global_residuals[startindex + i] = residuals[i];
	}

	
	// copy back jacobian
	startindex = index*ndata*nparam;
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < nparam; ++j) {
			global_jacobian[startindex + i*nparam + j] = jacobian[i*nparam + j];
		}
	}

	// copy back hessian
	startindex = index*nparam*nparam;
	for (int i = 0; i < nparam; ++i) {
		for (int j = 0; j < nparam; ++j) {
			global_hessian[startindex + i*nparam + j] = hessian[i*nparam + j];
		}
	}

}
)glsl";

	int ndata = 3;
	int nparam = 4;
	int nconst = 1;

	std::function<std::string()> code_func = [ndata, nparam, nconst]() -> std::string
	{
		std::string temp = code;
		util::replace_all(temp, "ndata", std::to_string(ndata));
		util::replace_all(temp, "nparam", std::to_string(nparam));
		util::replace_all(temp, "nconst", std::to_string(nconst));
		return temp;
	};

	expression::Expression expr("s0*(f*exp(-b*d1)+(1-f)*exp(-b*d2))", { "s0", "f", "d1", "d2", "b" });
	glsl::SymbolicContext context;
	context.insert_param(std::make_pair("s0", 0));
	context.insert_param(std::make_pair("f", 1));
	context.insert_param(std::make_pair("d1", 2));
	context.insert_param(std::make_pair("d2", 3));
	context.insert_const(std::make_pair("b", 0));

	glsl::Function func("main", {}, code_func,
		std::make_optional<std::vector<glsl::Function>>({
			glsl::symbolic::nlsq::nlsq_residuals_jacobian_hessian("mod", expr, context, ndata, nparam, nconst, true)
			})
	);

	shader.addFunction(func);


	std::string glsl_shader = shader.compile();

	//std::cout << glsl_shader << std::endl;

	auto spirv = glsl::compileSource(glsl_shader);

	std::string spirv_str(spirv.begin(), spirv.end());

	kp::Manager mgr;

	auto data = mgr.tensor(std::vector<float>(ndata * n, 0.0f));

	auto param = mgr.tensor(std::vector<float>(nparam * n, 0.0f));

	auto consts = mgr.tensor(std::vector<float>(nconst * ndata * n, 0.0f));

	auto residuals = mgr.tensor(std::vector<float>(ndata * n, 0.0f));

	auto jacobian = mgr.tensor(std::vector<float>(n * ndata * nparam, 0.0f));
	auto hessian = mgr.tensor(std::vector<float>(n * nparam * nparam, 0.0f));

	std::vector<std::shared_ptr<kp::Tensor>> params = { param, consts, data, residuals, jacobian, hessian };

	kp::Workgroup wg({ (size_t)n ,1,1 });
	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv, wg);

	auto start = std::chrono::steady_clock::now();
	
	auto r = mgr.sequence()->record<kp::OpTensorSyncDevice>(params);
	for (int i = 0; i < runs; ++i) {
		r = r->record<kp::OpAlgoDispatch>(algo);
	}
	r->record<kp::OpTensorSyncLocal>(params)->eval();

	auto end = std::chrono::steady_clock::now();

	std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< std::endl;

	if (print) {
		auto res = residuals->vector();
		auto jac = jacobian->vector();
		auto hes = hessian->vector();
		std::string printr = "";
		for (int i = 0; i < std::min(3,n); ++i) {
			printr += "residuals: \n";
			for (int j = 0; j < ndata; ++j) {
				printr += std::to_string(res[i * ndata + j]) + "  ";
			}
			printr += "\n jacobian: \n";
			for (int j = 0; j < ndata; ++j) {
				for (int k = 0; k < nparam; ++k) {
					printr += std::to_string(jac[i * ndata * nparam + j * nparam + k]) + "  ";
				}
				printr += "\n";
			}
			printr += "\n hessian: \n";
			for (int j = 0; j < nparam; ++j) {
				for (int k = 0; k < nparam; ++k) {
					printr += std::to_string(hes[i * nparam * nparam + j * nparam + k]) + "  ";
				}
				printr += "\n";
			}
		}

		std::cout << printr;
	}
}

void test_ldl() {

	glsl::Shader shader;
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 0, "float", "global_mat"));

	static std::string code =
R"glsl(
void main() {
	float mat[ndim*ndim];

	uint index = gl_GlobalInvocationID.x;
	uint startindex = index*ndim*ndim;

	// copy mat
	for (int i = 0; i < ndim; ++i) {
		for (int j = 0; j < ndim; ++j) {
			mat[i*ndim + j] = global_mat[i*ndim + j];
		}
	}
	
	ldl(mat);

	// copy back mat
	for (int i = 0; i < ndim; ++i) {
		for (int j = 0; j < ndim; ++j) {
			global_mat[i*ndim + j] = mat[i*ndim + j];
		}
	}

}
)glsl";

	int n = 1;
	int ndim = 3;

	std::function<std::string()> code_func = [ndim]() -> std::string
	{
		std::string temp = code;
		util::replace_all(temp, "ndim", std::to_string(ndim));
		return temp;
	};

	glsl::Function func("main", {}, code_func, std::make_optional<std::vector<glsl::Function>>({
		glsl::linalg::ldl(ndim, true)
		}));

	shader.addFunction(func);

	std::string glsl_shader = shader.compile();

	std::cout << util::add_line_numbers(glsl_shader) << std::endl;

	auto spirv = glsl::compileSource(glsl_shader);

	kp::Manager mgr;

	auto mat = mgr.tensor({
		3,2,1,
		2,4,2,
		1,2,5
		});

	std::vector<std::shared_ptr<kp::Tensor>> params = { mat };

	kp::Workgroup wg({ (size_t)n,1,1 });
	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv, wg);

	auto start = std::chrono::steady_clock::now();

	mgr.sequence()
		->record<kp::OpTensorSyncDevice>(params)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpTensorSyncLocal>(params)
		->eval();

	auto end = std::chrono::steady_clock::now();

	std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< std::endl;

	bool print = true;
	if (print) {
		auto res = mat->vector();
		std::string printr = "mat: \n";
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < ndim; ++j) {
				for (int k = 0; k < ndim; ++k) {
					printr += std::to_string(res[i*ndim*ndim + j*ndim + k]) + "  ";
				}
				printr += "\n";
			}
		}
		std::cout << printr;
	}

}

void test_gmw81() {

	glsl::Shader shader;
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 0, "float", "global_mat"));
	shader.addBinding(std::make_unique<glsl::BufferBinding>(0, 1, "float", "global_vec"));

	static std::string code =
R"glsl(
void main() {
	float mat[ndim*ndim];

	uint index = gl_GlobalInvocationID.x;
	uint startindex = index*ndim*ndim;

	// copy mat
	for (int i = 0; i < ndim; ++i) {
		for (int j = 0; j < ndim; ++j) {
			mat[i*ndim + j] = global_mat[startindex + i*ndim + j];
		}
	}
	
	gmw81(mat);

	// copy back mat
	for (int i = 0; i < ndim; ++i) {
		for (int j = 0; j < ndim; ++j) {
			global_mat[startindex + i*ndim + j] = mat[i*ndim + j];
		}
	}

	float rhs[ndim];
	startindex = index*ndim;
	for (int i = 0; i < ndim; ++i) {
		rhs[i] = global_vec[startindex + i];
	}
	float sol[ndim];

	ldl_solve(mat, rhs, sol);

	for (int i = 0; i < ndim; ++i) {
		global_vec[startindex + i] = sol[i];
	}

}
)glsl";

	int n = 1;
	int ndim = 3;

	std::function<std::string()> code_func = [ndim]() -> std::string
	{
		std::string temp = code;
		util::replace_all(temp, "ndim", std::to_string(ndim));
		return temp;
	};

	glsl::Function func("main", {}, code_func, std::make_optional<std::vector<glsl::Function>>({
		glsl::linalg::gmw81(ndim, true),
		glsl::linalg::ldl_solve(ndim, true)
		}));

	shader.addFunction(func);

	std::string glsl_shader = shader.compile();

	std::cout << util::add_line_numbers(glsl_shader) << std::endl;

	auto spirv = glsl::compileSource(glsl_shader);

	kp::Manager mgr;

	auto mat = mgr.tensor({
		3,2,1,
		2,4,2,
		1,2,5
		});

	auto vec = mgr.tensor({ 1, 1, 1 });

	std::vector<std::shared_ptr<kp::Tensor>> params = { mat, vec };

	kp::Workgroup wg({ (size_t)n,1,1 });
	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv, wg);

	auto start = std::chrono::steady_clock::now();

	mgr.sequence()
		->record<kp::OpTensorSyncDevice>(params)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpTensorSyncLocal>(params)
		->eval();

	auto end = std::chrono::steady_clock::now();

	std::cout << "time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
		<< std::endl;

	bool print = true;
	if (print) {
		auto res = mat->vector();
		std::string printr = "mat: \n";
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < ndim; ++j) {
				for (int k = 0; k < ndim; ++k) {
					printr += std::to_string(res[i * ndim * ndim + j * ndim + k]) + "  ";
				}
				printr += "\n";
			}
			printr += "\n\n";
		}
		res = vec->vector();
		printr += "vec: \n";
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < ndim; ++j) {
				printr += std::to_string(res[i * ndim + j]) + "  ";
			}
			printr += "\n\n";
		}
		std::cout << printr;
	}

}
*/

void test_new_gensystem()
{
	using namespace glsl;
	using namespace vc;

	Shader shader;
	
	ui32 n = 1;
	ui16 ndim = 3;

	auto mat = std::make_shared<glsl::MatrixVariable>("mat", ndim, ndim, true);
	auto rhs = std::make_shared<glsl::VectorVariable>("rhs", ndim, true);
	auto sol = std::make_shared<glsl::VectorVariable>("sol", ndim, true);

	shader.addInputOutputMatrix(mat, 0);
	shader.addInputVector(rhs, 1);
	shader.addOutputVector(sol, 2);

	shader.apply(linalg::gmw81(ndim, true), nullptr, { mat });
	shader.apply(linalg::ldl_solve(ndim, true), nullptr, { mat, rhs, sol });

	std::string glsl_shader = shader.compile();
	
	std::cout << util::add_line_numbers(glsl_shader) << std::endl;

	auto spirv = glsl::compileSource(glsl_shader);

	kp::Manager mgr;

	auto mat_tensor = mgr.tensor({
		3,2,1,
		2,4,2,
		1,2,5
		});

	auto rhs_tensor = mgr.tensor({
		1,1,1
		});

	auto sol_tensor = mgr.tensor({
		0,0,0
		});

	std::vector<std::shared_ptr<kp::Tensor>> params = { mat_tensor, rhs_tensor, sol_tensor };

	kp::Workgroup wg({ (size_t)n,1,1 });

	std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(params, spirv, wg);

	auto start = std::chrono::steady_clock::now();

	mgr.sequence()
		->record<kp::OpTensorSyncDevice>(params)
		->record<kp::OpAlgoDispatch>(algo)
		->record<kp::OpTensorSyncLocal>(params)
		->eval();

	auto end = std::chrono::steady_clock::now();

	std::cout << "time: " << 
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	bool print = true;
	if (print) {
		auto res = mat_tensor->vector();
		std::string printr = "decomp: \n";
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < ndim; ++j) {
				for (int k = 0; k < ndim; ++k) {
					printr += std::to_string(res[i * ndim * ndim + j * ndim + k]) + "  ";
				}
				printr += "\n";
			}
			printr += "\n\n";
		}
		res = sol_tensor->vector();
		printr += "sol: \n";
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < ndim; ++j) {
				printr += std::to_string(res[i * ndim + j]) + "  ";
			}
			printr += "\n\n";
		}
		std::cout << printr;
	}

}

int main() {
	
	test_new_gensystem();

	return 0;
}



