module;

#include <memory>
#include <string>

#include <kompute/Kompute.hpp>

module glsl;

using namespace glsl;

import linalg;

std::vector<uint32_t> glsl::compileSource(const std::string& source)
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

// FUNCTION

Function::Function(
	const std::string& function_name,
	const std::vector<size_t>& argument_hashes,
	const std::function<std::string()>& code_func,
	std::optional<std::vector<Function>> dependencies)
	:
	m_FunctionName(function_name),
	m_ArgumentHashes(argument_hashes),
	m_CodeFunc(code_func)

{
	if (dependencies.has_value())
		m_Dependencies = dependencies.value();
}

const std::vector<Function>& Function::getDependencies() const {
	return m_Dependencies;
}

std::string Function::getCode() const {
	return m_CodeFunc();
}

std::string Function::getName() const {
	return m_FunctionName;
}

bool glsl::operator==(const Function& lhs, const Function& rhs)
{
	return lhs.m_FunctionName == rhs.m_FunctionName &&
		lhs.m_ArgumentHashes == rhs.m_ArgumentHashes;
}


size_t Function::add_function(std::vector<Function>& funcs, const Function& func) {
	for (auto& f : func.m_Dependencies) {
		add_function(funcs, f);
	}

	auto it = std::find(funcs.begin(), funcs.end(), func);
	size_t pos = it - funcs.begin();
	if (it == funcs.end()) {
		funcs.push_back(func);
	}
	return pos;
}

// SHADER VARIABLE

size_t ShaderVariable::add_variable(std::vector<std::shared_ptr<ShaderVariable>>& vars,
	const std::shared_ptr<ShaderVariable>& var)
{

	auto it = std::find_if(vars.begin(), vars.end(), [&var](const std::shared_ptr<ShaderVariable>& v) {
			return *v == *var;
		});

	size_t pos = it - vars.begin();
	if (it == vars.end()) {
		vars.push_back(var);
	}
	return pos;
}

// MATRIX VARIABLE

MatrixVariable::MatrixVariable(const std::string& name,
	uint16_t ndim1, uint16_t ndim2, bool single_precission)
	: m_Name(name), m_NDim1(ndim1), m_NDim2(ndim2), m_SinglePrecission(single_precission)
{}

std::string MatrixVariable::getDeclaration() const
{
	std::string ret;
	if (m_SinglePrecission) {
		ret += "float ";
	}
	else {
		ret += "double ";
	}
	ret += m_Name + "[" +
		std::to_string(m_NDim1) + "*" +
		std::to_string(m_NDim2) + "];";
	return ret;
}

std::string MatrixVariable::getName() const
{
	return m_Name;
}

std::pair<uint16_t, uint16_t> MatrixVariable::getDimensions() const
{
	return std::make_pair(m_NDim1, m_NDim2);
}

bool MatrixVariable::isSinglePrecission() const
{
	return m_SinglePrecission;
}

uint16_t MatrixVariable::getNDim1() const { return m_NDim1; }

uint16_t MatrixVariable::getNDim2() const { return m_NDim2; }


// VECTOR VARIABLE

VectorVariable::VectorVariable(const std::string& name,
	uint16_t ndim, bool single_precission)
	: m_Name(name), m_NDim(ndim), m_SinglePrecission(single_precission)
{}

std::string VectorVariable::getDeclaration() const
{
	std::string ret;
	if (m_SinglePrecission) {
		ret += "float ";
	}
	else {
		ret += "double ";
	}
	ret += m_Name + "[" + std::to_string(m_NDim) + "];";
	return ret;
}

std::string VectorVariable::getName() const
{
	return m_Name;
}

uint16_t glsl::VectorVariable::getNDim() const
{
	return m_NDim;
}

bool glsl::VectorVariable::isSinglePrecission() const
{
	return m_SinglePrecission;
}

// SHADER 

void Shader::addFunction(const Function& func)
{
	_addFunction(func);
}

void Shader::addInputMatrix(const std::shared_ptr<MatrixVariable>& mat, uint16_t binding)
{
	_addInputMatrix(mat, binding, true);
}

void Shader::addOutputMatrix(const std::shared_ptr<MatrixVariable>& mat, uint16_t binding)
{
	_addOutputMatrix(mat, binding, true);
}

void Shader::addInputOutputMatrix(const std::shared_ptr<MatrixVariable>& mat, uint16_t binding)
{
	_addInputMatrix(mat, binding, true);
	_addOutputMatrix(mat, binding, false);
}

void Shader::addInputVector(const std::shared_ptr<VectorVariable>& vec, uint16_t binding)
{
	_addInputVector(vec, binding, true);
}

void Shader::addOutputVector(const std::shared_ptr<VectorVariable>& vec, uint16_t binding)
{
	_addOutputVector(vec, binding, true);
}

void Shader::addInputOutputVector(const std::shared_ptr<VectorVariable>& vec, uint16_t binding)
{
	_addInputVector(vec, binding, true);
	_addInputVector(vec, binding, false);
}

void Shader::addVariable(const std::shared_ptr<ShaderVariable>& var)
{
	_addVariable(var);
}

std::string Shader::compile() const
{
	std::string ret =
		R"glsl(
#version 450

layout (local_size_x = 1) in;

)glsl";

	for (auto& bind : m_Bindings) {
		ret += bind->operator()() + "\n";
	}

	for (auto& func : m_Functions) {
		ret += func.getCode() + "\n";
	}

	// open main
	ret += "void main() {\n";

	// declare variables
	for (auto& var : m_Variables) {
		ret += "\t" + var->getDeclaration() + "\n";
	}

	// copy globals to locals
	for (auto& input : m_Inputs) {
		ret += "\t";
		uint16_t func_pos = std::get<0>(input);
		uint16_t input1_pos = std::get<1>(input);
		uint16_t input2_pos = std::get<2>(input);

		ret += m_Functions[func_pos].getName() + "(" + m_Variables[input1_pos]->getName() +
			m_Variables[input2_pos]->getName() + ");\n";
	}

	// add in function calls
	for (auto& call : m_Calls) {
		ret += "\t";
		int16_t ret_pos = std::get<1>(call);
		if (ret_pos == -1) {
			ret += m_Variables[ret_pos]->getName() + " = ";
		}

		uint16_t func_pos = std::get<0>(call);

		ret += m_Functions[func_pos].getName() + "(";

		auto& input_pos_vec = std::get<2>(call);

		for (int i = 0; i < input_pos_vec.size(); ++i) {
			size_t arg_pos = input_pos_vec[i];
			ret += m_Variables[arg_pos]->getName();
			if ((i + 1) != input_pos_vec.size()) {
				ret += ", ";
			}
		}
		ret += ")\n";
	}


	// copy locals back to globals
	for (auto& input : m_Inputs) {
		ret += "\t";
		uint16_t func_pos = std::get<0>(input);
		uint16_t input1_pos = std::get<1>(input);
		uint16_t input2_pos = std::get<2>(input);

		ret += m_Functions[func_pos].getName() + "(" + m_Variables[input1_pos]->getName() +
			m_Variables[input2_pos]->getName() + ");\n";
	}

	// close main
	ret += "\n}\n";

	return ret;
}

uint16_t Shader::_addFunction(const Function& func) {
	return Function::add_function(m_Functions, func);
}

bool Shader::_addBinding(std::unique_ptr<Binding> binding)
{
	auto it = std::find_if(m_Bindings.begin(), m_Bindings.end(), [&binding](const std::unique_ptr<Binding>& b)
		{
			return binding->operator==(b.get());
		});

	if (it == m_Bindings.end()) {
		m_Bindings.emplace_back(std::move(binding));
		return true;
	}
	return false;
}

uint16_t Shader::_addVariable(const std::shared_ptr<ShaderVariable>& var)
{
	auto it = std::find(m_Variables.begin(), m_Variables.end(), var);
	size_t pos = it - m_Variables.end();
	if (it == m_Variables.end()) {
		m_Variables.emplace_back(var);
	}
	return pos;
}

void Shader::_addInputMatrix(const std::shared_ptr<MatrixVariable>& mat, uint16_t binding, bool add_binding)
{
	auto ndim1 = mat->getNDim1();
	auto ndim2 = mat->getNDim2();
	bool sp = mat->isSinglePrecission();

	auto global_mat = std::make_shared<MatrixVariable>(
		"global_" + mat->getName(), ndim1, ndim2, sp);

	uint16_t mat_index = _addVariable(mat);
	uint16_t global_mat_index = _addVariable(global_mat);

	uint16_t copy_func_index = _addFunction(linalg::copy_mat_istarted(ndim1, ndim2, sp));

	if (add_binding) {
		_addBinding(std::make_unique<BufferBinding>(binding, (sp ? "float" : "double"), mat->getName()));
	}

	m_Inputs.emplace_back(copy_func_index, mat_index, global_mat_index);
}

void Shader::_addOutputMatrix(const std::shared_ptr<MatrixVariable>& mat, uint16_t binding, bool add_binding)
{
	auto ndim1 = mat->getNDim1();
	auto ndim2 = mat->getNDim2();
	bool sp = mat->isSinglePrecission();

	auto global_mat = std::make_shared<MatrixVariable>(
		"global_" + mat->getName(), ndim1, ndim2, sp);

	uint16_t mat_index = _addVariable(mat);
	uint16_t global_mat_index = _addVariable(global_mat);

	uint16_t copy_func_index = _addFunction(linalg::copy_mat_ostarted(ndim1, ndim2, sp));

	if (add_binding) {
		_addBinding(std::make_unique<BufferBinding>(binding, (sp ? "float" : "double"), mat->getName()));
	}

	m_Outputs.emplace_back(copy_func_index, global_mat_index, mat_index);
}

void Shader::_addInputVector(const std::shared_ptr<VectorVariable>& vec, uint16_t binding, bool add_binding)
{
	auto ndim = vec->getNDim();
	bool sp = vec->isSinglePrecission();

	auto global_vec = std::make_shared<VectorVariable>(
		"global_" + vec->getName(), ndim, sp);

	uint16_t vec_index = _addVariable(vec);
	uint16_t global_vec_index = _addVariable(global_vec);

	uint16_t copy_func_index = _addFunction(linalg::copy_vec_istarted(ndim, sp));

	if (add_binding) {
		_addBinding(std::make_unique<BufferBinding>(binding, (sp ? "float" : "double"), vec->getName()));
	}

	m_Inputs.emplace_back(copy_func_index, global_vec_index, vec_index);
}

void Shader::_addOutputVector(const std::shared_ptr<VectorVariable>& vec, uint16_t binding, bool add_binding)
{
	auto ndim = vec->getNDim();
	bool sp = vec->isSinglePrecission();

	auto global_vec = std::make_shared<VectorVariable>(
		"global_" + vec->getName(), ndim, sp);

	uint16_t vec_index = _addVariable(vec);
	uint16_t global_vec_index = _addVariable(global_vec);

	uint16_t copy_func_index = _addFunction(linalg::copy_vec_ostarted(ndim, sp));

	if (add_binding) {
		_addBinding(std::make_unique<BufferBinding>(binding, (sp ? "float" : "double"), vec->getName()));
	}

	m_Inputs.emplace_back(copy_func_index, vec_index, global_vec_index);
}

// SYMBOLIC CONTEXT

void SymbolicContext::insert_const(const std::pair<std::string, uint32_t>& cp)
{
	if (symtype_map.contains(cp.first))
		throw std::runtime_error("const name already existed in SymbolicContext");

	for (auto& p : consts_map) {
		if (p.first == cp.first)
			throw std::runtime_error("const name already existed in SymbolicContext");
		if (p.second == cp.second)
			throw std::runtime_error("const index already existed in SymbolicContext");
	}

	symtype_map.insert({ cp.first, eSymbolicType::CONST_TYPE });
	consts_map.insert(cp);
}

void SymbolicContext::insert_param(const std::pair<std::string, uint32_t>& pp)
{
	if (symtype_map.contains(pp.first))
		throw std::runtime_error("const name already existed in SymbolicContext");

	for (auto& p : params_map) {
		if (p.first == pp.first)
			throw std::runtime_error("const name already existed in SymbolicContext");
		if (p.second == pp.second)
			throw std::runtime_error("const index already existed in SymbolicContext");
	}

	symtype_map.insert({ pp.first, eSymbolicType::PARAM_TYPE });
	params_map.insert(pp);
}

eSymbolicType SymbolicContext::get_symtype(const std::string& name) const
{
	return symtype_map.at(name);
}

uint32_t SymbolicContext::get_params_index(const std::string& name) const
{
	for (auto& v : params_map) {
		if (v.first == name)
			return v.second;
	}
	throw std::runtime_error("Name was not in params in SymbolicContext");
}

const std::string& SymbolicContext::get_params_name(size_t index) const
{
	for (auto& v : params_map) {
		if (v.second == index)
			return v.first;
	}
	throw std::runtime_error("Index was not in params in SymbolicContext");
}

uint32_t SymbolicContext::get_consts_index(const std::string& name) const
{
	for (auto& v : consts_map) {
		if (v.first == name)
			return v.second;
	}
	throw std::runtime_error("Name was not in consts in SymbolicContext");
}

const std::string& SymbolicContext::get_consts_name(size_t index) const {
	for (auto& v : consts_map) {
		if (v.second == index)
			return v.first;
	}
	throw std::runtime_error("Index was not in consts in SymbolicContext");
}

const std::string& SymbolicContext::get_consts_name() const {
	return consts_name;
}

const std::string& SymbolicContext::get_consts_iterable_by() const {
	return consts_iterable_by;
}

const std::string& SymbolicContext::get_params_iterable_by() const {
	return params_iterable_by;
}

std::string SymbolicContext::get_glsl_var_name(const std::string& name) const
{
	glsl::eSymbolicType stype = symtype_map.at(name);

	if (stype == glsl::eSymbolicType::PARAM_TYPE) {
		uint32_t index = get_params_index(name);
		return params_name + "[" + std::to_string(index) + "]";
	}

	if (stype == glsl::eSymbolicType::CONST_TYPE) {
		uint32_t index = get_consts_index(name);
		return consts_name + "[" + consts_iterable_by + "*" + nconst_name + "+" + std::to_string(index) + "]";
	}

	throw std::runtime_error("Variable was neither const nor param");
}

