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


// UTIL
std::string glsl::shader_variable_type_to_str(const ShaderVariableType& type)
{
	switch (type) {
	case ShaderVariableType::FLOAT:
		return "float";
	case ShaderVariableType::DOUBLE:
		return "double";
	case ShaderVariableType::INT:
		return "int";
	default:
		throw std::runtime_error("Unsupported type");
	}
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

// BUFFER BINDING
std::string BufferBinding::operator()() const
{
	return "layout(set = 0, binding = " + std::to_string(m_Binding) + ") buffer buf_global_" +
		m_Name + " { " + m_Type + " " + "global_" + m_Name + "[]; };";
}

bool BufferBinding::operator==(const Binding* other) const
{
	if (auto* b = dynamic_cast<const BufferBinding*>(other); b != nullptr) {
		return (b->m_Binding == m_Binding) &&
			(b->m_Type == m_Type) && (b->m_Name == m_Name);
	}
	return false;
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
	ui16 ndim1, ui16 ndim2, const ShaderVariableType& type)
	: m_Name(name), m_NDim1(ndim1), m_NDim2(ndim2), m_Type(type)
{}

std::string MatrixVariable::getDeclaration() const
{
	std::string ret;
	switch (m_Type) {
	case ShaderVariableType::FLOAT:
		ret += "float ";
		break;
	case ShaderVariableType::DOUBLE:
		ret += "double ";
		break;
	case ShaderVariableType::INT:
		ret += "int ";
		break;
	default:
		throw std::runtime_error("Unsupported type");
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

std::string MatrixVariable::getUniqueID() const
{
	auto ret = std::to_string(m_NDim1) + "_" + std::to_string(m_NDim2) + "_";
	switch (m_Type) {
	case ShaderVariableType::FLOAT:
		ret += "S";
		break;
	case ShaderVariableType::DOUBLE:
		ret += "D";
		break;
	case ShaderVariableType::INT:
		ret += "I";
		break;
	default:
		throw std::runtime_error("Unsupported type");
	}
	return ret;
}

size_t MatrixVariable::getHash() const
{
	std::vector<size_t> hashes(3);
	hashes[0] = (size_t)m_NDim1;
	hashes[1] = (size_t)m_NDim2;
	hashes[2] = (size_t)m_Type;
	return util::hash_combine(hashes);
}

std::pair<ui16, ui16> MatrixVariable::getDimensions() const
{
	return std::make_pair(m_NDim1, m_NDim2);
}

ShaderVariableType MatrixVariable::getType() const
{
	return m_Type;
}

uint16_t MatrixVariable::getNDim1() const { return m_NDim1; }

uint16_t MatrixVariable::getNDim2() const { return m_NDim2; }

// VECTOR VARIABLE

VectorVariable::VectorVariable(const std::string& name,
	ui16 ndim, const ShaderVariableType& type)
	: m_Name(name), m_NDim(ndim), m_Type(type)
{}

std::string VectorVariable::getDeclaration() const
{
	std::string ret;
	switch (m_Type) {
	case ShaderVariableType::FLOAT:
		ret += "float ";
		break;
	case ShaderVariableType::DOUBLE:
		ret += "double ";
		break;
	case ShaderVariableType::INT:
		ret += "int ";
		break;
	default:
		throw std::runtime_error("Unsupported type");
	}
	ret += m_Name + "[" + std::to_string(m_NDim) + "];";
	return ret;
}

std::string VectorVariable::getName() const
{
	return m_Name;
}

std::string VectorVariable::getUniqueID() const
{
	auto ret = std::to_string(m_NDim) + "_";
	switch (m_Type) {
	case ShaderVariableType::FLOAT:
		ret += "S";
		break;
	case ShaderVariableType::DOUBLE:
		ret += "D";
		break;
	case ShaderVariableType::INT:
		ret += "I";
		break;
	default:
		throw std::runtime_error("Unsupported type");
	}
	return ret;
}

size_t VectorVariable::getHash() const
{
	std::vector<size_t> hashes(2);
	hashes[0] = (size_t)m_NDim;
	hashes[1] = (size_t)m_Type;
	return util::hash_combine(hashes);
}

uint16_t glsl::VectorVariable::getNDim() const
{
	return m_NDim;
}

ShaderVariableType glsl::VectorVariable::getType() const
{
	return m_Type;
}



// SINGLE VARIABLE

SingleVariable::SingleVariable(const std::string& name, const ShaderVariableType& type,
	const std::optional<std::string>& value)
	: m_Name(name), m_Type(type), m_Value(value)
{}

std::string SingleVariable::getDeclaration() const
{
	std::string ret;
	switch (m_Type) {
	case ShaderVariableType::FLOAT:
		ret += "float ";
		break;
	case ShaderVariableType::DOUBLE:
		ret += "double ";
		break;
	case ShaderVariableType::INT:
		ret += "int ";
		break;
	default:
		throw std::runtime_error("Unsupported type");
	}
	ret += m_Name;

	if (m_Value.has_value()) {
		ret += " = " + m_Value.value() + ";";
	}
	else {
		ret += ";";
	}
	return ret;
}

std::string SingleVariable::getName() const
{
	return m_Name;
}

std::string SingleVariable::getUniqueID() const
{
	return glsl::shader_variable_type_to_str(m_Type);
}

size_t SingleVariable::getHash() const
{
	std::vector<size_t> hashes(2);
	hashes[0] = (size_t)m_NDim;
	hashes[1] = (size_t)m_Type;
	return util::hash_combine(hashes);
}

// TEXTED VARIABLE

TextedVariable::TextedVariable(const std::string& name,
	const std::string& type,
	const std::string& value)
	: m_Name(name), m_Type(type), m_Value(value)
{}

std::string TextedVariable::getDeclaration() const
{
	return m_Type + " " + m_Name + " = " + m_Value + ";\n";
}

std::string TextedVariable::getName() const
{
	return m_Name;
}

std::string TextedVariable::getUniqueID() const
{
	return m_Type;
}

size_t TextedVariable::getHash() const
{
	return std::hash<std::string>()(m_Type);
}

// FUNCTION FACTORY

FunctionFactory::FunctionFactory(const std::string& name, const ShaderVariableType& return_type)
	: m_Name(name), m_ReturnType(return_type)
{}

void FunctionFactory::addMatrix(const std::shared_ptr<MatrixVariable>& mat, const std::optional<FunctionFactory::InputType>& input)
{
	if (input.has_value()) {
		std::string ret;
		switch (input.value())
		{
		case FunctionFactory::InputType::IN:
			ret += "in ";
			break;
		case FunctionFactory::InputType::OUT:
			ret += "out ";
			break;
		case FunctionFactory::InputType::INOUT:
			ret += "inout ";
			break;
		default:
			throw std::runtime_error("Not implemented type");
		}

		ret += shader_variable_type_to_str(mat->getType()) + " " + mat->getName() +
			"[" + std::to_string(mat->getNDim1()) + "*" + std::to_string(mat->getNDim2()) + "]";

		m_Inputs.emplace_back(std::move(ret));
	}

	_addVariable(mat);
}

void FunctionFactory::addVector(const std::shared_ptr<VectorVariable>& mat, const std::optional<FunctionFactory::InputType>& input)
{
	if (input.has_value()) {
		std::string ret;
		switch (input.value())
		{
		case FunctionFactory::InputType::IN:
			ret += "in ";
			break;
		case FunctionFactory::InputType::OUT:
			ret += "out ";
			break;
		case FunctionFactory::InputType::INOUT:
			ret += "inout ";
			break;
		default:
			throw std::runtime_error("Not implemented type");
		}

		ret += shader_variable_type_to_str(mat->getType()) + " " + mat->getName() +
			"[" + std::to_string(mat->getNDim()) + "]";

		m_Inputs.emplace_back(std::move(ret));
	}

	_addVariable(mat);
}

void FunctionFactory::addSingle(const std::shared_ptr<SingleVariable>& mat, const std::optional<FunctionFactory::InputType>& input)
{
	if (input.has_value()) {
		std::string ret;
		switch (input.value())
		{
		case FunctionFactory::InputType::IN:
			ret += "in ";
			break;
		case FunctionFactory::InputType::OUT:
			ret += "out ";
			break;
		case FunctionFactory::InputType::INOUT:
			ret += "inout ";
			break;
		default:
			throw std::runtime_error("Not implemented type");
		}

		ret += shader_variable_type_to_str(mat->getType()) + " " + mat->getName();

		m_Inputs.emplace_back(std::move(ret));
	}

	_addVariable(mat);
}

::glsl::Function FunctionFactory::build()
{
	std::string ret;

	ret += glsl::shader_variable_type_to_str(m_ReturnType) + " " + m_Name + "_";

	// add uniqueid



}

ui16 FunctionFactory::_addFunction(const Function& func)
{

}

ui16 FunctionFactory::_addVariable(const std::shared_ptr<ShaderVariable>& var)
{

}

// STUPIDSHADER

std::string StupidShader::compile() const
{
	return m_Str;
}

// AUTOGENSHADER

void AutogenShader::addBinding(std::unique_ptr<Binding> binding)
{
	_addBinding(std::move(binding));
}

void AutogenShader::addFunction(const Function& func)
{
	_addFunction(func);
}

void AutogenShader::addInputMatrix(const std::shared_ptr<MatrixVariable>& mat, ui16 binding)
{
	_addInputMatrix(mat, binding, true);
}

void AutogenShader::addOutputMatrix(const std::shared_ptr<MatrixVariable>& mat, ui16 binding)
{
	_addOutputMatrix(mat, binding, true);
}

void AutogenShader::addInputOutputMatrix(const std::shared_ptr<MatrixVariable>& mat, ui16 binding)
{
	_addInputMatrix(mat, binding, true);
	_addOutputMatrix(mat, binding, false);
}

void AutogenShader::addInputVector(const std::shared_ptr<VectorVariable>& vec, ui16 binding)
{
	_addInputVector(vec, binding, true);
}

void AutogenShader::addOutputVector(const std::shared_ptr<VectorVariable>& vec, ui16 binding)
{
	_addOutputVector(vec, binding, true);
}

void AutogenShader::addInputOutputVector(const std::shared_ptr<VectorVariable>& vec, ui16 binding)
{
	_addInputVector(vec, binding, true);
	_addOutputVector(vec, binding, false);
}

void AutogenShader::addVariable(const std::shared_ptr<ShaderVariable>& var)
{
	_addVariable(var, false);
}

void AutogenShader::setBeforeCopyingFrom(const std::string& mi)
{
	m_BeforeCopyingFrom = mi;
}

void AutogenShader::setAfterCopyingFrom(const std::string& mi)
{
	m_AfterCopyingFrom = mi;
}

void AutogenShader::setBeforeCopyingBack(const std::string& mi)
{
	m_BeforeCopyingBack = mi;
}

void AutogenShader::setAfterCopyingBack(const std::string& mi)
{
	m_AfterCopyingBack = mi;
}

std::string copying_from(const std::shared_ptr<glsl::ShaderVariable>& v1, const std::shared_ptr<glsl::ShaderVariable>& v2) 
{
	std::string copy_str;
	{
		auto i1m = dynamic_cast<const MatrixVariable*>(v1.get());
		auto i2m = dynamic_cast<const MatrixVariable*>(v2.get());
		if (i1m != nullptr && i2m != nullptr) {
			copy_str +=
R"glsl(
	start_index = nrow*ncol*gl_GlobalInvocationID.x;
	for (int i = 0; i < nrow*ncol; ++i) {
		OUTPUT_NAME[i] = INPUT_NAME[start_index + i];
	}

)glsl";
			util::replace_all(copy_str, "nrow", std::to_string(i1m->getNDim1()));
			util::replace_all(copy_str, "ncol", std::to_string(i1m->getNDim2()));
			util::replace_all(copy_str, "INPUT_NAME", i1m->getName());
			util::replace_all(copy_str, "OUTPUT_NAME", i2m->getName());
			return copy_str;
		}
	}

	{
		auto i1v = dynamic_cast<const VectorVariable*>(v1.get());
		auto i2v = dynamic_cast<const VectorVariable*>(v2.get());
		if (i1v != nullptr && i2v != nullptr) {
			copy_str +=
R"glsl(
	start_index = ndim*gl_GlobalInvocationID.x;
	for (int i = 0; i < ndim; ++i) {
		OUTPUT_NAME[i] = INPUT_NAME[start_index + i];
	}
)glsl";
			util::replace_all(copy_str, "ndim", std::to_string(i1v->getNDim()));
			util::replace_all(copy_str, "INPUT_NAME", i1v->getName());
			util::replace_all(copy_str, "OUTPUT_NAME", i2v->getName());
			return copy_str;
		}
	}

	throw std::runtime_error("Both variables must be either Vectors and Matrices");
}

std::string copying_to(const std::shared_ptr<glsl::ShaderVariable>& v1, const std::shared_ptr<glsl::ShaderVariable>& v2)
{
	std::string copy_str;
	{
		auto i1m = dynamic_cast<const MatrixVariable*>(v1.get());
		auto i2m = dynamic_cast<const MatrixVariable*>(v2.get());
		if (i1m != nullptr && i2m != nullptr) {
			copy_str +=
R"glsl(
	start_index = nrow*ncol*gl_GlobalInvocationID.x;
	for (int i = 0; i < nrow*ncol; ++i) {
		OUTPUT_NAME[start_index + i] = INPUT_NAME[i];
	}
)glsl";
			util::replace_all(copy_str, "nrow", std::to_string(i1m->getNDim1()));
			util::replace_all(copy_str, "ncol", std::to_string(i1m->getNDim2()));
			util::replace_all(copy_str, "INPUT_NAME", i1m->getName());
			util::replace_all(copy_str, "OUTPUT_NAME", i2m->getName());
			return copy_str;
		}
	}

	{
		auto i1v = dynamic_cast<const VectorVariable*>(v1.get());
		auto i2v = dynamic_cast<const VectorVariable*>(v2.get());
		if (i1v != nullptr && i2v != nullptr) {
			copy_str +=
R"glsl(
	start_index = ndim*gl_GlobalInvocationID.x;
	for (int i = 0; i < ndim; ++i) {
		OUTPUT_NAME[start_index + i] = INPUT_NAME[i];
	}
)glsl";
			util::replace_all(copy_str, "ndim", std::to_string(i1v->getNDim()));
			util::replace_all(copy_str, "INPUT_NAME", i1v->getName());
			util::replace_all(copy_str, "OUTPUT_NAME", i2v->getName());
			return copy_str;
		}
	}

	throw std::runtime_error("Both variables must be either Vectors and Matrices");
}

std::string AutogenShader::compile() const
{
	std::string ret;
	ret.reserve(DEFAULT_SHADER_SIZE);

	ret +=
R"glsl(
#version 450

layout (local_size_x = 1) in;

)glsl";

	for (auto& bind : m_Bindings) {
		ret += bind->operator()() + "\n";
	}
	ret += "\n";

	for (auto& func : m_Functions) {
		ret += func.getCode() + "\n";
	}
	ret += "\n";

	// open main
	ret += "void main() {\n";

	// declare variables
	for (auto& var : m_Variables) {
		if (!var.second)
			ret += "\t" + var.first->getDeclaration() + "\n";
	}
	ret += "\n";

	// manual insertions before copy from
	ret += m_BeforeCopyingFrom;

	// copy globals to locals
	if (m_Inputs.size() != 0 || m_Outputs.size() != 0)
		ret += "\tuint start_index;\n";
	for (auto& input : m_Inputs) {	
		ret += copying_from(m_Variables[input.first].first, m_Variables[input.second].first);
	}
	ret += "\n";

	// manual insertions after copy from
	ret += m_AfterCopyingFrom;

	// add in function calls
	for (auto& call : m_Calls) {
		ret += "\t";
		ui16 ret_pos = std::get<1>(call);
		if (ret_pos != -1) {
			ret += m_Variables[ret_pos].first->getName() + " = ";
		}

		ui16 func_pos = std::get<0>(call);

		ret += m_Functions[func_pos].getName() + "(";

		auto& input_pos_vec = std::get<2>(call);

		for (int i = 0; i < input_pos_vec.size(); ++i) {
			size_t arg_pos = input_pos_vec[i];
			ret += m_Variables[arg_pos].first->getName();
			if ((i + 1) != input_pos_vec.size()) {
				ret += ", ";
			}
		}
		ret += ");\n";
	}
	ret += "\n";

	// manual insertions before copying back
	ret += m_BeforeCopyingBack;

	// copy locals back to globals
	for (auto& output : m_Outputs) {
		ret += copying_to(m_Variables[output.first].first, m_Variables[output.second].first);
	}

	// manual insertions after copying back
	ret += m_AfterCopyingBack;

	// close main
	ret += "}\n";

	return ret;
}

ui16 AutogenShader::_addFunction(const Function& func) {
	return Function::add_function(m_Functions, func);
}

bool AutogenShader::_addBinding(std::unique_ptr<Binding> binding)
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

ui16 AutogenShader::_addVariable(const std::shared_ptr<ShaderVariable>& var, bool is_global)
{
	auto it = std::find_if(m_Variables.begin(), m_Variables.end(), [&var](const std::pair<std::shared_ptr<ShaderVariable>,bool>& v) {
		return *var == *(v.first);
	});
	ui16 pos = it - m_Variables.begin();
	if (it == m_Variables.end()) {
		m_Variables.emplace_back(var, is_global);
	}
	return pos;
}

void AutogenShader::_addInputMatrix(const std::shared_ptr<MatrixVariable>& mat, uint16_t binding, bool add_binding)
{
	auto ndim1 = mat->getNDim1();
	auto ndim2 = mat->getNDim2();

	auto global_mat = std::make_shared<MatrixVariable>(
		"global_" + mat->getName(), ndim1, ndim2, mat->getType());

	ui16 mat_index = _addVariable(mat, false);
	ui16 global_mat_index = _addVariable(global_mat, true);


	if (add_binding) {
		_addBinding(std::make_unique<BufferBinding>(binding, mat->getType(), mat->getName()));
	}

	m_Inputs.emplace_back(global_mat_index, mat_index);
}

void AutogenShader::_addOutputMatrix(const std::shared_ptr<MatrixVariable>& mat, ui16 binding, bool add_binding)
{
	auto ndim1 = mat->getNDim1();
	auto ndim2 = mat->getNDim2();

	auto global_mat = std::make_shared<MatrixVariable>(
		"global_" + mat->getName(), ndim1, ndim2, mat->getType());

	uint16_t mat_index = _addVariable(mat, false);
	uint16_t global_mat_index = _addVariable(global_mat, true);

	if (add_binding) {
		_addBinding(std::make_unique<BufferBinding>(binding, shader_variable_type_to_str(mat->getType()), mat->getName()));
	}

	m_Outputs.emplace_back(mat_index, global_mat_index);
}

void AutogenShader::_addInputVector(const std::shared_ptr<VectorVariable>& vec, ui16 binding, bool add_binding)
{
	auto ndim = vec->getNDim();

	auto global_vec = std::make_shared<VectorVariable>(
		"global_" + vec->getName(), ndim, vec->getType());

	uint16_t vec_index = _addVariable(vec, false);
	uint16_t global_vec_index = _addVariable(global_vec, true);

	if (add_binding) {
		_addBinding(std::make_unique<BufferBinding>(binding, shader_variable_type_to_str(vec->getType()), vec->getName()));
	}

	m_Inputs.emplace_back(global_vec_index, vec_index);
}

void AutogenShader::_addOutputVector(const std::shared_ptr<VectorVariable>& vec, ui16 binding, bool add_binding)
{
	auto ndim = vec->getNDim();

	auto global_vec = std::make_shared<VectorVariable>(
		"global_" + vec->getName(), ndim, vec->getType());

	uint16_t vec_index = _addVariable(vec, false);
	uint16_t global_vec_index = _addVariable(global_vec, true);

	if (add_binding) {
		_addBinding(std::make_unique<BufferBinding>(binding, shader_variable_type_to_str(vec->getType()), vec->getName()));
	}

	m_Outputs.emplace_back(vec_index, global_vec_index);
}

// SYMBOLIC CONTEXT

void SymbolicContext::insert_const(const std::pair<std::string, ui16>& cp)
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

void SymbolicContext::insert_param(const std::pair<std::string, ui16>& pp)
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

ui16 SymbolicContext::get_params_index(const std::string& name) const
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
		ui16 index = get_params_index(name);
		return params_name + "[" + std::to_string(index) + "]";
	}

	if (stype == glsl::eSymbolicType::CONST_TYPE) {
		ui16 index = get_consts_index(name);
		return consts_name + "[" + consts_iterable_by + "*" + nconst_name + "+" + std::to_string(index) + "]";
	}

	throw std::runtime_error("Variable was neither const nor param");
}

