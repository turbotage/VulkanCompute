module;

#define KOMPUTE_LOG_LEVEL KOMPUTE_LOG_LEVEL_CRITICAL
#include <kompute/Kompute.hpp>

module tensor_var;

import vc;
import variable;
import util;

namespace {
	constexpr int max_number_length = 15;
	const char* whitespace_characters = "  ";
}

std::shared_ptr<kp::Tensor> glsl::tensor_from_matrix(const std::shared_ptr<kp::Manager>& mgr,
	const std::shared_ptr<glsl::MatrixVariable>& mat, vc::ui32 nelem)
{
	switch (mat->getType()) {
	case glsl::ShaderVariableType::INT:
	{
		std::vector<int32_t> v(nelem * mat->getNDim1() * mat->getNDim2());
		return mgr->tensor(v.data(), v.size(), sizeof(int32_t),
			kp::Tensor::TensorDataTypes::eInt);
	}
	break;
	case glsl::ShaderVariableType::FLOAT:
	{
		std::vector<float> v(nelem * mat->getNDim1() * mat->getNDim2());
		return mgr->tensor(v.data(), v.size(), sizeof(float),
			kp::Tensor::TensorDataTypes::eFloat);
	}
	break;
	case glsl::ShaderVariableType::DOUBLE:
	{
		std::vector<double> v(nelem * mat->getNDim1() * mat->getNDim2());
		return mgr->tensor(v.data(), v.size(), sizeof(double),
			kp::Tensor::TensorDataTypes::eDouble);
	}
	break;
	default:
		throw std::runtime_error("Unsupported type - tensor_from_variable");
	}
}

std::shared_ptr<kp::Tensor> glsl::tensor_from_vector(const std::shared_ptr<kp::Manager>& mgr,
	const std::shared_ptr<glsl::VectorVariable>& vec, vc::ui32 nelem)
{
	switch (vec->getType()) {
	case glsl::ShaderVariableType::INT:
	{
		std::vector<int32_t> v(nelem * vec->getNDim());
		return mgr->tensor(v.data(), v.size(), sizeof(int32_t),
			kp::Tensor::TensorDataTypes::eInt);
	}
	break;
	case glsl::ShaderVariableType::FLOAT:
	{
		std::vector<float> v(nelem * vec->getNDim());
		return mgr->tensor(v.data(), v.size(), sizeof(float),
			kp::Tensor::TensorDataTypes::eFloat);
	}
	break;
	case glsl::ShaderVariableType::DOUBLE:
	{
		std::vector<double> v(nelem * vec->getNDim());
		return mgr->tensor(v.data(), v.size(), sizeof(double),
			kp::Tensor::TensorDataTypes::eDouble);
	}
	break;
	default:
		throw std::runtime_error("Unsupported type - tensor_from_variable");
	}
}

std::shared_ptr<kp::Tensor> glsl::tensor_from_single(const std::shared_ptr<kp::Manager>& mgr,
	const std::shared_ptr<glsl::SingleVariable>& var, vc::ui32 nelem)
{
	switch (var->getType()) {
	case glsl::ShaderVariableType::INT:
	{
		std::vector<int32_t> v(nelem);
		return mgr->tensor(v.data(), v.size(), sizeof(int32_t),
			kp::Tensor::TensorDataTypes::eInt);
	}
	break;
	case glsl::ShaderVariableType::FLOAT:
	{
		std::vector<float> v(nelem);
		return mgr->tensor(v.data(), v.size(), sizeof(float),
			kp::Tensor::TensorDataTypes::eFloat);
	}
	break;
	case glsl::ShaderVariableType::DOUBLE:
	{
		std::vector<double> v(nelem);
		return mgr->tensor(v.data(), v.size(), sizeof(double),
			kp::Tensor::TensorDataTypes::eDouble);
	}
	break;
	default:
		throw std::runtime_error("Unsupported type - tensor_from_variable");
	}
}



std::string glsl::print_shader_variable(const std::shared_ptr<kp::Tensor>& tensor, 
	const std::shared_ptr<glsl::MatrixVariable>& mat, vc::ui32 index)
{
	auto printer_lambda = [&mat, &index]<typename T>(T * data) -> std::string {
		std::string ret;
		vc::ui32 ndim1 = mat->getNDim1();
		vc::ui32 ndim2 = mat->getNDim2();
		vc::ui32 start_index = ndim1 * ndim2 * index;
		for (int i = 0; i < ndim1; ++i) {
			for (int j = 0; j < ndim2; ++j) {
				ret += util::add_whitespace_until(
					std::to_string(data[start_index + i * ndim2 + j]), max_number_length) + whitespace_characters;
			}
			ret += "\n";
		}
		return ret;
	};

	switch (mat->getType()) {
	case glsl::ShaderVariableType::INT:
	{
		if (tensor->dataType() != kp::Tensor::TensorDataTypes::eInt) {
			throw std::runtime_error("TensorDataType and ShaderVariableType did not agree");
		}
		return printer_lambda(tensor->data<int32_t>());
	}
		break;
	case glsl::ShaderVariableType::FLOAT:
	{
		if (tensor->dataType() != kp::Tensor::TensorDataTypes::eFloat) {
			throw std::runtime_error("TensorDataType and ShaderVariableType did not agree");
		}
		return printer_lambda(tensor->data<float>());
	}
	break;
	case glsl::ShaderVariableType::DOUBLE:
	{
		if (tensor->dataType() != kp::Tensor::TensorDataTypes::eDouble) {
			throw std::runtime_error("TensorDataType and ShaderVariableType did not agree");
		}
		return printer_lambda(tensor->data<double>());
	}
	break;
	default:
		throw std::runtime_error("Unsupported TensorDataType");
	}
}

std::string glsl::print_shader_variable(const std::shared_ptr<kp::Tensor>& tensor, 
	const std::shared_ptr<glsl::VectorVariable>& vec, vc::ui32 index)
{
	auto printer_lambda = [&vec, &index]<typename T>(T * data) -> std::string {
		std::string ret;
		vc::ui32 ndim = vec->getNDim();
		vc::ui32 start_index = ndim * index;
		for (int i = 0; i < ndim; ++i) {
			ret += util::add_whitespace_until(
				std::to_string(data[start_index + i]), max_number_length) + whitespace_characters;
		}
		return ret + "\n";
	};

	switch (vec->getType()) {
	case glsl::ShaderVariableType::INT:
	{
		if (tensor->dataType() != kp::Tensor::TensorDataTypes::eInt) {
			throw std::runtime_error("TensorDataType and ShaderVariableType did not agree");
		}
		return printer_lambda(tensor->data<int32_t>());
	}
	break;
	case glsl::ShaderVariableType::FLOAT:
	{
		if (tensor->dataType() != kp::Tensor::TensorDataTypes::eFloat) {
			throw std::runtime_error("TensorDataType and ShaderVariableType did not agree");
		}
		return printer_lambda(tensor->data<float>());
	}
	break;
	case glsl::ShaderVariableType::DOUBLE:
	{
		if (tensor->dataType() != kp::Tensor::TensorDataTypes::eDouble) {
			throw std::runtime_error("TensorDataType and ShaderVariableType did not agree");
		}
		return printer_lambda(tensor->data<double>());
	}
	break;
	default:
		throw std::runtime_error("Unsupported TensorDataType");
	}
}

std::string glsl::print_shader_variable(const std::shared_ptr<kp::Tensor>& tensor, 
	const std::shared_ptr<glsl::SingleVariable>& var, vc::ui32 index)
{
	auto printer_lambda = [&var, &index]<typename T>(T * data) -> std::string {
		std::string ret;
		ret += util::add_whitespace_until(
			std::to_string(data[index]), max_number_length) + whitespace_characters;
		return ret + "\n";
	};

	switch (var->getType()) {
	case glsl::ShaderVariableType::INT:
	{
		if (tensor->dataType() != kp::Tensor::TensorDataTypes::eInt) {
			throw std::runtime_error("TensorDataType and ShaderVariableType did not agree");
		}
		return printer_lambda(tensor->data<int32_t>());
	}
	break;
	case glsl::ShaderVariableType::FLOAT:
	{
		if (tensor->dataType() != kp::Tensor::TensorDataTypes::eFloat) {
			throw std::runtime_error("TensorDataType and ShaderVariableType did not agree");
		}
		return printer_lambda(tensor->data<float>());
	}
	break;
	case glsl::ShaderVariableType::DOUBLE:
	{
		if (tensor->dataType() != kp::Tensor::TensorDataTypes::eDouble) {
			throw std::runtime_error("TensorDataType and ShaderVariableType did not agree");
		}
		return printer_lambda(tensor->data<double>());
	}
	break;
	default:
		throw std::runtime_error("Unsupported TensorDataType");
	}
}

