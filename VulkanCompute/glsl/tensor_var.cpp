module;

#include <kompute/Kompute.hpp>

module tensor_var;

std::shared_ptr<kp::Tensor> glsl::tensor_from_matrix(std::shared_ptr<kp::Manager> mgr,
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

std::shared_ptr<kp::Tensor> glsl::tensor_from_vector(std::shared_ptr<kp::Manager> mgr,
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

std::shared_ptr<kp::Tensor> glsl::tensor_from_single(std::shared_ptr<kp::Manager> mgr,
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
