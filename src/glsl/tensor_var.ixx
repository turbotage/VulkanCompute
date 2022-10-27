module;

export module tensor_var;

import <memory>;
import <string>;
import <stdexcept>;
import <optional>;

import <filesystem>;

import vc;
import variable;

import util;


namespace kp {
	export class Tensor;
	export class Manager;
}


namespace glsl {

	export std::shared_ptr<kp::Tensor> kp_tensor_from_matrix(const std::shared_ptr<kp::Manager>& mgr,
		const std::shared_ptr<glsl::MatrixVariable>& mat, vc::ui32 nelem);

	export std::shared_ptr<kp::Tensor> kp_tensor_from_vector(const std::shared_ptr<kp::Manager>& mgr,
		const std::shared_ptr<glsl::VectorVariable>& vec, vc::ui32 nelem);

	export std::shared_ptr<kp::Tensor> kp_tensor_from_single(const std::shared_ptr<kp::Manager>& mgr,
		const std::shared_ptr<glsl::SingleVariable>& var, vc::ui32 nelem);


	export std::shared_ptr<kp::Tensor> kp_tensor_from_matrix_file(const std::shared_ptr<kp::Manager>& mgr,
		const std::shared_ptr<glsl::MatrixVariable>& mat, const std::filesystem::path& filepath);
	
	export std::shared_ptr<kp::Tensor> kp_tensor_from_vector_file(const std::shared_ptr<kp::Manager>& mgr,
		const std::shared_ptr<glsl::VectorVariable>& mat, const std::filesystem::path& filepath);
	
	export std::shared_ptr<kp::Tensor> kp_tensor_from_single_file(const std::shared_ptr<kp::Manager>& mgr,
		const std::shared_ptr<glsl::SingleVariable>& mat, const std::filesystem::path& filepath);


	export std::shared_ptr<kp::Tensor> kp_tensor_from_file(const std::shared_ptr<kp::Manager>& mgr,
		const glsl::ShaderVariableType& type, const std::filesystem::path& filepath);

	export void kp_tensor_to_file(const std::shared_ptr<kp::Tensor>& mgr,
		const glsl::ShaderVariableType& type, const std::filesystem::path& filepath);

	export std::string print_shader_variable(const std::shared_ptr<kp::Tensor>& tensor,
		const std::shared_ptr<glsl::MatrixVariable>& mat, vc::ui32 index);

	export std::string print_shader_variable(const std::shared_ptr<kp::Tensor>& tensor,
		const std::shared_ptr<glsl::VectorVariable>& vec, vc::ui32 index);

	export std::string print_shader_variable(const std::shared_ptr<kp::Tensor>& tensor,
		const std::shared_ptr<glsl::SingleVariable>& var, vc::ui32 index);



}
