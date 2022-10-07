module;

export module tensor_var;

import <memory>;
import <string>;
import <stdexcept>;
import <optional>;

import vc;
import variable;

import util;


namespace kp {
	export class Tensor;
	export class Manager;
}


namespace glsl {

	export std::shared_ptr<kp::Tensor> tensor_from_matrix(const std::shared_ptr<kp::Manager>& mgr,
		const std::shared_ptr<glsl::MatrixVariable>& mat, vc::ui32 nelem);

	export std::shared_ptr<kp::Tensor> tensor_from_vector(const std::shared_ptr<kp::Manager>& mgr,
		const std::shared_ptr<glsl::VectorVariable>& vec, vc::ui32 nelem);

	export std::shared_ptr<kp::Tensor> tensor_from_single(const std::shared_ptr<kp::Manager>& mgr,
		const std::shared_ptr<glsl::SingleVariable>& var, vc::ui32 nelem);



	export std::string print_shader_variable(const std::shared_ptr<kp::Tensor>& tensor,
		const std::shared_ptr<glsl::MatrixVariable>& mat, vc::ui32 index);

	export std::string print_shader_variable(const std::shared_ptr<kp::Tensor>& tensor,
		const std::shared_ptr<glsl::VectorVariable>& vec, vc::ui32 index);

	export std::string print_shader_variable(const std::shared_ptr<kp::Tensor>& tensor,
		const std::shared_ptr<glsl::SingleVariable>& var, vc::ui32 index);

}
