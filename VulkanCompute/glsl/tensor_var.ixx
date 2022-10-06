module;

//#include <kompute/Kompute.hpp>

export module tensor_var;

import <memory>;

import vc;
import variable;

namespace kp {
	export class Tensor;
	export class Manager;
}

namespace glsl {

	export std::shared_ptr<kp::Tensor> tensor_from_matrix(std::shared_ptr<kp::Manager> mgr,
		const std::shared_ptr<glsl::MatrixVariable>& mat, vc::ui32 nelem);

	export std::shared_ptr<kp::Tensor> tensor_from_vector(std::shared_ptr<kp::Manager> mgr,
		const std::shared_ptr<glsl::VectorVariable>& vec, vc::ui32 nelem);

	export std::shared_ptr<kp::Tensor> tensor_from_single(std::shared_ptr<kp::Manager> mgr,
		const std::shared_ptr<glsl::SingleVariable>& var, vc::ui32 nelem);

}
