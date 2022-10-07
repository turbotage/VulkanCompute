#pragma once

#include <kompute/Kompute.hpp>

#include <optional>
#include <memory>

import vc;
import variable;
import util;

namespace {
	constexpr int max_number_length = 10;
}

namespace glsl {

	template<typename T>
	std::string print_matrix(const std::shared_ptr<kp::Tensor>& tensor,
		const std::shared_ptr<glsl::MatrixVariable>& mat, vc::ui32 index,
		std::optional<vc::refw<std::vector<T>>> opt_vec)
	{
		vc::ui32 ndim1 = mat->getNDim1();
		vc::ui32 ndim2 = mat->getNDim2();

		std::string ret;
		vc::ui32 start_index = ndim1 * ndim2 * index;

		std::vector<T> vec_copy;
		std::vector<T>* vec;
		if (opt_vec.has_value()) {
			vec = &opt_vec.value().get();
		}
		else {
			vec_copy = tensor->vector<T>();
			vec = &vec_copy;
		}

		for (int i = 0; i < mat->getNDim1(); ++i) {
			for (int j = 0; j < mat->getNDim2(); ++j) {
				ret += util::add_whitespace_until(
					std::to_string(vec->at(start_index + i * ndim2 + j)), max_number_length);
			}
			ret += "\n";
		}

		return ret;
	}

	template<typename T>
	std::string print_vector(const std::shared_ptr<kp::Tensor>& tensor,
		const std::shared_ptr<glsl::VectorVariable>& vec, vc::ui32 index,
		std::optional<vc::refw<std::vector<T>>> opt_vec)
	{
		vc::ui32 ndim = vec->getNDim();

		std::string ret;
		vc::ui32 start_index = ndim * index;

		std::vector<T> vec_copy;
		std::vector<T>* vecp;
		if (opt_vec.has_value()) {
			vecp = &opt_vec.value().get();
		}
		else {
			vec_copy = tensor->vector<T>();
			vecp = &vec_copy;
		}

		for (int i = 0; i < ndim; ++i) {
			ret += util::add_whitespace_until(
				std::to_string(vecp->at(start_index + i)), max_number_length);
		}

		return ret;
	}

	template<typename T>
	std::string print_single(const std::shared_ptr<kp::Tensor>& tensor,
		const std::shared_ptr<glsl::SingleVariable>& var, vc::ui32 index,
		std::optional<vc::refw<std::vector<T>>> opt_vec)
	{

		std::string ret;
		vc::ui32 start_index = index;

		std::vector<T> vec_copy;
		std::vector<T>* vecp;
		if (opt_vec.has_value()) {
			vecp = &opt_vec.value().get();
		}
		else {
			vec_copy = tensor->vector<T>();
			vecp = &vec_copy;
		}

		ret += util::add_whitespace_until(
			std::to_string(vecp->at(start_index)), max_number_length);

		return ret;
	}

}

