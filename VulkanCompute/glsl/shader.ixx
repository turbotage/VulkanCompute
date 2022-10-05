module;

export module shader;

import <string>;
import <memory>;
import <optional>;
import <stdexcept>;

import vc;
import util;
import glsl;

import variable;
import function;

namespace {
	// most shaders are smaller than 30 kB
	constexpr auto DEFAULT_SHADER_SIZE = 30000;
}

namespace glsl {

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

	export class ShaderBase {
	public:

		virtual std::string compile() const = 0;

	private:

	};

	export class StupidShader : public ShaderBase {
	public:

		StupidShader(const std::string& str)
			: m_Str(str) {}

		std::string compile() const override
		{
			return m_Str;
		}

	private:
		std::string m_Str;
	};

	export class AutogenShader : public ShaderBase {
	public:

		void addBinding(std::unique_ptr<Binding> binding)
		{
			_addBinding(std::move(binding));
		}

		void addFunction(const std::shared_ptr<Function>& func)
		{
			_addFunction(func);
		}

		void addInputMatrix(const std::shared_ptr<MatrixVariable>& mat, vc::ui16 binding)
		{
			_addInputMatrix(mat, binding, true);
		}

		void addOutputMatrix(const std::shared_ptr<MatrixVariable>& mat, vc::ui16 binding)
		{
			_addOutputMatrix(mat, binding, true);
		}

		void addInputOutputMatrix(const std::shared_ptr<MatrixVariable>& mat, vc::ui16 binding)
		{
			_addInputMatrix(mat, binding, true);
			_addOutputMatrix(mat, binding, false);
		}

		void addInputVector(const std::shared_ptr<VectorVariable>& vec, vc::ui16 binding)
		{
			_addInputVector(vec, binding, true);
		}

		void addOutputVector(const std::shared_ptr<VectorVariable>& vec, vc::ui16 binding)
		{
			_addOutputVector(vec, binding, true);
		}

		void addInputOutputVector(const std::shared_ptr<VectorVariable>& vec, vc::ui16 binding)
		{
			_addInputVector(vec, binding, true);
			_addOutputVector(vec, binding, false);
		}

		void addVariable(const std::shared_ptr<ShaderVariable>& var)
		{
			_addVariable(var, false);
		}

		void setBeforeCopyingFrom(const std::string& mi)
		{
			m_BeforeCopyingFrom = mi;
		}

		void setAfterCopyingFrom(const std::string& mi)
		{
			m_AfterCopyingFrom = mi;
		}

		void setBeforeCopyingBack(const std::string& mi)
		{
			m_BeforeCopyingBack = mi;
		}

		void setAfterCopyingBack(const std::string& mi)
		{
			m_AfterCopyingBack = mi;
		}

		template<ShaderVariableIterator SVIterator>
		void apply(const std::shared_ptr<Function>& func,
			const std::shared_ptr<ShaderVariable>& ret,
			std::optional<std::pair<SVIterator, SVIterator>> args_it)
		{
			vc::ui16 func_pos = Function::add_function(m_Functions, func);

			int16_t ret_pos = -1;
			if (ret) {
				ret_pos = _addVariable(ret, false);
			}

			std::vector<vc::ui16> input_pos;
			if (args_it.has_value()) {
				auto& its = args_it.value();
				size_t ninputs = std::distance(its.first, its.second);
				input_pos.reserve(ninputs);
				for (auto it = its.first; it != its.second; ++it) {
					input_pos.emplace_back(_addVariable(*it, false));
				}
			}

			m_Calls.emplace_back(func_pos, ret_pos, std::move(input_pos));
		}

		void apply(const std::shared_ptr<Function>& func,
			const std::shared_ptr<ShaderVariable>& ret,
			const std::vector<std::shared_ptr<ShaderVariable>>& args)
		{
			apply(func, ret, std::make_optional(std::make_pair(args.begin(), args.end())));
		}

		std::string compile() const override
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
				ret += func->getCode() + "\n";
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
				vc::ui16 ret_pos = std::get<1>(call);
				if (ret_pos != -1) {
					ret += m_Variables[ret_pos].first->getName() + " = ";
				}

				vc::ui16 func_pos = std::get<0>(call);

				ret += m_Functions[func_pos]->getName() + "(";

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

	private:

		vc::ui16 _addFunction(const std::shared_ptr<Function>& func)
		{
			return Function::add_function(m_Functions, func);
		}

		bool _addBinding(std::unique_ptr<Binding> binding)
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

		vc::ui16 _addVariable(const std::shared_ptr<ShaderVariable>& var, bool is_global)
		{
			auto it = std::find_if(m_Variables.begin(), m_Variables.end(), [&var](const std::pair<std::shared_ptr<ShaderVariable>, bool>& v) {
				return *var == *(v.first);
				});
			vc::ui16 pos = it - m_Variables.begin();
			if (it == m_Variables.end()) {
				m_Variables.emplace_back(var, is_global);
			}
			return pos;
		}

		void _addInputMatrix(const std::shared_ptr<MatrixVariable>& mat, vc::ui16 binding, bool add_binding)
		{
			auto ndim1 = mat->getNDim1();
			auto ndim2 = mat->getNDim2();

			auto global_mat = std::make_shared<MatrixVariable>(
				"global_" + mat->getName(), ndim1, ndim2, mat->getType());

			vc::ui16 mat_index = _addVariable(mat, false);
			vc::ui16 global_mat_index = _addVariable(global_mat, true);


			if (add_binding) {
				_addBinding(std::make_unique<BufferBinding>(binding, mat->getType(), mat->getName()));
			}

			m_Inputs.emplace_back(global_mat_index, mat_index);
		}

		void _addOutputMatrix(const std::shared_ptr<MatrixVariable>& mat, vc::ui16 binding, bool add_binding)
		{
			auto ndim1 = mat->getNDim1();
			auto ndim2 = mat->getNDim2();

			auto global_mat = std::make_shared<MatrixVariable>(
				"global_" + mat->getName(), ndim1, ndim2, mat->getType());

			vc::ui16 mat_index = _addVariable(mat, false);
			vc::ui16 global_mat_index = _addVariable(global_mat, true);

			if (add_binding) {
				_addBinding(std::make_unique<BufferBinding>(binding, shader_variable_type_to_str(mat->getType()), mat->getName()));
			}

			m_Outputs.emplace_back(mat_index, global_mat_index);
		}

		void _addInputVector(const std::shared_ptr<VectorVariable>& vec, vc::ui16 binding, bool add_binding)
		{
			auto ndim = vec->getNDim();

			auto global_vec = std::make_shared<VectorVariable>(
				"global_" + vec->getName(), ndim, vec->getType());

			vc::ui16 vec_index = _addVariable(vec, false);
			vc::ui16 global_vec_index = _addVariable(global_vec, true);

			if (add_binding) {
				_addBinding(std::make_unique<BufferBinding>(binding, shader_variable_type_to_str(vec->getType()), vec->getName()));
			}

			m_Inputs.emplace_back(global_vec_index, vec_index);
		}

		void _addOutputVector(const std::shared_ptr<VectorVariable>& vec, vc::ui16 binding, bool add_binding)
		{
			auto ndim = vec->getNDim();

			auto global_vec = std::make_shared<VectorVariable>(
				"global_" + vec->getName(), ndim, vec->getType());

			vc::ui16 vec_index = _addVariable(vec, false);
			vc::ui16 global_vec_index = _addVariable(global_vec, true);

			if (add_binding) {
				_addBinding(std::make_unique<BufferBinding>(binding, shader_variable_type_to_str(vec->getType()), vec->getName()));
			}

			m_Outputs.emplace_back(vec_index, global_vec_index);
		}

	private:

		std::vector<std::unique_ptr<Binding>> m_Bindings;
		std::vector<std::shared_ptr<Function>> m_Functions;

		std::vector<std::pair<std::shared_ptr<ShaderVariable>, bool>> m_Variables;

		std::string m_BeforeCopyingFrom;
		std::string m_AfterCopyingFrom;
		std::string m_BeforeCopyingBack;
		std::string m_AfterCopyingBack;

		std::vector<std::pair<vc::ui16, vc::ui16>> m_Inputs;
		std::vector<std::pair<vc::ui16, vc::ui16>> m_Outputs;
		// m_Calls[0] is index to function in m_Functions
		// m_Calls[i] for i > 0 is variable to use in function call, index is to
		// m_Variables
		std::vector<std::tuple<vc::ui16, vc::ui16, std::vector<vc::ui16>>> m_Calls;

	};



}