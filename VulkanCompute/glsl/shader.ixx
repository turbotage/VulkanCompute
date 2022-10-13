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

import func_factory;

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

		{
			auto i1v = dynamic_cast<const SingleVariable*>(v1.get());
			auto i2v = dynamic_cast<const SingleVariable*>(v2.get());
			if (i1v != nullptr && i2v != nullptr) {
				copy_str +=
R"glsl(
	OUTPUT_NAME = INPUT_NAME[gl_GlobalInvocationID.x];
)glsl";
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

		{
			auto i1v = dynamic_cast<const SingleVariable*>(v1.get());
			auto i2v = dynamic_cast<const SingleVariable*>(v2.get());
			if (i1v != nullptr && i2v != nullptr) {
				copy_str +=
R"glsl(
	OUTPUT_NAME[gl_GlobalInvocationID.x] = INPUT_NAME;
)glsl";
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

	export enum class IOShaderVariableType
	{
		INPUT_TYPE = 1,
		OUTPUT_TYPE = 2,
		INPUT_OUTPUT_TYPE = INPUT_TYPE | OUTPUT_TYPE,
		CONST_TYPE = 4,
		LOCAL_TYPE = 8
	};

	export class AutogenShader : public ShaderBase, public ScopeBase {
	public:

		void addBinding(std::unique_ptr<Binding> binding)
		{
			_addBinding(std::move(binding));
		}

		// MATRIX
		void addMatrix(const std::shared_ptr<MatrixVariable>& mat, vc::ui16 binding, const IOShaderVariableType& type)
		{
			_addMatrix(mat, binding, type);
		}

		// VECTOR
		void addVector(const std::shared_ptr<VectorVariable>& vec, vc::ui16 binding, const IOShaderVariableType& type)
		{
			_addVector(vec, binding, type);
		}

		// SINGLE
		void addSingle(const std::shared_ptr<SingleVariable>& var, vc::ui16 binding, const IOShaderVariableType& type)
		{
			_addSingle(var, binding, type);
		}

		vc::ui16 scope_level() const override {
			return 0;
		}

		vc::ui16 addVariable(const std::shared_ptr<ShaderVariable>& var) override {
			return _addVariable(var, IOShaderVariableType::LOCAL_TYPE);
		}

		const std::shared_ptr<ShaderVariable>& getVariable(vc::ui16 index) const override {
			return m_Variables[index];
		}

		vc::ui16 addFunction(const std::shared_ptr<Function>& func) override {
			return _addFunction(func);
		}

		const std::shared_ptr<Function>& getFunction(vc::ui16 index) const override {
			return m_Functions[index];
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
			if (m_Bindings.size() > 0)
				ret += "\n";

			for (auto& func : m_Functions) {
				ret += func->getCode() + "\n";
			}
			if (m_Functions.size() > 0)
				ret += "\n";

			// open main
			ret += "void main() {\n";

			// declare variables
			for (int i = 0; i < m_Variables.size(); ++i) {
				if (m_VariableTypes[i] == IOShaderVariableType::LOCAL_TYPE)
					ret += "\t" + m_Variables[i]->getDeclaration();
			}
			if (m_Variables.size() > 0)
				ret += "\n";

			// manual insertions before copy from
			if (m_BeforeCopyingFrom != "")
				ret += m_BeforeCopyingFrom + "\n";

			// copy globals to locals
			if (m_Inputs.size() != 0 || m_Outputs.size() != 0)
				ret += "\tuint start_index;\n";
			for (auto& input : m_Inputs) {
				ret += copying_from(m_Variables[input.first], m_Variables[input.second]);
			}
			ret += "\n";

			// manual insertions after copy from
			if (m_AfterCopyingFrom != "")
				ret += m_AfterCopyingFrom + "\n";

			// add in function calls
			for (auto& child : m_Children) {
				ret += child->build();
			}
			if (m_Children.size() > 0)
				ret += "\n";

			// manual insertions before copying back
			if (m_BeforeCopyingBack != "")
				ret += m_BeforeCopyingBack + "\n";

			// copy locals back to globals
			for (auto& output : m_Outputs) {
				ret += copying_to(m_Variables[output.first], m_Variables[output.second]);
			}
			if (m_Outputs.size() > 0)
				ret += "\n";

			// manual insertions after copying back
			if (m_AfterCopyingBack != "")
				ret += m_AfterCopyingBack + "\n";

			// close main
			ret += "}\n";

			return ret;
		}

	protected:

		void on_chain() override {
			throw std::runtime_error("One cannot chain on a shader");
		}

		void on_build() const override {
			throw std::runtime_error("One shall not build on a shader, use compile instead");
		}

		std::string header() const override {
			throw std::runtime_error("header should not be called on a shader");
		}

	private:

		vc::ui16 _addFunction(const std::shared_ptr<Function>& func)
		{
			return Function::add_function(m_Functions, func);
		}

		bool _addBinding(std::unique_ptr<Binding> binding)
		{
			return Binding::add_binding(m_Bindings, std::move(binding));
		}

		vc::ui16 _addVariable(const std::shared_ptr<ShaderVariable>& var, IOShaderVariableType type)
		{
			auto it = std::find_if(m_Variables.begin(), m_Variables.end(), [&var](const std::shared_ptr<ShaderVariable>& v) {
				return *var == *v;
				});
			vc::ui16 pos = it - m_Variables.begin();
			if (it == m_Variables.end()) {
				m_VariableTypes.emplace_back(type);
				m_Variables.emplace_back(var);
			}
			return pos;
		}

		void _addMatrix(const std::shared_ptr<MatrixVariable>& mat, vc::ui16 binding, const IOShaderVariableType& type)
		{
			auto ndim1 = mat->getNDim1();
			auto ndim2 = mat->getNDim2();

			std::shared_ptr<MatrixVariable> global_var = std::make_shared<MatrixVariable>(
				"global_" + mat->getName(), ndim1, ndim2, mat->getType());

			if (static_cast<int>(type) & static_cast<int>(IOShaderVariableType::CONST_TYPE)) {
				_addBinding(std::make_unique<ConstBufferBinding>(binding, mat->getType(), mat->getName(), mat->getNDim1() * mat->getNDim2()));
				_addVariable(mat, IOShaderVariableType::CONST_TYPE);
				return;
			}

			bool added_global_var = false;
			vc::ui16 var_index = _addVariable(mat, IOShaderVariableType::LOCAL_TYPE);
			vc::ui16 global_var_index;

			if (static_cast<int>(type) & static_cast<int>(IOShaderVariableType::INPUT_TYPE)) {
				global_var_index = _addVariable(global_var, type);
				added_global_var = true;
				_addBinding(std::make_unique<BufferBinding>(binding, global_var->getType(), global_var->getName()));
				m_Inputs.emplace_back(global_var_index, var_index);
			}

			if (static_cast<int>(type) & static_cast<int>(IOShaderVariableType::OUTPUT_TYPE)) {
				if (!added_global_var) {
					global_var_index = _addVariable(global_var, type);
					added_global_var = true;
					_addBinding(std::make_unique<BufferBinding>(binding, global_var->getType(), global_var->getName()));
				}
				m_Outputs.emplace_back(var_index, global_var_index);
			}

		}

		void _addVector(const std::shared_ptr<VectorVariable>& vec, vc::ui16 binding, const IOShaderVariableType& type)
		{
			auto ndim = vec->getNDim();

			std::shared_ptr<VectorVariable> global_var = std::make_shared<VectorVariable>(
				"global_" + vec->getName(), ndim, vec->getType());

			if (static_cast<int>(type) & static_cast<int>(IOShaderVariableType::CONST_TYPE)) {
				_addBinding(std::make_unique<ConstBufferBinding>(binding, vec->getType(), vec->getName(), vec->getNDim()));
				_addVariable(vec, IOShaderVariableType::CONST_TYPE);
				return;
			}

			bool added_global_var = false;
			vc::ui16 var_index = _addVariable(vec, IOShaderVariableType::LOCAL_TYPE);
			vc::ui16 global_var_index;

			if (static_cast<int>(type) & static_cast<int>(IOShaderVariableType::INPUT_TYPE)) {
				global_var_index = _addVariable(global_var, type);
				added_global_var = true;
				_addBinding(std::make_unique<BufferBinding>(binding, global_var->getType(), global_var->getName()));
				m_Inputs.emplace_back(global_var_index, var_index);
			}

			if (static_cast<int>(type) & static_cast<int>(IOShaderVariableType::OUTPUT_TYPE)) {
				if (!added_global_var) {
					global_var_index = _addVariable(global_var, type);
					added_global_var = true;
					_addBinding(std::make_unique<BufferBinding>(binding, global_var->getType(), global_var->getName()));
				}
				m_Outputs.emplace_back(var_index, global_var_index);
			}
		}

		void _addSingle(const std::shared_ptr<SingleVariable>& var, vc::ui16 binding, const IOShaderVariableType& type)
		{
			std::shared_ptr<SingleVariable> global_var = std::make_shared<SingleVariable>(
				"global_" + var->getName(), var->getType(), var->getValue());

			if (static_cast<int>(type) & static_cast<int>(IOShaderVariableType::CONST_TYPE)) {
				_addBinding(std::make_unique<ConstBinding>(binding, var->getType(), var->getName()));
				_addVariable(var, IOShaderVariableType::CONST_TYPE);
				return;
			}

			bool added_global_var = false;
			vc::ui16 var_index = _addVariable(var, IOShaderVariableType::LOCAL_TYPE);
			vc::ui16 global_var_index;
			
			if (static_cast<int>(type) & static_cast<int>(IOShaderVariableType::INPUT_TYPE)) {
				global_var_index = _addVariable(global_var, type);
				added_global_var = true;
				_addBinding(std::make_unique<BufferBinding>(binding, global_var->getType(), global_var->getName()));
				m_Inputs.emplace_back(global_var_index, var_index);
			}

			if (static_cast<int>(type) & static_cast<int>(IOShaderVariableType::OUTPUT_TYPE)) {
				if (!added_global_var) {
					global_var_index = _addVariable(global_var, type);
					added_global_var = true;
					_addBinding(std::make_unique<BufferBinding>(binding, global_var->getType(), global_var->getName()));
				}
				m_Outputs.emplace_back(var_index, global_var_index);
			}

		}

	private:

		std::vector<std::unique_ptr<Binding>> m_Bindings;
		std::vector<std::shared_ptr<Function>> m_Functions;

		std::vector<IOShaderVariableType> m_VariableTypes;
		std::vector<std::shared_ptr<ShaderVariable>> m_Variables;

		std::string m_BeforeCopyingFrom;
		std::string m_AfterCopyingFrom;
		std::string m_BeforeCopyingBack;
		std::string m_AfterCopyingBack;

		std::vector<std::pair<vc::ui16, vc::ui16>> m_Inputs;
		std::vector<std::pair<vc::ui16, vc::ui16>> m_Outputs;

	};



}