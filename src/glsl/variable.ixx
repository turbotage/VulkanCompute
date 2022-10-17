module;

export module variable;

import <string>;
import <optional>;
import <vector>;
import <functional>;
import <unordered_set>;
import <set>;
import <type_traits>;
import <memory>;
import <stdexcept>;

import vc;
import util;

namespace glsl {

	export enum class ShaderVariableType {
		FLOAT = 1,
		DOUBLE = 2,
		INT = 4,
		VOID = 8,
	};

	export std::string shader_variable_type_to_str(const ShaderVariableType& type)
	{
		switch (type) {
		case ShaderVariableType::FLOAT:
			return "float";
		case ShaderVariableType::DOUBLE:
			return "double";
		case ShaderVariableType::INT:
			return "int";
		case ShaderVariableType::VOID:
			return "void";
		default:
			throw std::runtime_error("Unsupported type");
		}
	}

	export class Binding {
	public:

		virtual std::string operator()() const = 0;

		virtual bool operator==(const Binding* other) const = 0;

		static size_t add_binding(std::vector<std::unique_ptr<Binding>>& vars,
			std::unique_ptr<Binding> binding)
		{
			auto it = std::find_if(vars.begin(), vars.end(), [&binding](const std::unique_ptr<Binding>& b)
				{
					return binding->operator==(b.get());
				});

			if (it == vars.end()) {
				vars.emplace_back(std::move(binding));
				return true;
			}
			return false;
		}

	};

	export class ConstBinding : public Binding {
	public:

		ConstBinding(vc::ui16 binding, const std::string& type, const std::string& name)
			: m_Binding(binding), m_Type(type), m_Name(name) {}

		ConstBinding(vc::ui16 binding, const ShaderVariableType& type, const std::string& name)
			: m_Binding(binding), m_Type(shader_variable_type_to_str(type)), m_Name(name) {}

		std::string operator()() const override
		{
			return "layout(set = 0, binding = " + std::to_string(m_Binding) + ") buffer buf_" +
				m_Name + " { " + m_Type + " " + m_Name + "; };";
		}

		bool operator==(const Binding* other) const override
		{
			if (auto* b = dynamic_cast<const ConstBinding*>(other); b != nullptr) {
				return (b->m_Binding == m_Binding) &&
					(b->m_Type == m_Type) && (b->m_Name == m_Name);
			}
			return false;
		}

	private:
		vc::ui16 m_Binding;
		std::string m_Type;
		std::string m_Name;
	};

	export class BufferBinding : public Binding {
	public:

		BufferBinding(vc::ui16 binding, const std::string& type, const std::string& name)
			: m_Binding(binding), m_Type(type), m_Name(name) {}

		BufferBinding(vc::ui16 binding, const ShaderVariableType& type, const std::string& name)
			: m_Binding(binding), m_Type(shader_variable_type_to_str(type)), m_Name(name) {}

		std::string operator()() const override
		{
			return "layout(set = 0, binding = " + std::to_string(m_Binding) + ") buffer buf_" +
				m_Name + " { " + m_Type + " " + m_Name + "[]; };";
		}

		bool operator==(const Binding* other) const override
		{
			if (auto* b = dynamic_cast<const BufferBinding*>(other); b != nullptr) {
				return (b->m_Binding == m_Binding) &&
					(b->m_Type == m_Type) && (b->m_Name == m_Name);
			}
			return false;
		}

	private:
		vc::ui16 m_Binding;
		std::string m_Type;
		std::string m_Name;
	};

	export class ConstBufferBinding : public Binding {
	public:

		ConstBufferBinding(vc::ui16 binding, const std::string& type, const std::string& name, vc::ui16 nelem)
			: m_Binding(binding), m_Type(type), m_Name(name), m_NElem(nelem) {}

		ConstBufferBinding(vc::ui16 binding, const ShaderVariableType& type, const std::string& name, vc::ui16 nelem)
			: m_Binding(binding), m_Type(shader_variable_type_to_str(type)), m_Name(name), m_NElem(nelem) {}

		std::string operator()() const override
		{
			return "layout(set = 0, binding = " + std::to_string(m_Binding) + ") buffer buf_" +
				m_Name + " { " + m_Type + " " + m_Name + "[" + std::to_string(m_NElem) + "]; };";
		}

		bool operator==(const Binding* other) const override
		{
			if (auto* b = dynamic_cast<const ConstBufferBinding*>(other); b != nullptr) {
				return (b->m_Binding == m_Binding) &&
					(b->m_Type == m_Type) && (b->m_Name == m_Name) && (b->m_NElem == m_NElem);
			}
			return false;
		}

	private:
		vc::ui16 m_Binding;
		std::string m_Type;
		std::string m_Name;
		vc::ui16 m_NElem;
	};

	export class ShaderVariable {
	public:

		virtual std::string getInputDeclaration() const = 0;

		virtual std::string getDeclaration() const = 0;

		virtual std::string getName() const = 0;

		virtual std::string getUniqueID() const = 0;

		virtual size_t getHash() const = 0;

		friend bool operator==(const ShaderVariable& left, const ShaderVariable& right) {
			return left.getName() == right.getName();
		}

		static size_t add_variable(std::vector<std::shared_ptr<ShaderVariable>>& vars,
			const std::shared_ptr<ShaderVariable>& var)
		{
			auto it = std::find_if(vars.begin(), vars.end(), [&var](const std::shared_ptr<ShaderVariable>& v) {
				return *v == *var;
				});

			size_t pos = it - vars.begin();
			if (it == vars.end()) {
				vars.emplace_back(var);
			}
			return pos;
		}

	};

	export template<typename T>
		concept ShaderVariableIterator = std::is_same_v<std::shared_ptr<ShaderVariable>, typename std::iterator_traits<T>::value_type>;

	export class TextedVariable : public ShaderVariable {
	public:

		TextedVariable(
			const std::string& name,
			const std::string& type,
			const std::string& value)
			: m_Name(name), m_Type(type), m_Value(value)
		{}

		std::string getInputDeclaration() const override
		{
			return m_Type + " " + m_Name;
		}

		std::string getDeclaration() const override
		{
			return getInputDeclaration() + " = " + m_Value + ";\n";
		}

		std::string getName() const override
		{
			return m_Name;
		}

		std::string getUniqueID() const override
		{
			return m_Type;
		}

		size_t getHash() const override
		{
			return std::hash<std::string>()(m_Type);
		}

	private:
		std::string m_Name;
		std::string m_Type;
		std::string m_Value;
	};

	export class SingleVariable : public ShaderVariable {
	public:

		SingleVariable(const std::string& name, const ShaderVariableType& type,
			const std::optional<std::string>& value)
			: m_Name(name), m_Type(type), m_Value(value)
		{}

		std::string getInputDeclaration() const override
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

			return ret;
		}

		std::string getDeclaration() const override
		{
			std::string ret = getInputDeclaration();
			if (m_Value.has_value()) {
				ret += " = " + m_Value.value() + ";\n";
			}
			else {
				ret += ";\n";
			}
			return ret;
		}

		std::string getName() const override
		{
			return m_Name;
		}

		std::string getUniqueID() const override
		{
			return glsl::shader_variable_type_to_str(m_Type);
		}

		std::optional<std::string> getValue() const {
			return m_Value;
		}

		size_t getHash() const override
		{
			return (size_t)m_Type;
		}

		ShaderVariableType getType() const
		{
			return m_Type;
		}

	private:
		std::string m_Name;
		ShaderVariableType m_Type;
		std::optional<std::string> m_Value;
	};

	export class VectorVariable : public ShaderVariable {
	public:

		VectorVariable(const std::string& name,
			vc::ui16 ndim, const ShaderVariableType& type)
			: m_Name(name), m_NDim(ndim), m_Type(type)
		{}

		std::string getInputDeclaration() const override
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
			ret += m_Name + "[" + std::to_string(m_NDim) + "]";
			return ret;
		}

		std::string getDeclaration() const override
		{
			return getInputDeclaration() + ";\n";
		}

		std::string getName() const override
		{
			return m_Name;
		}

		std::string getUniqueID() const override
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

		size_t getHash() const override
		{
			std::vector<size_t> hashes(2);
			hashes[0] = (size_t)m_NDim;
			hashes[1] = (size_t)m_Type;
			return util::hash_combine(hashes);
		}


		uint16_t getNDim() const
		{
			return m_NDim;
		}

		ShaderVariableType getType() const
		{
			return m_Type;
		}

	private:
		std::string m_Name;
		uint16_t m_NDim;
		ShaderVariableType m_Type;
	};

	export class MatrixVariable : public ShaderVariable {
	public:

		MatrixVariable(const std::string& name,
			vc::ui16 ndim1, vc::ui16 ndim2,
			const ShaderVariableType& type)
			: m_Name(name), m_NDim1(ndim1), m_NDim2(ndim2), m_Type(type)
		{}

		std::string getInputDeclaration() const override
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
				std::to_string(m_NDim2) + "]";
			return ret;
		}

		std::string getDeclaration() const override
		{
			return getInputDeclaration() + ";\n";
		}

		std::string getName() const override
		{
			return m_Name;
		}

		std::string getUniqueID() const override
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

		size_t getHash() const override
		{
			std::vector<size_t> hashes(3);
			hashes[0] = (size_t)m_NDim1;
			hashes[1] = (size_t)m_NDim2;
			hashes[2] = (size_t)m_Type;
			return util::hash_combine(hashes);
		}

		std::pair<vc::ui16, vc::ui16> getDimensions() const
		{
			return std::make_pair(m_NDim1, m_NDim2);
		}

		vc::ui16 getNDim1() const { return m_NDim1; }

		vc::ui16 getNDim2() const { return m_NDim2; }

		bool isSquare() const { return getNDim1() == getNDim2(); }

		ShaderVariableType getType() const
		{
			return m_Type;
		}

	private:
		std::string m_Name;
		vc::ui16 m_NDim1;
		vc::ui16 m_NDim2;
		ShaderVariableType m_Type;
	};

}