module;

#include <string>
#include <optional>
#include <vector>
#include <functional>
#include <unordered_set>
#include <fstream>
#include <set>
#include <type_traits>

#include <kompute/Kompute.hpp>

export module glsl;

import util;
import vc;

namespace glsl {

	export std::vector<uint32_t> compileSource(const std::string& source);

	export class Function {
	public:

		Function(
			const std::string& function_name,
			const std::vector<size_t>& argument_hashes,
			const std::function<std::string()>& code_func,
			std::optional<std::vector<Function>> dependencies);

		const std::vector<Function>& getDependencies() const;

		std::string getCode() const;

		std::string getName() const;

		friend bool operator==(const Function& lhs, const Function& rhs);

		struct HashFunction {
			std::size_t operator()(const Function& func) const {
				std::size_t hash = util::hash_combine(func.m_ArgumentHashes);
				return util::hash_combine(hash, func.m_FunctionName);
			}
		};

		static size_t add_functions(std::vector<Function>& funcs, const Function& func);

	private:
		
		std::string m_FunctionName;
		std::vector<std::size_t> m_ArgumentHashes;
		std::function<std::string()> m_CodeFunc;
		std::vector<Function> m_Dependencies;

		std::string m_HashName;
	};

	export class Binding {
	public:

		virtual std::string operator()() const = 0;

		virtual bool operator==(const Binding* other) const = 0;

	};

	export class BufferBinding : public Binding {
	public:

		BufferBinding(uint16_t binding, std::string type, std::string name)
			: m_Binding(binding), m_Type(type), m_Name(name) {}

		std::string operator()() const override {
			return "layout(set = 0, binding = " + std::to_string(m_Binding) + ") buffer buf_global_" +
				m_Name + " { " + m_Type + " " + "global_" + m_Name + "[]; };";
		}

		bool operator==(const Binding* other) const override {
			if (auto* b = dynamic_cast<const BufferBinding*>(other); b != nullptr) {
				return (b->m_Binding == m_Binding) && 
					(b->m_Type == m_Type) && (b->m_Name == m_Name);
			}
			return false;
		}

	private:
		uint16_t m_Binding;
		std::string m_Type;
		std::string m_Name;
	};

	export class ShaderVariable {
	public:

		virtual std::string getDeclaration() const = 0;

		virtual std::string getName() const = 0;

		friend bool operator==(const ShaderVariable& left, const ShaderVariable& right) {
			return left.getName() == right.getName();
		}

	};

	export template<typename T>
	concept ShaderVariableIterator = std::is_same_v<std::shared_ptr<ShaderVariable>, typename std::iterator_traits<T>::value_type>;

	export class MatrixVariable : public ShaderVariable {
	public:

		MatrixVariable(const std::string& name,
			uint16_t ndim1, uint16_t ndim2,
			bool single_precission = true);

		std::string getDeclaration() const override;
		
		std::string getName() const override;

		std::pair<uint16_t, uint16_t> getDimensions() const;

		uint16_t getNDim1() const;

		uint16_t getNDim2() const;

		bool isSinglePrecission() const;

	private:
		std::string m_Name;
		uint16_t m_NDim1;
		uint16_t m_NDim2;
		bool m_SinglePrecission;
	};

	export class VectorVariable : public ShaderVariable {
	public:

		VectorVariable(const std::string& name,
			uint16_t ndim, bool single_precission = true)
			: m_Name(name), m_NDim(ndim), m_SinglePrecission(single_precission)
		{}

		std::string getDeclaration() const override;

		std::string getName() const override;

		uint16_t getDimension() const;

		bool isSinglePrecission() const;

	private:
		std::string m_Name;
		uint16_t m_NDim;
		bool m_SinglePrecission;
	};

	export class Shader {
	public:

		void addFunction(const Function& func);

		void addInputMatrix(const std::shared_ptr<MatrixVariable>& mat, uint16_t binding);

		void addVariable(const std::shared_ptr<ShaderVariable>& var);

		template<ShaderVariableIterator SVIterator> 
		void apply(const Function& func, SVIterator begin, SVIterator end)
		{
			size_t func_pos = Function::add_functions(m_Functions, func);

			m_Calls.emplace_back(func_pos, {});
			auto& back = m_Calls.back();

			for (auto it = begin; it != end; ++it) {
				back.second.emplace_back(*it);
			}
		}

		void apply(const Function& func, const std::vector<std::shared_ptr<ShaderVariable>>& vars);

		std::string compile() const;

	private:

		bool addBinding(std::unique_ptr<Binding> binding);

	private:

		std::vector<std::unique_ptr<Binding>> m_Bindings;
		std::vector<Function> m_Functions;

		std::vector<std::shared_ptr<ShaderVariable>> m_Variables;
		std::vector<std::pair<size_t, std::vector<std::shared_ptr<ShaderVariable>>>> m_Calls;
	};

	export enum class eSymbolicType {
		CONST_TYPE,
		PARAM_TYPE,
	};

	export class SymbolicContext {
	public:

		void insert_const(const std::pair<std::string, uint32_t>& cp);

		void insert_param(const std::pair<std::string, uint32_t>& pp);

		eSymbolicType get_symtype(const std::string& name) const;

		uint32_t get_params_index(const std::string& name) const;

		const std::string& get_params_name(size_t index) const;

		uint32_t get_consts_index(const std::string& name) const;

		const std::string& get_consts_name(size_t index) const;

		const std::string& get_consts_name() const;

		const std::string& get_consts_iterable_by() const;

		const std::string& get_params_iterable_by() const;

		std::string get_glsl_var_name(const std::string& name) const;

		std::set<std::pair<std::string, uint32_t>> params_map;
		std::set<std::pair<std::string, uint32_t>> consts_map;

		std::unordered_map<std::string, eSymbolicType> symtype_map;

		std::string params_name = "params";
		std::string consts_name = "consts";

		std::string consts_iterable_by = "i";
		std::string params_iterable_by = "i";

		std::string ndata_name = "ndata";
		std::string nconst_name = "nconst";
	};

}

