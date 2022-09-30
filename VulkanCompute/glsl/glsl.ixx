module;

#include <kompute/Kompute.hpp>

export module glsl;

import <string>;
import <optional>;
import <vector>;
import <functional>;
import <unordered_set>;
import <set>;
import <type_traits>;

import util;
import vc;

using namespace vc;

namespace glsl {

	export constexpr auto UNIQUE_ID = "UNIQUEID";

	export std::vector<ui32> compileSource(const std::string& source, bool optimize = true);

	export enum class ShaderVariableType {
		FLOAT,
		DOUBLE,
		INT,
	};

	export std::string shader_variable_type_to_str(const ShaderVariableType& type);
	
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

		static size_t add_function(std::vector<Function>& funcs, const Function& func);

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

		BufferBinding(ui16 binding, const std::string& type, const std::string& name)
			: m_Binding(binding), m_Type(type), m_Name(name) {}

		BufferBinding(ui16 binding, const ShaderVariableType& type, const std::string& name)
			: m_Binding(binding), m_Type(shader_variable_type_to_str(type)), m_Name(name) {}

		std::string operator()() const override;

		bool operator==(const Binding* other) const override;

	private:
		ui16 m_Binding;
		std::string m_Type;
		std::string m_Name;
	};

	export class ShaderVariable {
	public:

		virtual std::string getDeclaration() const = 0;

		virtual std::string getName() const = 0;

		virtual std::string getUniqueID() const = 0;

		virtual size_t getHash() const = 0;

		friend bool operator==(const ShaderVariable& left, const ShaderVariable& right) {
			return left.getName() == right.getName();
		}

		static size_t add_variable(std::vector<std::shared_ptr<ShaderVariable>>& vars, 
			const std::shared_ptr<ShaderVariable>& var);

	};

	export template<typename T>
	concept ShaderVariableIterator = std::is_same_v<std::shared_ptr<ShaderVariable>, typename std::iterator_traits<T>::value_type>;

	export class MatrixVariable : public ShaderVariable {
	public:

		MatrixVariable(const std::string& name,
			ui16 ndim1, ui16 ndim2,
			const ShaderVariableType& type);

		std::string getDeclaration() const override;
		
		std::string getName() const override;

		std::string getUniqueID() const override;

		size_t getHash() const override;

		std::pair<ui16, ui16> getDimensions() const;

		ui16 getNDim1() const;

		ui16 getNDim2() const;

		ShaderVariableType getType() const;

	private:
		std::string m_Name;
		ui16 m_NDim1;
		ui16 m_NDim2;
		ShaderVariableType m_Type;
	};

	export class VectorVariable : public ShaderVariable {
	public:

		VectorVariable(const std::string& name,
			ui16 ndim, const ShaderVariableType& type)
			: m_Name(name), m_NDim(ndim), m_Type(type)
		{}

		std::string getDeclaration() const override;

		std::string getName() const override;

		std::string getUniqueID() const override;

		size_t getHash() const override;

		uint16_t getNDim() const;

		ShaderVariableType getType() const;

	private:
		std::string m_Name;
		uint16_t m_NDim;
		ShaderVariableType m_Type;
	};

	export class SingleVariable : public ShaderVariable {
	public:

		SingleVariable(const std::string& name, const ShaderVariableType& type,
			const std::optional<std::string>& value);

		std::string getDeclaration() const override;

		std::string getName() const override;

		std::string getUniqueID() const override;

		size_t getHash() const override;

		ShaderVariableType getType() const;

	private:
		std::string m_Name;
		ShaderVariableType m_Type;
		std::optional<std::string> m_Value;
	};

	export class TextedVariable : public ShaderVariable {
	public:

		TextedVariable(
			const std::string& name,
			const std::string& type,
			const std::string& value);

		std::string getDeclaration() const override;

		std::string getName() const override;

		std::string getUniqueID() const override;

		size_t getHash() const override;

	private:
		std::string m_Name;
		std::string m_Type;
		std::string m_Value;
	};

	export class FunctionFactory {
	public:

		enum class InputType {
			IN,
			OUT,
			INOUT
		};

		FunctionFactory(const std::string& name, const ShaderVariableType& return_type);

		void addMatrix(const std::shared_ptr<MatrixVariable>& mat, const std::optional<FunctionFactory::InputType>& input);

		void addVector(const std::shared_ptr<VectorVariable>& vec, const std::optional<FunctionFactory::InputType>& input);

		void addSingle(const std::shared_ptr<SingleVariable>& var, const std::optional<FunctionFactory::InputType>& input);

		template<ShaderVariableIterator SVIterator>
		void apply(const Function& func,
			const std::shared_ptr<ShaderVariable>& ret,
			std::optional<std::pair<SVIterator, SVIterator>> args_it)
		{
			uint16_t func_pos = Function::add_function(m_Functions, func);

			int16_t ret_pos = -1;
			if (ret) {
				ret_pos = _addVariable(ret);
			}

			std::vector<uint16_t> input_pos;
			if (args_it.has_value()) {
				auto& its = args_it.value();
				size_t ninputs = std::distance(its.first, its.second);
				input_pos.reserve(ninputs);
				for (auto it = its.first; it != its.second; ++it) {
					input_pos.emplace_back(_addVariable(*it));
				}
			}

			m_Calls.emplace_back(func_pos, ret_pos, std::move(input_pos));
		}

		void apply(const Function& func,
			const std::shared_ptr<ShaderVariable>& ret,
			const std::vector<std::shared_ptr<ShaderVariable>>& args)
		{
			apply(func, ret, std::make_optional(std::make_pair(args.begin(), args.end())));
		}

		::glsl::Function build();

	private:

		ui16 _addFunction(const Function& func);

		ui16 _addVariable(const std::shared_ptr<ShaderVariable>& var);

	private:

		std::string m_Name;
		ShaderVariableType m_ReturnType;

		std::vector<std::string> m_Inputs;
		std::vector<std::shared_ptr<ShaderVariable>> m_Variables;

		std::vector<Function> m_Functions;

		std::vector<std::tuple<ui16, ui16, std::vector<ui16>>> m_Calls;
	};

	export class ShaderBase {
	public:

		virtual std::string compile() const = 0;

	private:

	};

	export class StupidShader : public ShaderBase {
	public:

		StupidShader(const std::string& str)
			: m_Str(str) {}

		std::string compile() const override;

	private:
		std::string m_Str;
	};

	export class AutogenShader : public ShaderBase {
	public:

		void addBinding(std::unique_ptr<Binding> binding);

		void addFunction(const Function& func);

		void addInputMatrix(const std::shared_ptr<MatrixVariable>& mat, ui16 binding);

		void addOutputMatrix(const std::shared_ptr<MatrixVariable>& mat, ui16 binding);

		void addInputOutputMatrix(const std::shared_ptr<MatrixVariable>& mat, ui16 binding);

		void addInputVector(const std::shared_ptr<VectorVariable>& vec, ui16 binding);

		void addOutputVector(const std::shared_ptr<VectorVariable>& vec, ui16 binding);

		void addInputOutputVector(const std::shared_ptr<VectorVariable>& vec, ui16 binding);

		void addVariable(const std::shared_ptr<ShaderVariable>& var);

		void setBeforeCopyingFrom(const std::string& mi);

		void setAfterCopyingFrom(const std::string& mi);

		void setBeforeCopyingBack(const std::string& mi);

		void setAfterCopyingBack(const std::string& mi);

		template<ShaderVariableIterator SVIterator> 
		void apply(const Function& func, 
			const std::shared_ptr<ShaderVariable>& ret, 
			std::optional<std::pair<SVIterator,SVIterator>> args_it)
		{
			uint16_t func_pos = Function::add_function(m_Functions, func);

			int16_t ret_pos = -1;
			if (ret) {
				ret_pos = _addVariable(ret, false);
			}

			std::vector<uint16_t> input_pos;
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

		void apply(const Function& func,
			const std::shared_ptr<ShaderVariable>& ret,
			const std::vector<std::shared_ptr<ShaderVariable>>& args)
		{
			apply(func, ret, std::make_optional(std::make_pair(args.begin(), args.end())));
		}

		std::string compile() const override;

	private:

		uint16_t _addFunction(const Function& func);

		bool _addBinding(std::unique_ptr<Binding> binding);

		uint16_t _addVariable(const std::shared_ptr<ShaderVariable>& var, bool is_global);

		void _addInputMatrix(const std::shared_ptr<MatrixVariable>& mat, ui16 binding, bool add_binding);

		void _addOutputMatrix(const std::shared_ptr<MatrixVariable>& mat, ui16 binding, bool add_binding);
		
		void _addInputVector(const std::shared_ptr<VectorVariable>& vec, ui16 binding, bool add_binding);

		void _addOutputVector(const std::shared_ptr<VectorVariable>& vec, ui16 binding, bool add_binding);

	private:

		std::vector<std::unique_ptr<Binding>> m_Bindings;
		std::vector<Function> m_Functions;

		std::vector<std::pair<std::shared_ptr<ShaderVariable>,bool>> m_Variables;

		std::string m_BeforeCopyingFrom;
		std::string m_AfterCopyingFrom;
		std::string m_BeforeCopyingBack;
		std::string m_AfterCopyingBack;

		std::vector<std::pair<ui16, ui16>> m_Inputs;
		std::vector<std::pair<ui16, ui16>> m_Outputs;
		// m_Calls[0] is index to function in m_Functions
		// m_Calls[i] for i > 0 is variable to use in function call, index is to
		// m_Variables
		std::vector<std::tuple<ui16, ui16, std::vector<ui16>>> m_Calls;

	};

	export enum class eSymbolicType {
		CONST_TYPE,
		PARAM_TYPE,
	};

	export class SymbolicContext {
	public:

		void insert_const(const std::pair<std::string, ui16>& cp);

		void insert_param(const std::pair<std::string, ui16>& pp);

		eSymbolicType get_symtype(const std::string& name) const;

		ui16 get_params_index(const std::string& name) const;

		const std::string& get_params_name(size_t index) const;

		ui16 get_consts_index(const std::string& name) const;

		const std::string& get_consts_name(size_t index) const;

		const std::string& get_consts_name() const;

		const std::string& get_consts_iterable_by() const;

		const std::string& get_params_iterable_by() const;

		std::string get_glsl_var_name(const std::string& name) const;

		std::set<std::pair<std::string, ui16>> params_map;
		std::set<std::pair<std::string, ui16>> consts_map;

		std::unordered_map<std::string, eSymbolicType> symtype_map;

		std::string params_name = "params";
		std::string consts_name = "consts";

		std::string consts_iterable_by = "i";
		std::string params_iterable_by = "i";

		std::string ndata_name = "ndata";
		std::string nconst_name = "nconst";
	};

}

