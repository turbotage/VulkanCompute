module;

export module func_factory;

import <string>;
import <optional>;
import <vector>;
import <functional>;
import <unordered_set>;
import <set>;
import <type_traits>;
import <memory>;
import <stdexcept>;
import <tuple>;

import vc;
import util;

import variable;
import function;

namespace glsl {

	export class ScopeBase;

	export class FunctionScope {
	public:

		FunctionScope() = default;

		virtual vc::ui16 scope_level() const = 0;

		virtual std::vector<std::shared_ptr<ShaderVariable>>& variables() = 0;
		virtual const std::vector<std::shared_ptr<ShaderVariable>>& variables() const = 0;

		virtual std::vector<std::shared_ptr<Function>>& functions() = 0;
		virtual const std::vector<std::shared_ptr<Function>>& functions() const = 0;

		virtual std::string build() const = 0;

	protected:
		vc::raw_ptr<ScopeBase> m_Parent;

		friend class ScopeBase;
	};

	export class ScopeBase : public FunctionScope {
	protected:

		virtual void on_apply() {}
		virtual void on_apply_scope() {}
		virtual void on_chain() {}
		virtual void on_build() const {}

	public:

		ScopeBase() = default;

		static std::unique_ptr<ScopeBase> make()
		{
			return std::make_unique<ScopeBase>();
		}

		template<ShaderVariableIterator SVIterator>
		ScopeBase& apply(const std::shared_ptr<Function>& func,
			const std::shared_ptr<ShaderVariable>& ret,
			std::optional<std::pair<SVIterator, SVIterator>> args_it);

		ScopeBase& apply(const std::shared_ptr<Function>& func,
			const std::shared_ptr<ShaderVariable>& ret,
			const std::vector<std::shared_ptr<ShaderVariable>>& args)
		{
			return apply(func, ret, std::make_optional(std::make_pair(args.begin(), args.end())));
		}
		
		ScopeBase& apply(const FunctionApplier& applier)
		{
			return apply(applier.func, applier.ret_var, applier.args);
		}

		ScopeBase& apply_scope(std::unique_ptr<ScopeBase> scope)
		{
			on_apply_scope();

			scope->m_Parent = *this;
			ScopeBase& ret = *scope;
			m_Children.push_back(std::move(scope));
			return ret;
		}

		ScopeBase& chain(std::unique_ptr<ScopeBase> scope)
		{
			on_chain();

			if (m_ChainChild != nullptr)
				throw std::runtime_error("ScopeBase already had chained child?");

			scope->m_Parent = *this;
			ScopeBase& ret = *scope;
			m_ChainChild = std::move(scope);
			return ret;
		}

		virtual std::string header() const
		{
			return "";
		}

		virtual vc::ui16 scope_level() const override
		{
			return m_Parent->scope_level() + 1;
		}

		virtual std::vector<std::shared_ptr<ShaderVariable>>& variables() override
		{
			return m_Parent->variables();
		}

		virtual const std::vector<std::shared_ptr<ShaderVariable>>& variables() const override
		{
			return m_Parent->variables();
		}

		virtual std::vector<std::shared_ptr<Function>>& functions() override
		{
			return m_Parent->functions();
		}

		virtual const std::vector<std::shared_ptr<Function>>& functions() const override
		{
			return m_Parent->functions();
		}

		virtual std::string build() const override {

			on_build();

			std::string ret;
			ret += header() + " {\n";

			for (auto& child : m_Children) {
				ret += child->build();
			}
			util::add_n_str(ret, "\t", scope_level());
			ret += "}\n";
			if (m_ChainChild != nullptr)
				ret += m_ChainChild->build();

			return ret;
		}

	protected:
		std::vector<std::unique_ptr<FunctionScope>> m_Children;
		std::unique_ptr<FunctionScope> m_ChainChild;
	};
	
	export class TextedScope : public ScopeBase {
	public:

		TextedScope(const std::string& str)
			: m_Text(str)
		{}

		static std::unique_ptr<TextedScope> make(const std::string& str)
		{
			return std::make_unique<TextedScope>(str);
		}

		std::string build() const override {
			std::string ret = util::add_after_newline(m_Text, std::string(scope_level(), '\t'));
			if (ret.back() != '\n') {
				ret += "\n";
			}
			return ret;
		}

	protected:
		vc::raw_ptr<ScopeBase> m_Parent;

		std::string m_Text;

		friend class ScopeBase;
	};

	export class CallScope : public FunctionScope {
	public:

		CallScope(const std::tuple<vc::ui16, vc::i16, std::vector<vc::ui16>>& call)
			: m_Call(call) {}

		static std::unique_ptr<CallScope> make(const std::tuple<vc::ui16, vc::i16, std::vector<vc::ui16>>& call)
		{
			return std::make_unique<CallScope>(call);
		}

		vc::ui16 scope_level() const override
		{
			return m_Parent->scope_level() + 1;
		}

		virtual std::vector<std::shared_ptr<ShaderVariable>>& variables() override
		{
			return m_Parent->variables();
		}

		virtual const std::vector<std::shared_ptr<ShaderVariable>>& variables() const override
		{
			return m_Parent->variables();
		}


		virtual std::vector<std::shared_ptr<Function>>& functions() override
		{
			return m_Parent->functions();
		}

		virtual const std::vector<std::shared_ptr<Function>>& functions() const override
		{
			return m_Parent->functions();
		}

		std::string build() const override {
			std::string code_str;

			vc::ui16 slevel = scope_level();

			util::add_n_str(code_str, "\t", slevel);

			auto& func = (functions())[std::get<0>(m_Call)];

			auto ret_idx = std::get<1>(m_Call);
			if (ret_idx >= 0) {
				auto& ret_var = (variables())[ret_idx];
				code_str += ret_var->getName() + " = ";
			}

			code_str += func->getName() + "(";

			// add input variables
			auto& inputs = std::get<2>(m_Call);
			for (int i = 0; i < inputs.size(); ++i) {
				code_str += (variables())[inputs[i]]->getName();
				if (i < inputs.size() - 1) {
					code_str += ", ";
				}
			}
			code_str += ");\n";

			return code_str;
		}

	protected:

		std::tuple<vc::ui16, vc::i16, std::vector<vc::ui16>> m_Call;

	};

	export class IfScope : public ScopeBase {
	public:

		IfScope(const std::string& condition)
			: m_Condition(condition) {}

		static std::unique_ptr<IfScope> make(const std::string& condition)
		{
			return std::make_unique<IfScope>(condition);
		}

		std::string header() const override
		{
			std::string ret;
			util::add_n_str(ret, "\t", scope_level());
			ret += "if (" + m_Condition + ")";
			return ret;
		}

	private:
		std::string m_Condition;
	};

	export class ElseIfScope : public ScopeBase {
	public:

		ElseIfScope(const std::string& condition) 
			: m_Condition(condition) {}

		static std::unique_ptr<ElseIfScope> make(const std::string& condition)
		{
			return std::make_unique<ElseIfScope>(condition);
		}

		std::string header() const override
		{
			std::string ret;
			util::add_n_str(ret, "\t", scope_level());
			ret += "else if (" + m_Condition + ")";
			return ret;
		}

	private:
		std::string m_Condition;
	};

	export class ElseScope : public ScopeBase {
	public:

		ElseScope() {}

		static std::unique_ptr<ElseScope> make()
		{
			return std::make_unique<ElseScope>();
		}

		std::string header() const override
		{
			std::string ret;
			util::add_n_str(ret, "\t", scope_level());
			ret += "else";
			return ret;
		}

	};

	export class ForScope : public ScopeBase {
	public:

		ForScope(const std::string& loop_statement)
			: m_LoopStatement(loop_statement) {}

		static std::unique_ptr<ForScope> make(const std::string& loop_statement)
		{
			return std::make_unique<ForScope>(loop_statement);
		}

		std::string header() const override
		{
			std::string ret;
			util::add_n_str(ret, "\t", scope_level());
			ret += "for (" + m_LoopStatement + ")";
			return ret;
		}

	private:
		std::string m_LoopStatement;
	};
	
	export class FunctionFactory : public ScopeBase {
	protected:

		void on_build() const override
		{
			throw std::runtime_error("build should not be called on a FunctionFactory");
		}

		std::string header() const override
		{
			throw std::runtime_error("header should not be called on a FunctionFactory");
		}

	public:

		enum class InputType {
			IN,
			OUT,
			INOUT
		};

		FunctionFactory(const std::string& name, const ShaderVariableType& return_type)
			: m_Name(name), m_ReturnType(return_type) {}

		vc::ui16 scope_level() const override 
		{
			return 0;
		}

		std::vector<std::shared_ptr<ShaderVariable>>& variables() override
		{
			return m_Variables;
		}

		const std::vector<std::shared_ptr<ShaderVariable>>& variables() const override
		{
			return m_Variables;
		}

		std::vector<std::shared_ptr<Function>>& functions() override
		{
			return m_Functions;
		}

		const std::vector<std::shared_ptr<Function>>& functions() const override
		{
			return m_Functions;
		}

		void addMatrix(const std::shared_ptr<MatrixVariable>& mat, const std::optional<FunctionFactory::InputType>& input)
		{
			auto varidx = ShaderVariable::add_variable(m_Variables, mat);

			if (input.has_value()) {
				m_Inputs.emplace_back(varidx, input.value());
			}
		}

		void addVector(const std::shared_ptr<VectorVariable>& vec, const std::optional<FunctionFactory::InputType>& input)
		{
			auto varidx = ShaderVariable::add_variable(m_Variables, vec);
			if (input.has_value()) {
				m_Inputs.emplace_back(varidx, input.value());
			}
		}

		void addSingle(const std::shared_ptr<SingleVariable>& var, const std::optional<FunctionFactory::InputType>& input)
		{
			auto varidx = ShaderVariable::add_variable(m_Variables, var);
			if (input.has_value()) {
				m_Inputs.emplace_back(varidx, input.value());
			}
		}

		std::shared_ptr<::glsl::Function> build_function() const
		{
			std::string code_str;

			std::string uniqueid;
			std::vector<size_t> hashes;
			std::vector<std::shared_ptr<Function>> dependencies;

			// create code str
			{
				code_str += glsl::shader_variable_type_to_str(m_ReturnType) + " " + m_Name + "_";

				// add uniqueid
				hashes.reserve(m_Inputs.size());
				for (auto& input : m_Inputs) {
					hashes.emplace_back(m_Variables[input.first]->getHash());
				}
				uniqueid = util::stupid_compress(util::hash_combine(hashes));
				code_str += uniqueid + "(";

				// add inputs
				std::set<vc::ui16> input_idxs;
				for (int i = 0; i < m_Inputs.size(); ++i) {
					auto& input = m_Inputs[i];

					input_idxs.insert(input.first);

					switch (input.second) {
					case FunctionFactory::InputType::IN:
						code_str += "in ";
						break;
					case FunctionFactory::InputType::OUT:
						code_str += "out ";
						break;
					case FunctionFactory::InputType::INOUT:
						code_str += "inout ";
						break;
					default:
						throw std::runtime_error("Unsupported InputType");
					}

					code_str += m_Variables[input.first]->getInputDeclaration();
					if (i < m_Inputs.size() - 1) {
						code_str += ", ";
					}
				}
				code_str += ") {\n";

				// declare non input variables
				for (int i = 0; i < m_Variables.size(); ++i) {
					auto& var = m_Variables[i];
					if (!input_idxs.contains(i)) {
						code_str += "\t" + var->getDeclaration();
					}
				}
				code_str += "\n";

				// declare scope calls
				for (auto& child : m_Children)
				{
					code_str += child->build();
				}

				code_str += "\n}";
			}

			return std::make_shared<Function>(
				m_Name + "_" + uniqueid,
				hashes,
				[code_str]() {
					return code_str;
				},
				dependencies.size() > 0 ? std::make_optional(dependencies) : std::nullopt);
		}

		FunctionApplier build_applier() const 
		{
			std::string code_str;

			std::string uniqueid;
			std::vector<size_t> hashes;
			std::vector<std::shared_ptr<Function>> dependencies;
			dependencies.reserve(m_Functions.size());

			// dependencies
			for (auto& func : m_Functions) {
				dependencies.emplace_back(func);
			}

			// create code str
			{
				code_str += glsl::shader_variable_type_to_str(m_ReturnType) + " " + m_Name + "_";

				// add uniqueid
				hashes.reserve(m_Inputs.size());
				for (auto& input : m_Inputs) {
					hashes.emplace_back(m_Variables[input.first]->getHash());
				}
				uniqueid = util::stupid_compress(util::hash_combine(hashes));
				code_str += uniqueid + "(";

				// add inputs
				std::set<vc::ui16> input_idxs;
				for (int i = 0; i < m_Inputs.size(); ++i) {
					auto& input = m_Inputs[i];

					input_idxs.insert(input.first);

					switch (input.second) {
					case FunctionFactory::InputType::IN:
						code_str += "in ";
						break;
					case FunctionFactory::InputType::OUT:
						code_str += "out ";
						break;
					case FunctionFactory::InputType::INOUT:
						code_str += "inout ";
						break;
					default:
						throw std::runtime_error("Unsupported InputType");
					}

					code_str += m_Variables[input.first]->getInputDeclaration();
					if (i < m_Inputs.size() - 1) {
						code_str += ", ";
					}
				}
				code_str += ") {\n";

				// declare non input variables
				for (int i = 0; i < m_Variables.size(); ++i) {
					auto& var = m_Variables[i];
					if (!input_idxs.contains(i)) {
						code_str += "\t" + var->getDeclaration();
					}
				}
				code_str += "\n";

				// declare scope calls
				for (auto& child : m_Children)
				{
					code_str += child->build();
				}

				code_str += "\n}";
			}

			auto ret_func = std::make_shared<Function>(
				m_Name + "_" + uniqueid,
				hashes,
				[code_str]() {
					return code_str;
				},
				dependencies.size() > 0 ? std::make_optional(dependencies) : std::nullopt);

			std::vector<std::shared_ptr<ShaderVariable>> input_vars;
			input_vars.reserve(m_Inputs.size());
			for (auto& inp : m_Inputs) {
				input_vars.push_back(m_Variables[inp.first]);
			}

			return FunctionApplier{ ret_func, nullptr, input_vars, uniqueid };

		}

	private:

		std::string m_Name;
		ShaderVariableType m_ReturnType;

		std::vector<std::pair<vc::ui16, FunctionFactory::InputType>> m_Inputs;
		std::vector<std::shared_ptr<ShaderVariable>> m_Variables;

		std::vector<std::shared_ptr<Function>> m_Functions;
	};

	// IMPLEMENTATIONS

	template<ShaderVariableIterator SVIterator>
	ScopeBase& ScopeBase::apply(const std::shared_ptr<Function>& func, const std::shared_ptr<ShaderVariable>& ret, std::optional<std::pair<SVIterator, SVIterator>> args_it)
	{
		{
			on_apply();

			vc::ui16 func_pos = Function::add_function(functions(), func);

			vc::i16 ret_pos = -1;
			if (ret) {
				ret_pos = ShaderVariable::add_variable(variables(), ret);
			}

			std::vector<vc::ui16> input_pos;
			if (args_it.has_value()) {
				auto& its = args_it.value();
				size_t ninputs = std::distance(its.first, its.second);
				input_pos.reserve(ninputs);
				for (auto it = its.first; it != its.second; ++it) {
					input_pos.emplace_back(ShaderVariable::add_variable(variables(), *it));
				}
			}

			auto call = CallScope::make(std::make_tuple(func_pos, ret_pos, std::move(input_pos)));
			call->m_Parent = *this;

			m_Children.push_back(std::move(call));

			return *this;
		}
	}

}