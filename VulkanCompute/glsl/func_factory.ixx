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

	export class FunctionScope {
	public:

		FunctionScope(const std::weak_ptr<FunctionScope>& parent_scope)
			: parent(parent_scope) {}

		virtual vc::ui16 scope_level() const = 0;

		virtual std::vector<std::shared_ptr<ShaderVariable>>& variables() = 0;

		virtual std::vector<std::shared_ptr<Function>>& functions() = 0;

		virtual std::string build() = 0;

		std::weak_ptr<FunctionScope> parent;
		std::vector<std::shared_ptr<FunctionScope>> children;
		std::shared_ptr<FunctionScope> chain_child;
	};

	export class CallScope : public FunctionScope {
	public:

		CallScope(const std::weak_ptr<FunctionScope>& parent_scope, const std::tuple<vc::ui16, vc::ui16, std::vector<vc::ui16>>& call)
			: FunctionScope(!parent_scope.expired() ? parent_scope : throw std::runtime_error("CallScope was called with nullptr parent")),
			call(call)
		{}

		vc::ui16 scope_level() const override
		{
			return parent.lock()->scope_level();
		}

		std::vector<std::shared_ptr<ShaderVariable>>& variables() override
		{
			return parent.lock()->variables();
		}

		std::vector<std::shared_ptr<Function>>& functions() override
		{
			return parent.lock()->functions();
		}

		std::string build() override {
			std::string code_str;

			auto pparent = parent.lock();

			vc::ui16 scope_level = pparent->scope_level();

			auto add_scope = [scope_level](std::string& scoped_str) {
				for (int i = 0; i < scope_level; ++i) {
					scoped_str += "\t";
				}
			};

			add_scope(code_str);

			auto& func = pparent->functions()[std::get<0>(call)];

			auto ret_idx = std::get<1>(call);
			if (ret_idx >= 0) {
				auto& ret_var = pparent->variables()[ret_idx];
				code_str += ret_var->getName() + " = ";
			}

			code_str += func->getName() + "(";

			// add input variables
			auto& inputs = std::get<2>(call);
			for (int i = 0; i < inputs.size(); ++i) {
				code_str += pparent->variables()[inputs[i]]->getName();
				if (i < inputs.size() - 1) {
					code_str += ", ";
				}
			}
			code_str += ");\n";
		}

		std::tuple<vc::ui16, vc::ui16, std::vector<vc::ui16>> call;
	};

	export class ScopeBase : public FunctionScope {
	public:

		ScopeBase(const std::shared_ptr<FunctionScope>& parent_scope)
			: FunctionScope(parent_scope)
		{}

		template<ShaderVariableIterator SVIterator>
		ScopeBase& apply(const std::shared_ptr<Function>& func,
			const std::shared_ptr<ShaderVariable>& ret,
			std::optional<std::pair<SVIterator, SVIterator>> args_it)
		{
			auto pparent = parent.lock();

			vc::ui16 func_pos = Function::add_function(pparent->functions(), func);

			int16_t ret_pos = -1;
			if (ret) {
				ret_pos = ShaderVariable::add_variable(pparent->variables(), ret);
			}

			std::vector<vc::ui16> input_pos;
			if (args_it.has_value()) {
				auto& its = args_it.value();
				size_t ninputs = std::distance(its.first, its.second);
				input_pos.reserve(ninputs);
				for (auto it = its.first; it != its.second; ++it) {
					input_pos.emplace_back(ShaderVariable::add_variable(pparent->variables(), *it));
				}
			}

			scope_chain.emplace_back(
				std::move(std::weak_ptr<IfScope>(shared_from_this())),
				std::make_tuple(func_pos, ret_pos, std::move(input_pos));

			return *this;
		}

		ScopeBase& apply(const std::shared_ptr<Function>& func,
			const std::shared_ptr<ShaderVariable>& ret,
			const std::vector<std::shared_ptr<ShaderVariable>>& args)
		{
			return apply(func, ret, std::make_optional(std::make_pair(args.begin(), args.end())));
		}
		
		ScopeBase& apply_scope(std::unique_ptr<ScopeBase> scope)
		{
			ScopeBase& ret = *scope;
			children.emplace_back(std::move(scope));
			return ret;
		}

		ScopeBase& chain(std::unique_ptr<ScopeBase> scope)
		{
			ScopeBase& ret = *scope;
			chain_child = std::move(scope);
			return ret;
		}

	private:

	};


	export class FunctionFactory : public FunctionScope {
	public:

		enum class InputType {
			IN,
			OUT,
			INOUT
		};

		FunctionFactory(const std::string& name, const ShaderVariableType& return_type)
			: FunctionScope(nullptr), m_Name(name), m_ReturnType(return_type)
		{}

		vc::ui16 scope_level() const override 
		{
			return 1;
		}

		std::vector<std::shared_ptr<ShaderVariable>>& variables() override
		{
			return m_Variables;
		}

		std::vector<std::shared_ptr<Function>>& functions() override
		{
			return m_Functions;
		}

		void addMatrix(const std::shared_ptr<MatrixVariable>& mat, const std::optional<FunctionFactory::InputType>& input)
		{
			auto varidx = _addVariable(mat);

			if (input.has_value()) {
				m_Inputs.emplace_back(varidx, input.value());
			}
		}

		void addVector(const std::shared_ptr<VectorVariable>& vec, const std::optional<FunctionFactory::InputType>& input)
		{
			auto varidx = _addVariable(vec);
			if (input.has_value()) {
				m_Inputs.emplace_back(varidx, input.value());
			}
		}

		void addSingle(const std::shared_ptr<SingleVariable>& var, const std::optional<FunctionFactory::InputType>& input)
		{
			auto varidx = _addVariable(var);
			if (input.has_value()) {
				m_Inputs.emplace_back(varidx, input.value());
			}
		}

		template<ShaderVariableIterator SVIterator>
		void apply(const std::shared_ptr<Function>& func,
			const std::shared_ptr<ShaderVariable>& ret,
			std::optional<std::pair<SVIterator, SVIterator>> args_it)
		{
			uint16_t func_pos = Function::add_function(m_Functions, func);

			int16_t ret_pos = -1;
			if (ret) {
				ret_pos = ShaderVariable::add_variable(parent->variables(), ret);
			}

			std::vector<uint16_t> input_pos;
			if (args_it.has_value()) {
				auto& its = args_it.value();
				size_t ninputs = std::distance(its.first, its.second);
				input_pos.reserve(ninputs);
				for (auto it = its.first; it != its.second; ++it) {
					input_pos.emplace_back(ShaderVariable::add_variable(parent->variables(), *it));
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

		std::shared_ptr<::glsl::Function> build()
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
				uniqueid = util::hash_combine(hashes);
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

				// add functions calls
				for (auto& call : m_Calls) {
					auto& func = m_Functions[std::get<0>(call)];

					dependencies.emplace_back(func);

					code_str += "\t";

					// add return variable if there is one
					auto ret_idx = std::get<1>(call);
					if (ret_idx >= 0) {
						auto& ret_var = m_Variables[ret_idx];
						code_str += ret_var->getName() + " = ";
					}

					// add function name
					code_str += func->getName() + "(";

					// add input variables
					auto& inputs = std::get<2>(call);
					for (int i = 0; i < inputs.size(); ++i) {
						code_str += m_Variables[inputs[i]]->getName();
						if (i < inputs.size() - 1) {
							code_str += ", ";
						}
					}
					code_str += ");\n";
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

	private:

		std::string m_Name;
		ShaderVariableType m_ReturnType;

		std::vector<std::pair<vc::ui16, FunctionFactory::InputType>> m_Inputs;
		std::vector<std::shared_ptr<ShaderVariable>> m_Variables;

		std::vector<std::shared_ptr<Function>> m_Functions;
	};

}