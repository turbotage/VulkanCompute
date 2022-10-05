module;

export module function;

import <string>;
import <vector>;
import <functional>;
import <optional>;

import vc;
import util;

import variable;

namespace glsl {

	export class Function;

	using vecptrfunc = std::vector<std::shared_ptr<Function>>;
	using refvecptrfunc = vc::refw<std::vector<std::shared_ptr<Function>>>;

	export class Function {
	public:

		Function(
			const std::string& function_name,
			const std::vector<size_t>& argument_hashes,
			const std::function<std::string()>& code_func,
			const std::optional<std::vector<std::shared_ptr<Function>>>& dependencies)
			:
			m_FunctionName(function_name),
			m_ArgumentHashes(argument_hashes),
			m_CodeFunc(code_func),
			m_Dependencies(dependencies.has_value() ? dependencies.value() : vecptrfunc())
		{}

		const std::vector<std::shared_ptr<Function>>& getDependencies() const
		{
			return m_Dependencies;
		}

		std::string getCode() const
		{
			return m_CodeFunc();
		}

		std::string getName() const
		{
			return m_FunctionName;
		}

		friend bool operator==(const Function& lhs, const Function& rhs)
		{
			return lhs.m_FunctionName == rhs.m_FunctionName &&
				lhs.m_ArgumentHashes == rhs.m_ArgumentHashes;
		}

		struct HashFunction {
			std::size_t operator()(const Function& func) const {
				std::size_t hash = util::hash_combine(func.m_ArgumentHashes);
				return util::hash_combine(hash, func.m_FunctionName);
			}
		};

		static size_t add_function(std::vector<std::shared_ptr<Function>>& funcs, const std::shared_ptr<Function>& func)
		{
			for (auto& f : func->m_Dependencies) {
				add_function(funcs, f);
			}

			auto it = std::find_if(funcs.begin(), funcs.end(), [&func](const std::shared_ptr<Function>& f) {
				return *func == *f;
				});
			size_t pos = it - funcs.begin();
			if (it == funcs.end()) {
				funcs.push_back(func);
			}
			return pos;
		}

	private:

		std::string m_FunctionName;
		std::vector<std::size_t> m_ArgumentHashes;
		std::function<std::string()> m_CodeFunc;
		std::vector<std::shared_ptr<Function>> m_Dependencies;

		std::string m_HashName;
	};

	export struct FunctionApplier {
		std::shared_ptr<Function> func;
		std::shared_ptr<ShaderVariable> ret_var;
		std::vector<std::shared_ptr<ShaderVariable>> args;

		std::string unique_id;
	};

}

