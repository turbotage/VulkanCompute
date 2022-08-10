module;

#include <string>
#include <optional>
#include <vector>
#include <functional>

export module glsl;

import util;

namespace glsl {
	
	export class Function {
	public:

		Function(
			const std::string& function_name,
			const std::vector<size_t>& argument_hashes,
			const std::function<std::string()>& code_func,
			std::optional<std::vector<Function>> dependencies)
			: 
			m_FunctionName(function_name),
			m_ArgumentHashes(argument_hashes),
			m_CodeFunc(code_func)
			
		{
			if (dependencies.has_value())
				m_Dependencies = dependencies.value();
		}

		std::vector<Function> getDependencies() {
			return m_Dependencies;
		}

		friend bool operator==(const Function& lhs, const Function& rhs) {
			return lhs.m_FunctionName == rhs.m_FunctionName &&
				lhs.m_ArgumentHashes == rhs.m_ArgumentHashes;
		}

		struct HashFunction {
			std::size_t operator()(const Function& func) const {
				std::size_t hash = util::hash_combine(func.m_ArgumentHashes);
				return util::hash_combine(hash, func.m_FunctionName);
			}
		};

	private:

		std::string m_FunctionName;
		std::vector<std::size_t> m_ArgumentHashes;
		std::function<std::string()> m_CodeFunc;
		std::vector<Function> m_Dependencies;

	};


	export class Context {
	public:


	private:


	};


}

