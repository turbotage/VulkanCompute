module;

#include <string>
#include <optional>
#include <vector>
#include <functional>
#include <unordered_set>
#include <fstream>

export module glsl;

import util;

namespace glsl {
	

	export std::vector<uint32_t>
		compileSource(const std::string& source)
	{
		std::ofstream fileOut("tmp_kp_shader.comp");
		fileOut << source;
		fileOut.close();
		if (system(
			std::string(
				"glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv")
			.c_str()))
			throw std::runtime_error("Error running glslangValidator command");
		std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
		std::vector<char> buffer;
		buffer.insert(
			buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
		return { (uint32_t*)buffer.data(),
				 (uint32_t*)(buffer.data() + buffer.size()) };
	}

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

		std::string getCode() {
			return m_CodeFunc();
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

		friend void add_functions(std::vector<Function>& funcs, const Function& func);
		
		std::string m_FunctionName;
		std::vector<std::size_t> m_ArgumentHashes;
		std::function<std::string()> m_CodeFunc;
		std::vector<Function> m_Dependencies;

	};
	
	void add_functions(std::vector<Function>& funcs, const Function& func) {
		for (auto& f : func.m_Dependencies) {
			add_functions(funcs, f);
		}

		if (std::find(funcs.begin(), funcs.end(), func) == funcs.end()) {
			funcs.push_back(func);
		}
	}

	export class Binding {
	public:

		virtual std::string operator()() const = 0;

		virtual bool operator==(const Binding* other) const = 0;

	};

	export class BufferBinding : public Binding {
	public:

		BufferBinding(int set, int binding, std::string type, std::string name)
			: m_Set(set), m_Binding(binding), m_Type(type), m_Name(name) {}

		std::string operator()() const override {
			return "layout(set = " + std::to_string(m_Set) + ", binding = " + std::to_string(m_Binding) + ") buffer buf_" +
				m_Name + " { " + m_Type + " " + m_Name + "[]; };";
		}

		bool operator==(const Binding* other) const override {
			if (auto* b = dynamic_cast<const BufferBinding*>(other); b != nullptr) {
				return (b->m_Set == m_Set) && (b->m_Binding == m_Binding) && 
					(b->m_Type == m_Type) && (b->m_Name == m_Name);
			}
			return false;
		}

	private:
		int m_Set;
		int m_Binding;
		std::string m_Type;
		std::string m_Name;
	};

	export class Shader {
	public:

		void addBinding(std::unique_ptr<Binding> binding)
		{

			auto it = std::find_if(m_Bindings.begin(), m_Bindings.end(), [&binding](const std::unique_ptr<Binding>& b) 
			{
				return binding->operator==(b.get());
			});

			if (it == m_Bindings.end()) {
				m_Bindings.emplace_back(std::move(binding));
			}
		}

		void addFunction(const Function& func)
		{
			add_functions(m_Functions, func);
		}

		std::string compile() {
			std::string ret = 
R"(
#version 450

layout (local_size_x = 1) in;

)";
			
			for (auto& bind : m_Bindings) {
				ret += bind->operator()() + "\n";
			}

			for (auto& func : m_Functions) {
				ret += func.getCode() + "\n";
			}

			return ret;
		}

	private:

		std::vector<std::unique_ptr<Binding>> m_Bindings;
		std::vector<Function> m_Functions;
		//std::unordered_set<Function, Function::HashFunction> m_Functions;

	};

}

