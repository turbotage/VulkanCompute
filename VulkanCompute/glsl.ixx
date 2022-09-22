module;

#include <string>
#include <optional>
#include <vector>
#include <functional>
#include <unordered_set>
#include <fstream>
#include <set>

#include <kompute/Kompute.hpp>

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
R"glsl(
#version 450

layout (local_size_x = 1) in;

)glsl";
			
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
	};

	export enum class eSymbolicType {
		CONST_TYPE,
		PARAM_TYPE,
	};

	export class SymbolicContext {
	public:

		void insert_const(const std::pair<std::string, uint32_t>& cp)
		{
			if (symtype_map.contains(cp.first))
				throw std::runtime_error("const name already existed in SymbolicContext");

			for (auto& p : consts_map) {
				if (p.first == cp.first)
					throw std::runtime_error("const name already existed in SymbolicContext");
				if (p.second == cp.second)
					throw std::runtime_error("const index already existed in SymbolicContext");
			}
			
			symtype_map.insert({ cp.first, eSymbolicType::CONST_TYPE });
			consts_map.insert(cp);
		}

		void insert_param(const std::pair<std::string, uint32_t>& pp)
		{
			if (symtype_map.contains(pp.first))
				throw std::runtime_error("const name already existed in SymbolicContext");

			for (auto& p : params_map) {
				if (p.first == pp.first)
					throw std::runtime_error("const name already existed in SymbolicContext");
				if (p.second == pp.second)
					throw std::runtime_error("const index already existed in SymbolicContext");
			}

			symtype_map.insert({ pp.first, eSymbolicType::PARAM_TYPE });
			params_map.insert(pp);
		}

		eSymbolicType get_symtype(const std::string& name) const 
		{
			return symtype_map.at(name);
		}

		uint32_t get_params_index(const std::string& name) const 
		{
			for (auto& v : params_map) {
				if (v.first == name)
					return v.second;
			}
			throw std::runtime_error("Name was not in params in SymbolicContext");
		}

		const std::string& get_params_name(size_t index) const
		{
			for (auto& v : params_map) {
				if (v.second == index)
					return v.first;
			}
			throw std::runtime_error("Index was not in params in SymbolicContext");
		}

		uint32_t get_consts_index(const std::string& name) const
		{
			for (auto& v : consts_map) {
				if (v.first == name)
					return v.second;
			}
			throw std::runtime_error("Name was not in consts in SymbolicContext");
		}

		const std::string& get_consts_name(size_t index) const {
			for (auto& v : consts_map) {
				if (v.second == index)
					return v.first;
			}
			throw std::runtime_error("Index was not in consts in SymbolicContext");
		}

		const std::string& get_consts_name() const {
			return consts_name;
		}

		const std::string& get_consts_iterable_by() const {
			return consts_iterable_by;
		}

		const std::string& get_params_iterable_by() const {
			return params_iterable_by;
		}

		std::string get_glsl_var_name(const std::string& name) const
		{
			glsl::eSymbolicType stype = symtype_map.at(name);

			if (stype == glsl::eSymbolicType::PARAM_TYPE) {
				uint32_t index = get_params_index(name);
				return params_name + "[" + std::to_string(index) + "]";
			}

			if (stype == glsl::eSymbolicType::CONST_TYPE) {
				uint32_t index = get_consts_index(name);
				return consts_name + "[" + consts_iterable_by + "*" + nconst_name + "+" + std::to_string(index) + "]";
			}

			throw std::runtime_error("Variable was neither const nor param");
		}

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

