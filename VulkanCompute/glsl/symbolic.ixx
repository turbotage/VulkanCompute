module;

export module symbolic;

import <string>;
import <set>;
import <unordered_map>;
import <stdexcept>;

import vc;
import util;

namespace glsl {

	export enum class SymbolicType {
		CONST_TYPE,
		PARAM_TYPE,
	};

	export class SymbolicContext {
	public:

		void insert_const(const std::pair<std::string, vc::ui16>& cp)
		{
			if (symtype_map.contains(cp.first))
				throw std::runtime_error("const name already existed in SymbolicContext");

			for (auto& p : consts_map) {
				if (p.first == cp.first)
					throw std::runtime_error("const name already existed in SymbolicContext");
				if (p.second == cp.second)
					throw std::runtime_error("const index already existed in SymbolicContext");
			}

			symtype_map.insert({ cp.first, SymbolicType::CONST_TYPE });
			consts_map.insert(cp);
		}

		void insert_param(const std::pair<std::string, vc::ui16>& pp)
		{
			if (symtype_map.contains(pp.first))
				throw std::runtime_error("const name already existed in SymbolicContext");

			for (auto& p : params_map) {
				if (p.first == pp.first)
					throw std::runtime_error("const name already existed in SymbolicContext");
				if (p.second == pp.second)
					throw std::runtime_error("const index already existed in SymbolicContext");
			}

			symtype_map.insert({ pp.first, SymbolicType::PARAM_TYPE });
			params_map.insert(pp);
		}

		SymbolicType get_symtype(const std::string& name) const
		{
			return symtype_map.at(name);
		}

		vc::ui16 get_params_index(const std::string& name) const
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

		vc::ui16 get_consts_index(const std::string& name) const
		{
			for (auto& v : consts_map) {
				if (v.first == name)
					return v.second;
			}
			throw std::runtime_error("Name was not in consts in SymbolicContext");
		}

		const std::string& get_consts_name(size_t index) const
		{
			for (auto& v : consts_map) {
				if (v.second == index)
					return v.first;
			}
			throw std::runtime_error("Index was not in consts in SymbolicContext");
		}

		const std::string& get_consts_name() const
		{
			return consts_name;
		}

		const std::string& get_consts_iterable_by() const
		{
			return consts_iterable_by;
		}

		const std::string& get_params_iterable_by() const
		{
			return params_iterable_by;
		}

		std::string get_glsl_var_name(const std::string& name) const
		{
			SymbolicType stype = symtype_map.at(name);

			if (stype == SymbolicType::PARAM_TYPE) {
				vc::ui16 index = get_params_index(name);
				return params_name + "[" + std::to_string(index) + "]";
			}

			if (stype == SymbolicType::CONST_TYPE) {
				vc::ui16 index = get_consts_index(name);
				return consts_name + "[" + consts_iterable_by + "*" + nconst_name + "+" + std::to_string(index) + "]";
			}

			throw std::runtime_error("Variable was neither const nor param");
		}

		std::set<std::pair<std::string, vc::ui16>> params_map;
		std::set<std::pair<std::string, vc::ui16>> consts_map;

		std::unordered_map<std::string, SymbolicType> symtype_map;

		std::string params_name = "params";
		std::string consts_name = "consts";

		std::string consts_iterable_by = "i";
		std::string params_iterable_by = "i";

		std::string ndata_name = "ndata";
		std::string nconst_name = "nconst";
	};

}