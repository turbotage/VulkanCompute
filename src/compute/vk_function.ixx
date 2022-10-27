module;

#include <auto_vk_toolkit.hpp>

export module vk_function;

import <mutex>;

import vc;
import vk_variable;

import glsl;
import shader;

namespace compute {

	avk::compute_pipeline create_pipeline(const std::string& source, const std::map<std::string, std::shared_ptr<VkVariable>>& vars)
	{
		static std::mutex mut;

		std::unique_lock<std::mutex> lock(mut);

		std::string source_name = glsl::compileSourceToFile(source);

		avk::compute_pipeline_config cfg;
		cfg.mShaderInfo = avk::shader_info::describe(source_name);
		if (cfg.mShaderInfo->mShaderType != avk::shader_type::compute) {
			throw avk::logic_error("The shader's type is not compute");
		}

		// Bindings
		int i = 0;
		for (auto& var : vars) {
			cfg.mResourceBindings.push_back(avk::descriptor_binding(0, i, var.second->get_storage()));
			++i;
		}

		// TODO add push constants

		return avk::root::create_compute_pipeline(std::move(cfg));
	}

	export class VkFunction {
	public:

		VkFunction(vc::ui32 batch_size, std::shared_ptr<glsl::ShaderBase> shader, const std::map<std::string, std::shared_ptr<VkVariable>>& vars)
			: m_BatchSize(batch_size), m_BindVariables(vars)
		{
			// Initialize compute-pipeline
			m_ComputePipeline = create_pipeline(shader->compile(), m_BindVariables);
		}

		const std::map<std::string, std::shared_ptr<VkVariable>>& bound_vars() const
		{
			return m_BindVariables;
		}

		void record() 
		{
			std::vector<avk::recorded_commands_t> cmds;
			cmds.reserve(3);

			// Bind compute pipeline
			cmds.push_back(avk::command::bind_pipeline(m_ComputePipeline.as_reference()));

			// descriptor init-list lambda
			auto init_list_lambda = [this]() -> std::initializer_list<avk::binding_data>
			{
				std::vector<avk::binding_data> bind_datas;
				bind_datas.reserve(m_BindVariables.size());
				int i = 0;
				for (auto& bvar : m_BindVariables) {
					bind_datas.push_back(avk::descriptor_binding(0, i, bvar.second->get_storage()));
					++i;
				}

				return std::initializer_list<avk::binding_data>(&bind_datas.front(), &bind_datas.back());
			};

			// Bind tensors
			cmds.push_back(avk::command::bind_descriptors(m_ComputePipeline->layout(), m_DescriptorCache
				->get_or_create_descriptor_sets(init_list_lambda())));

			// dispatch
			cmds.push_back(avk::command::dispatch(m_BatchSize, 1, 1));

		}

	private:
		avk::compute_pipeline m_ComputePipeline;
		avk::descriptor_cache m_DescriptorCache;
		std::map<std::string, std::shared_ptr<VkVariable>> m_BindVariables;

		vc::ui32 m_BatchSize;

	};

}