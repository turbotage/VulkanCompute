module;

#include <auto_vk_toolkit.hpp>

export module vk_function;

import vk_variable;

namespace compute {

	export class VkFunction {
	public:

		void record() 
		{
			std::vector<avk::command::state_type_command> cmds;
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
					bind_datas.push_back(avk::descriptor_binding(0, i, bvar->get_storage()));
				}

				return std::initializer_list<avk::binding_data>(&bind_datas.front(), &bind_datas.back());
			};

			// Bind tensors
			cmds.push_back(avk::command::bind_descriptors(m_ComputePipeline->layout(), (*m_pDescriptorCache)
				->get_or_create_descriptor_sets(init_list_lambda())));

			// dispatch
			//cmds.push_back()

		}

	private:
		avk::compute_pipeline m_ComputePipeline;
		std::shared_ptr<avk::descriptor_cache> m_pDescriptorCache;
		std::vector<std::shared_ptr<VkVariable>> m_BindVariables;

	};

}