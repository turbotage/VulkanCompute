module;

#include <auto_vk_toolkit.hpp>

export module vk_variable;

namespace compute {

	export class VkVariable {
	public:

		const avk::buffer& get_storage() const {
			return m_StorageBuffer;
		}

	private:
		avk::buffer m_StorageBuffer;
	};

}