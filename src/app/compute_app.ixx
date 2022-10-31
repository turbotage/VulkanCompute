module;

#include <auto_vk_toolkit.hpp>

#include <imgui.h>
#include <imgui_impl_vulkan.h>

export module compute_app;

class MIMaterial {
public:

private:
	//avk::buffer m_StorageBuffer;
};

class ComputeApp : public avk::invokee
{
private:

	struct Vertex {
		glm::vec3 pos;
		glm::vec2 uv;
	};

	struct MatricesForUbo {
		glm::mat4 projection;
		glm::mat4 model;
	};

	// Vertices for a quad
	const std::vector<Vertex> m_VertexData = {
		{{  1.0f,  1.0f, 0.0f }, { 1.0f, 1.0f }},
		{{ -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f }},
		{{ -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f }},
		{{  1.0f, -1.0f, 0.0f }, { 1.0f, 0.0f }},
	};

	// Indices for a quad:
	const std::vector<uint16_t> m_Indices = {
		0, 1, 2,   2, 3, 0
	};

public:

	ComputeApp(avk::queue& aQueue)
		: m_Queue(aQueue), m_RotationSpeed(1.0f)
	{}

	void initialize() override
	{
		m_VertexBuffer = avk::context().create_buffer(
			avk::memory_usage::device, {},
			avk::vertex_buffer_meta::create_from_data(m_VertexBuffer)
		);
		avk::context().record_and_submit_with_fence(
			{ m_VertexBuffer->fill(m_VertexData.data(), 0) }, m_Queue)->wait_until_signalled();

		m_IndexBuffer = avk::context().create_buffer(
			avk::memory_usage::device, {},
			avk::index_buffer_meta::create_from_data(m_Indices)
		);
		avk::context().record_and_submit_with_fence(
			{ m_IndexBuffer->fill(m_Indices.data(), 0) }, m_Queue)->wait_until_signalled();

		auto fif = avk::context().main_window()->number_of_frames_in_flight();
		for (decltype(fif) i = 0; i < fif; ++i) {
			m_Ubo.emplace_back(avk::context().create_buffer(
				avk::memory_usage::host_coherent, {},
				avk::uniform_buffer_meta::create_from_data(MatricesForUbo{})
			));
		}

	}

private:

	avk::queue& m_Queue;
	avk::buffer m_VertexBuffer;
	avk::buffer m_IndexBuffer;
	std::vector<avk::buffer> m_Ubo;
	avk::image_sampler m_InputImageAndSampler;
	avk::image_sampler m_TargetImageAndSampler;
	avk::descriptor_cache m_DescriptorCache;

	avk::graphics_pipeline m_GraphicsPipeline;

	std::optional<avk::semaphore> m_UpdateToRenderDependency;

	float m_RotationSpeed;
};
