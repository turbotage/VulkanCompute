// VulkanCompute.cpp : Defines the entry point for the application.
//

#include "VulkanCompute.hpp"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

import <vector>;
import <memory>;
import <optional>;
import <random>;
import <chrono>;
import <string>;
import <iomanip>;
import <filesystem>;

import vc;
import util;
import glsl;
import linalg;
import solver;
import symbolic;
import expr;
import symm;
import nlsq;
import nlsq_symbolic;

import qmri;

import variable;
import function;
import shader;
import func_factory;

import tensor_var;

void test_new_autogen() {
	using namespace glsl;

	auto shader1 = qmri::ivim_guess_shader(21, true);
	auto shader2 = qmri::ivim_full_nlsq_shader(21, true);

	auto sh1_compiled = shader1->compile();
	std::cout << util::add_line_numbers(
		sh1_compiled
	);

	auto spirv1 = glsl::compileSource(sh1_compiled);

	std::cout << "\n\n\n\n";

	auto sh2_compiled = shader2->compile();
	std::cout << util::add_line_numbers(
		sh2_compiled
	);

	auto spirv2 = glsl::compileSource(sh2_compiled);
	
}

void run_qmri_ivim() {

	uint32_t ndata = 21;
	uint32_t nconst = 1;


	auto params = std::make_shared<glsl::VectorVariable>("params", 4, glsl::ShaderVariableType::eFloat);
	auto consts = std::make_shared<glsl::MatrixVariable>("consts", ndata, nconst, glsl::ShaderVariableType::eFloat);
	auto data = std::make_shared<glsl::VectorVariable>("data", ndata, glsl::ShaderVariableType::eFloat);
	auto bsplit = std::make_shared<glsl::VectorVariable>("bsplit", 2, glsl::ShaderVariableType::eInt);
	auto weights = std::make_shared<glsl::VectorVariable>("weights", ndata, glsl::ShaderVariableType::eFloat);
	auto lambda = std::make_shared<glsl::SingleVariable>("lambda", glsl::ShaderVariableType::eFloat, std::nullopt);
	auto step_type = std::make_shared<glsl::SingleVariable>("step_type", glsl::ShaderVariableType::eInt, std::nullopt);
	auto nlstep = std::make_shared<glsl::VectorVariable>("nlstep", 4, glsl::ShaderVariableType::eFloat);
	auto error = std::make_shared<glsl::SingleVariable>("error", glsl::ShaderVariableType::eFloat, std::nullopt);
	auto new_error = std::make_shared<glsl::SingleVariable>("new_error", glsl::ShaderVariableType::eFloat, std::nullopt);
	auto residuals = std::make_shared<glsl::VectorVariable>("residuals", ndata, glsl::ShaderVariableType::eFloat);
	auto jacobian = std::make_shared<glsl::MatrixVariable>("jacobian", ndata, 4, glsl::ShaderVariableType::eFloat);
	auto hessian = std::make_shared<glsl::MatrixVariable>("hessian", 4, 4, glsl::ShaderVariableType::eFloat);
	auto lambda_hessian = std::make_shared<glsl::MatrixVariable>("lambda_hessian", 4, 4, glsl::ShaderVariableType::eFloat);
	auto upper_bound = std::make_shared<glsl::VectorVariable>("upper_bound", 4, glsl::ShaderVariableType::eFloat);
	auto lower_bound = std::make_shared<glsl::VectorVariable>("lower_bound", 4, glsl::ShaderVariableType::eFloat);

	auto pShader1 = glsl::qmri::ivim_guess_shader(ndata, true);
	auto shaderStr1 = pShader1->compile();
	std::cout << shaderStr1 << std::endl;
	std::cout << "\n\n\n" << std::endl;

	auto pShader2 = glsl::qmri::ivim_partial_nlsq_shader(ndata, true);
	auto shaderStr2 = pShader2->compile();
	std::cout << shaderStr2 << std::endl;
	std::cout << "\n\n\n" << std::endl;

	auto pShader3 = glsl::qmri::ivim_full_nlsq_shader(ndata, true);
	auto shaderStr3 = pShader3->compile();
	std::cout << shaderStr3 << std::endl;
	std::cout << "\n\n\n" << std::endl;

	namespace fs = std::filesystem;
	auto d_path = fs::current_path() / "data" / "ivim_data.vcdat";
	auto c_path = fs::current_path() / "data" / "ivim_bvals.vcdat";

	size_t nelem = fs::file_size(d_path) / ndata / sizeof(float);

	auto mgr = std::make_shared<kp::Manager>();

	auto kp_params = glsl::tensor_from_vector(mgr, params, nelem);
	auto kp_consts = glsl::tensor_from_file(mgr, glsl::ShaderVariableType::eFloat, c_path);
	auto kp_data = glsl::tensor_from_file(mgr, glsl::ShaderVariableType::eFloat, d_path);
	auto kp_bsplit = glsl::tensor_from_vector(mgr, bsplit, 1);
	std::vector<int32_t> data_bsplit(2); data_bsplit[0] = 11; data_bsplit[1] = 15;
	std::memcpy(kp_bsplit->data<int32_t>(), data_bsplit.data(), sizeof(int32_t) * data_bsplit.size());

	auto kp_weights = glsl::tensor_from_vector(mgr, weights, 1);
	std::vector<float> data_weights(21, 1.0f);
	std::memcpy(kp_weights->data<float>(), data_weights.data(), sizeof(float) * data_weights.size());

	auto kp_lambda = glsl::tensor_from_single(mgr, lambda, nelem);
	std::vector<float> data_lambda(nelem, 1.0f);
	std::memcpy(kp_lambda->data<float>(), data_lambda.data(), data_lambda.size() * sizeof(float));

	auto kp_step_type = glsl::tensor_from_single(mgr, step_type, nelem);
	auto kp_nlstep = glsl::tensor_from_vector(mgr, nlstep, nelem);
	auto kp_error = glsl::tensor_from_single(mgr, error, nelem);
	auto kp_new_error = glsl::tensor_from_single(mgr, new_error, nelem);
	auto kp_residuals = glsl::tensor_from_vector(mgr, residuals, nelem);
	auto kp_jacobian = glsl::tensor_from_matrix(mgr, jacobian, nelem);
	auto kp_hessian = glsl::tensor_from_matrix(mgr, hessian, nelem);


	auto spirv1 = glsl::compileSource(shaderStr1);
	auto spirv2 = glsl::compileSource(shaderStr2);
	auto spirv3 = glsl::compileSource(shaderStr3);

	std::vector<std::shared_ptr<kp::Tensor>> shader_inputs = {
		kp_params, kp_consts, kp_data, kp_bsplit, kp_weights, kp_lambda, kp_step_type,
		kp_nlstep, kp_error, kp_new_error, kp_residuals, kp_jacobian, kp_hessian
	};

	kp::Workgroup wg{ (size_t)nelem, 1, 1 };
	
	std::shared_ptr<kp::Algorithm> algo1 = mgr->algorithm(shader_inputs, spirv1, wg);
	std::shared_ptr<kp::Algorithm> algo2 = mgr->algorithm(shader_inputs, spirv2, wg);
	std::shared_ptr<kp::Algorithm> algo3 = mgr->algorithm(shader_inputs, spirv3, wg);

	auto start = std::chrono::steady_clock::now();

	auto seq = mgr->sequence()
		->record<kp::OpTensorSyncDevice>(shader_inputs)
		->record<kp::OpAlgoDispatch>(algo1)
		->record<kp::OpMemoryBarrier>(shader_inputs, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
			vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader)
		->record<kp::OpAlgoDispatch>(algo2)
		->record<kp::OpMemoryBarrier>(shader_inputs, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
			vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader)
		->record<kp::OpAlgoDispatch>(algo3)
		->record<kp::OpTensorSyncLocal>(shader_inputs)
		->eval();

	auto end = std::chrono::steady_clock::now();

	std::cout << "run time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

	bool print = true;
	if (print) {
		std::cout.precision(10);
		std::cout << "CONST TYPES: " << std::endl;
		std::cout << "bsplit: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_bsplit, bsplit, 0) << std::endl;
		std::cout << "weights: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_weights, weights, 0) << std::endl;
		std::cout << "consts: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_consts, consts, 0) << std::endl;


		std::cout << "FIRST ELEMENTS" << std::endl;
		std::cout << "residuals: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_residuals, residuals, 0) << std::endl;
		std::cout << "jacobian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_jacobian, jacobian, 0) << std::endl;
		std::cout << "hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_hessian, hessian, 0) << std::endl;
		std::cout << "data: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_data, data, 0) << std::endl;
		std::cout << "params: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_params, params, 0) << std::endl;
		std::cout << "lambda: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda, lambda, 0) << std::endl;
		std::cout << "step_type: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_step_type, step_type, 0) << std::endl;
		std::cout << "step: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_nlstep, nlstep, 0) << std::endl;
	

		std::cout << "MID ELEMENTS" << std::endl;
		std::cout << "residuals: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_residuals, residuals, nelem / 2) << std::endl;
		std::cout << "jacobian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_jacobian, jacobian, nelem / 2) << std::endl;
		std::cout << "hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_hessian, hessian, nelem / 2) << std::endl;
		std::cout << "data: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_data, data, nelem / 2) << std::endl;
		std::cout << "params: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_params, params, nelem / 2) << std::endl;
		std::cout << "lambda: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda, lambda, nelem / 2) << std::endl;
		std::cout << "step_type: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_step_type, step_type, nelem / 2) << std::endl;
		std::cout << "step: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_nlstep, nlstep, nelem / 2) << std::endl;

		
		std::cout << "LAST ELEMENTS" << std::endl;
		std::cout << "residuals: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_residuals, residuals, nelem - 1) << std::endl;
		std::cout << "jacobian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_jacobian, jacobian, nelem - 1) << std::endl;
		std::cout << "hessian: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_hessian, hessian, nelem - 1) << std::endl;
		std::cout << "data: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_data, data, nelem - 1) << std::endl;
		std::cout << "params: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_params, params, nelem - 1) << std::endl;
		std::cout << "lambda: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_lambda, lambda, nelem - 1) << std::endl;
		std::cout << "step_type: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_step_type, step_type, nelem - 1) << std::endl;
		std::cout << "step: " << std::endl;
		std::cout << glsl::print_shader_variable(kp_nlstep, nlstep, nelem - 1) << std::endl;
	}

	auto p_path = fs::current_path() / "data" / "ivim_params.vcdat";
	glsl::tensor_to_file(kp_params, glsl::ShaderVariableType::eFloat, p_path);

}

class ComputeAndRenderApp : public avk::invokee
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

	const std::vector<Vertex> m_VertexData = {
		{{  1.0f,  1.0f, 0.0f }, { 1.0f, 1.0f }},
		{{ -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f }},
		{{ -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f }},
		{{  1.0f, -1.0f, 0.0f }, { 1.0f, 0.0f }},
	};

	// Indices for the quad:
	const std::vector<uint16_t> m_Indices = {
		 0, 1, 2,   2, 3, 0
	};

public:

	ComputeAndRenderApp(avk::queue& aQueue)
		: m_Queue(&aQueue), m_RotationSpeed(1.0f)
	{}

	void initialize() override
	{
		// Create and upload vertex data for a quad
		m_VertexBuffer = avk::context().create_buffer(
			avk::memory_usage::device, {},
			avk::vertex_buffer_meta::create_from_data(m_VertexBuffer)
		);
		avk::context().record_and_submit_with_fence({ m_VertexBuffer->fill(m_VertexData.data(), 0) }, *m_Queue)->wait_until_signalled();

		// Create and upload incides for drawing the quad
		m_IndexBuffer = avk::context().create_buffer(
			avk::memory_usage::device, {},
			avk::index_buffer_meta::create_from_data(m_Indices)
		);
		avk::context().record_and_submit_with_fence({ m_IndexBuffer->fill(m_Indices.data(), 0) }, *m_Queue)->wait_until_signalled();

		// Create a host-coherent buffer for the matrices
		auto fif = avk::context().main_window()->number_of_frames_in_flight();
		for (decltype(fif) i = 0; i < fif; ++i) {
			m_Ubo.emplace_back(avk::context().create_buffer(
				avk::memory_usage::host_coherent, {}, // Note: This flag lets the buffer be created in a different memory region than vertex and index buffers above
				avk::uniform_buffer_meta::create_from_data(MatricesForUbo{})
			));
		}

		// Load an image from file, upload it and then create a view and a sampler for it for usage in shaders:
		auto [image, uploadInputImageCommand] = avk::create_image_from_file(
			"assets/lion.png", false, false, true, 4,
			avk::layout::transfer_src, // For now, transfer the image into transfer_dst layout, because we'll need to copy from it:
			avk::memory_usage::device, // The device shall be stored in (fast) device-local memory. For this reason, the function will also return commands that we need to execute later
			avk::image_usage::general_storage_image // Note: We could bind the image as a texture instead of a (readonly) storage image, then we would not need the "storage_image" usage flag 
		);
		// The uploadInputImageCommand will contain a copy operation from a temporary host-coherent buffer into a device-local buffer.
		// We schedule it for execution a bit further down.		

		m_InputImageAndSampler = avk::context().create_image_sampler(
			avk::context().create_image_view(image),
			avk::context().create_sampler(avk::filter_mode::bilinear, avk::border_handling_mode::clamp_to_edge)
		);
		const auto wdth = m_InputImageAndSampler->width();
		const auto hght = m_InputImageAndSampler->height();
		const auto frmt = m_InputImageAndSampler->format();

		// Create an image to write the modified result into, also create view and sampler for that
		m_TargetImageAndSampler = avk::context().create_image_sampler(
			avk::context().create_image_view(
				avk::context().create_image( // Create an image and set some properties:
					wdth, hght, frmt, 1 /* one layer */, avk::memory_usage::device, /* in your GPU's device-local memory */
					avk::image_usage::general_storage_image // This flag means (among other usages) that the image can be written to, because it includes the "storage_image" usage flag.
				)
			),
			avk::context().create_sampler(avk::filter_mode::bilinear, avk::border_handling_mode::clamp_to_edge)
		);

		// Execute the uploadInputImageCommand command, wait until that one has completed (by using an automatically created barrier), 
		// then initialize the target image with the contents of the input image:
		avk::context().record_and_submit_with_fence({
			// Copy into the source image:
			std::move(uploadInputImageCommand),

			// Wait until the copy has completed:
			avk::sync::global_memory_barrier(avk::stage::auto_stage >> avk::stage::auto_stage, avk::access::auto_access >> avk::access::auto_access),

			// Transition the target image into a useful image layout:
			avk::sync::image_memory_barrier(m_TargetImageAndSampler->get_image(), avk::stage::none >> avk::stage::auto_stage, avk::access::none >> avk::access::auto_access)
				.with_layout_transition(avk::layout::undefined >> avk::layout::general),

			// Copy source to target:
			avk::copy_image_to_another(image.as_reference(), avk::layout::transfer_src, m_TargetImageAndSampler->get_image(), avk::layout::general),

			// Finally, transition the source image into general layout for use in compute and graphics pipelines
			avk::sync::image_memory_barrier(image.as_reference(), avk::stage::auto_stage >> avk::stage::none, avk::access::auto_access >> avk::access::none)
				.with_layout_transition(avk::layout::transfer_src >> avk::layout::general),
			}, *m_Queue)->wait_until_signalled(); // Finally, wait with a fence until everything has completed.

		// Create our rasterization graphics pipeline with the required configuration:
		m_GraphicsPipeline = avk::context().create_graphics_pipeline_for(
			avk::from_buffer_binding(0)->stream_per_vertex(&Vertex::pos)->to_location(0),
			avk::from_buffer_binding(0)->stream_per_vertex(&Vertex::uv)->to_location(1),
			"shaders/texture1.vert",
			"shaders/texture1.frag",
			avk::cfg::front_face::define_front_faces_to_be_clockwise(),
			avk::cfg::culling_mode::disabled,
			avk::cfg::viewport_depth_scissors_config::from_framebuffer(avk::context().main_window()->backbuffer_reference_at_index(0)).enable_dynamic_viewport(),
			avk::context().create_renderpass({
					avk::attachment::declare(
						avk::format_from_window_color_buffer(avk::context().main_window()),
						avk::on_load::clear.from_previous_layout(avk::layout::undefined), avk::usage::color(0), avk::on_store::store.in_layout(avk::layout::color_attachment_optimal) // Not presentable format yet, because ImGui renders afterwards
					).set_clear_color({0.f, 0.5f, 0.75f, 0.0f}), // Set a different clear color
				}, avk::context().main_window()->renderpass_reference().subpass_dependencies() // Use the same subpass dependencies as main window's renderpass
			),
			// Define bindings:
			avk::descriptor_binding(0, 0, m_Ubo[0]),	// Just take any UBO, as this is just used to describe the pipeline's layout.
			avk::descriptor_binding<avk::combined_image_sampler_descriptor_info>(0, 1, 1u)
		);

		// Create 3 compute pipelines:
		m_ComputePipelines.resize(3);
		m_ComputePipelines[0] = avk::context().create_compute_pipeline_for(
			"shaders/emboss.comp",
			avk::descriptor_binding(0, 0, m_InputImageAndSampler->get_image_view()->as_storage_image(avk::layout::general)),
			avk::descriptor_binding(0, 1, m_TargetImageAndSampler->get_image_view()->as_storage_image(avk::layout::general))
		);
		m_ComputePipelines[1] = avk::context().create_compute_pipeline_for(
			"shaders/edgedetect.comp",
			avk::descriptor_binding(0, 0, m_InputImageAndSampler->get_image_view()->as_storage_image(avk::layout::general)),
			avk::descriptor_binding(0, 1, m_TargetImageAndSampler->get_image_view()->as_storage_image(avk::layout::general))
		);
		m_ComputePipelines[2] = avk::context().create_compute_pipeline_for(
			"shaders/sharpen.comp",
			avk::descriptor_binding(0, 0, m_InputImageAndSampler->get_image_view()->as_storage_image(avk::layout::general)),
			avk::descriptor_binding(0, 1, m_TargetImageAndSampler->get_image_view()->as_storage_image(avk::layout::general))
		);

		mUpdater.emplace();
		mUpdater->on(avk::shader_files_changed_event(m_ComputePipelines[0].as_reference())).update(m_ComputePipelines[0]);
		mUpdater->on(avk::shader_files_changed_event(m_ComputePipelines[1].as_reference())).update(m_ComputePipelines[1]);
		mUpdater->on(avk::shader_files_changed_event(m_ComputePipelines[2].as_reference())).update(m_ComputePipelines[2]);

		// Create a descriptor cache that helps us to conveniently create descriptor sets:
		m_DescriptorCache = avk::context().create_descriptor_cache();

		auto* imguiManager = avk::current_composition()->element_by_type<avk::imgui_manager>();
		if (nullptr != imguiManager) {
			imguiManager->add_callback([this, imguiManager]() {
				ImGui::Begin("Info & Settings");
				ImGui::SetWindowPos(ImVec2(1.0f, 1.0f), ImGuiCond_FirstUseEver);
				ImGui::Text("%.3f ms/frame", 1000.0f / ImGui::GetIO().Framerate);
				ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
				ImGui::InputFloat("Rotation Speed", &m_RotationSpeed, 0.1f, 1.0f);

				ImGui::Separator();

				ImTextureID inputTexId = imguiManager->get_or_create_texture_descriptor(m_InputImageAndSampler.as_reference(), avk::layout::general);
				auto inputTexWidth = static_cast<float>(m_InputImageAndSampler->get_image_view()->get_image().create_info().extent.width);
				auto inputTexHeight = static_cast<float>(m_InputImageAndSampler->get_image_view()->get_image().create_info().extent.height);
				ImGui::Text("Input image (%.0fx%.0f):", inputTexWidth, inputTexHeight);
				ImGui::Image(inputTexId, ImVec2(inputTexWidth / 6.0f, inputTexHeight / 6.0f), ImVec2(0, 0), ImVec2(1, 1), ImVec4(1.0f, 1.0f, 1.0f, 1.0f), ImVec4(1.0f, 1.0f, 1.0f, 0.5f));

				ImTextureID targetTexId = imguiManager->get_or_create_texture_descriptor(m_TargetImageAndSampler.as_reference(), avk::layout::general);
				auto targetTexWidth = static_cast<float>(m_TargetImageAndSampler->get_image_view()->get_image().create_info().extent.width);
				auto targetTexHeight = static_cast<float>(m_TargetImageAndSampler->get_image_view()->get_image().create_info().extent.height);
				ImGui::Text("Output image (%.0fx%.0f):", targetTexWidth, targetTexHeight);
				ImGui::Image(targetTexId, ImVec2(targetTexWidth / 6.0f, targetTexHeight / 6.0f), ImVec2(0, 0), ImVec2(1, 1), ImVec4(1.0f, 1.0f, 1.0f, 1.0f), ImVec4(1.0f, 1.0f, 1.0f, 0.5f));

				ImGui::End();
				});
		}
	}

	void update() override
	{
		assert(!m_UpdateToRenderDependency.has_value());

		// Handle some input:
		if (avk::input().key_pressed(avk::key_code::num0)) {
			// [0] => Copy the input image to the target image and use a semaphore to sync the next draw call
			auto semaphore = avk::context().record_and_submit_with_semaphore({
				// Copy source to target:
				avk::copy_image_to_another(m_InputImageAndSampler->get_image(), avk::layout::general, m_TargetImageAndSampler->get_image(), avk::layout::general),
				}, *m_Queue, avk::stage::auto_stage);

			// We'll wait for it in render():
			m_UpdateToRenderDependency = std::move(semaphore);
		}
		else if (avk::input().key_down(avk::key_code::num1) || avk::input().key_pressed(avk::key_code::num2) || avk::input().key_pressed(avk::key_code::num3)) {
			// [1], [2], or [3] => Use a compute shader to modify the image

			size_t computeIndex = 0;
			if (avk::input().key_pressed(avk::key_code::num2)) { computeIndex = 1; }
			if (avk::input().key_pressed(avk::key_code::num3)) { computeIndex = 2; }

			auto& commandPool = avk::context().get_command_pool_for_single_use_command_buffers(*m_Queue);
			auto cmdbfr = commandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
			cmdbfr->begin_recording();

			auto semaphore = avk::context().record_and_submit_with_semaphore({
				// Bind the compute pipeline:
				avk::command::bind_pipeline(m_ComputePipelines[computeIndex].as_reference()),

				// Bind all the resources:
				avk::command::bind_descriptors(m_ComputePipelines[computeIndex]->layout(), m_DescriptorCache->get_or_create_descriptor_sets({
					avk::descriptor_binding(0, 0, m_InputImageAndSampler->get_image_view()->as_storage_image(avk::layout::general)),
					avk::descriptor_binding(0, 1, m_TargetImageAndSampler->get_image_view()->as_storage_image(avk::layout::general))
				})),

				// Make a dispatch call:
				avk::command::dispatch(m_InputImageAndSampler->width() / 16, m_InputImageAndSampler->height() / 16, 1),
				}, *m_Queue, avk::stage::auto_stage);

			// We'll wait for it in render():
			m_UpdateToRenderDependency = std::move(semaphore);
		}

		if (avk::input().key_pressed(avk::key_code::escape)) {
			// Stop the current composition:
			avk::current_composition()->stop();
		}
	}

	void render() override
	{
		// Update the UBO's data:
		auto* mainWnd = avk::context().main_window();
		const auto w = mainWnd->swap_chain_extent().width;
		const auto halfW = w * 0.5f;
		const auto h = mainWnd->swap_chain_extent().height;

		MatricesForUbo uboVS{};
		uboVS.projection = glm::perspective(glm::radians(60.0f), w * 0.5f / h, 0.1f, 256.0f);
		uboVS.model = glm::translate(glm::mat4{ 1.0f }, glm::vec3(0.0f, 0.0f, -3.0));
		uboVS.model = uboVS.model * glm::rotate(glm::mat4{ 1.0f }, glm::radians(avk::time().time_since_start() * m_RotationSpeed * 90.0f), glm::vec3(0.0f, 1.0f, 0.0f));

		// Update the buffer:
		const auto ifi = mainWnd->current_in_flight_index();
		auto emptyCommands = m_Ubo[ifi]->fill(&uboVS, 0); // We are updating the buffer in host-coherent memory.
		// ^ Because of its memory region (host-coherent), we can be sure that the returned commands are empty. Hence, we do not need to execute them.

		// Get a command pool to allocate command buffers from:
		auto& commandPool = avk::context().get_command_pool_for_single_use_command_buffers(*m_Queue);

		// The swap chain provides us with an "image available semaphore" for the current frame.
		// Only after the swapchain image has become available, we may start rendering into it.
		auto imageAvailableSemaphore = mainWnd->consume_current_image_available_semaphore();

		// Create a command buffer and render into the *current* swap chain image:
		auto cmdBfr = commandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

		// Prepare two viewports:
		auto vpLeft = vk::Viewport{ 0.0f, 0.0f, halfW, static_cast<float>(h) };
		auto vpRight = vk::Viewport{ halfW, 0.0f, halfW, static_cast<float>(h) };

		auto submission = avk::context().record({
			// Begin and end one renderpass:
			avk::command::render_pass(m_GraphicsPipeline->renderpass_reference(), avk::context().main_window()->current_backbuffer_reference(), {

				// Draw left viewport:
				avk::command::custom_commands([&,this](avk::command_buffer_t& cb) { // If there is no avk::command::... struct for a particular command, we can always use avk::command::custom_commands
					cb.handle().setViewport(0, 1, &vpLeft);
				}),

				// Bind a pipeline and perform an indexed draw call:
				avk::command::bind_pipeline(m_GraphicsPipeline.as_reference()),
				avk::command::bind_descriptors(m_GraphicsPipeline->layout(), m_DescriptorCache->get_or_create_descriptor_sets({
					avk::descriptor_binding(0, 0, m_Ubo[ifi]),
					avk::descriptor_binding(0, 1, m_InputImageAndSampler->as_combined_image_sampler(avk::layout::general))
				})),
				avk::command::draw_indexed(m_IndexBuffer.as_reference(), m_VertexBuffer.as_reference()),

					// Draw right viewport:
					avk::command::custom_commands([&,this](avk::command_buffer_t& cb) { // If there is no avk::command::... struct for a particular command, we can always use avk::command::custom_commands
						cb.handle().setViewport(0, 1, &vpRight);
					}),

					// Bind a pipeline and perform an indexed draw call:
					avk::command::bind_pipeline(m_GraphicsPipeline.as_reference()),
					avk::command::bind_descriptors(m_GraphicsPipeline->layout(), m_DescriptorCache->get_or_create_descriptor_sets({
						avk::descriptor_binding(0, 0, m_Ubo[ifi]),
						avk::descriptor_binding(0, 1, m_TargetImageAndSampler->as_combined_image_sampler(avk::layout::general))
					})),
					avk::command::draw_indexed(m_IndexBuffer.as_reference(), m_VertexBuffer.as_reference())
				})

			})
			.into_command_buffer(cmdBfr)
						.then_submit_to(*m_Queue)
						// Do not start to render before the image has become available:
						.waiting_for(imageAvailableSemaphore >> avk::stage::color_attachment_output)
						.store_for_now();

					if (m_UpdateToRenderDependency.has_value()) {
						// If there are some (pending) updates submitted from update(), establish a dependency to them:
						submission.waiting_for(m_UpdateToRenderDependency.value() >> avk::stage::fragment_shader); // Images are read in the fragment_shader stage

						cmdBfr->handle_lifetime_of(std::move(m_UpdateToRenderDependency.value()));
						m_UpdateToRenderDependency.reset();
					}

					submission.submit();

					// Use a convenience function of avk::window to take care of the command buffer's lifetime:
					// It will get deleted in the future after #concurrent-frames have passed by.
					avk::context().main_window()->handle_lifetime(std::move(cmdBfr));
	}

private:

	avk::queue* m_Queue;
	avk::buffer m_VertexBuffer;
	avk::buffer m_IndexBuffer;
	std::vector<avk::buffer> m_Ubo;
	avk::image_sampler m_InputImageAndSampler;
	avk::image_sampler m_TargetImageAndSampler;
	avk::descriptor_cache m_DescriptorCache;

	avk::graphics_pipeline m_GraphicsPipeline;

	std::vector<avk::compute_pipeline> m_ComputePipelines;

	std::optional<avk::semaphore> m_UpdateToRenderDependency;

	float m_RotationSpeed;
};

int main() {

	//run_qmri_ivim();
	int result = EXIT_FAILURE;
	try {
		// Create a window and open it
		auto* mainWnd = avk::context().create_window("Compute Image Effects Example");
		mainWnd->set_resolution({ 1600, 800 });
		mainWnd->set_presentaton_mode(avk::presentation_mode::mailbox);
		mainWnd->set_number_of_concurrent_frames(3u);
		mainWnd->open();

		auto& singleQueue = avk::context().create_queue(vk::QueueFlagBits::eCompute, avk::queue_selection_preference::versatile_queue, mainWnd);
		mainWnd->set_queue_family_ownership(singleQueue.family_index());
		mainWnd->set_present_queue(singleQueue);

		// Create an instance of our main avk::element which contains all the functionality:
		auto app = ComputeAndRenderApp(singleQueue);
		// Create another element for drawing the UI with ImGui
		auto ui = avk::imgui_manager(singleQueue);

		// Compile all the configuration parameters and the invokees into a "composition":
		auto composition = configure_and_compose(
			avk::application_name("Auto-Vk-Toolkit Example: Compute Image Effects Exampl"),
			[](avk::validation_layers& config) {
				config.enable_feature(vk::ValidationFeatureEnableEXT::eSynchronizationValidation);
			},
			// Pass windows:
				mainWnd,
				// Pass invokees:
				app, ui
				);

		// Create an invoker object, which defines the way how invokees/elements are invoked
		// (In this case, just sequentially in their execution order):
		avk::sequential_invoker invoker;

		// With everything configured, let us start our render loop:
		composition.start_render_loop(
			// Callback in the case of update:
			[&invoker](const std::vector<avk::invokee*>& aToBeInvoked) {
				// Call all the update() callbacks:
				invoker.invoke_updates(aToBeInvoked);
			},
			// Callback in the case of render:
				[&invoker](const std::vector<avk::invokee*>& aToBeInvoked) {
				// Sync (wait for fences and so) per window BEFORE executing render callbacks
				avk::context().execute_for_each_window([](avk::window* wnd) {
					wnd->sync_before_render();
					});

				// Call all the render() callbacks:
				invoker.invoke_renders(aToBeInvoked);

				// Render per window:
				avk::context().execute_for_each_window([](avk::window* wnd) {
					wnd->render_frame();
					});
			}
			); // This is a blocking call, which loops until avk::current_composition()->stop(); has been called (see update())

		result = EXIT_SUCCESS;
	}
	catch (avk::logic_error&) {}
	catch (avk::runtime_error&) {}
	return result;

	return 0;
}



