/**
 * @file      vulkanComputePipeline.hpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef VULKAN_COMPUTE_PIPELINE_HPP
#define VULKAN_COMPUTE_PIPELINE_HPP

#ifdef SAF_VULKAN_COMPUTE

#include <core/types.hpp>
#include <core/vulkanBuffer.hpp>
#include <core/vulkanImage.hpp>
#include <vector>
#include <volk.h>

namespace saf
{
    /** @brief A handle to a compute pipeline held by the Application. */
    using ComputePipelineHandle = Handle<struct ComputePipelineTag>;

    struct ComputeBinding
    {
        Str name; // empty if the SPIR-V carries no debug names
        U32 set;
        U32 binding;
        VkDescriptorType type;

        // The bound resource, sticky until rebound; the zero handle means unbound
        ImageHandle image;
        BufferHandle buffer;
    };

    struct VulkanComputePipeline
    {
        VkPipeline pipeline            = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        std::vector<VkDescriptorSetLayout> setLayouts; // sets 0 to the highest used

        std::vector<ComputeBinding> bindings;
        U32 pushConstantSize = 0;        // 0 if the shader has no push constant block
        std::vector<Byte> pushConstants; // empty until set
    };
} // namespace saf

#endif // SAF_VULKAN_COMPUTE

#endif // VULKAN_COMPUTE_PIPELINE_HPP
