/**
 * @file      vulkanImage.hpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef VULKAN_IMAGE_HPP
#define VULKAN_IMAGE_HPP

#include <core/types.hpp>
#include <volk.h>

#ifdef SAF_CUDA_INTEROP
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace saf
{
    /** @brief A handle to an image held by the Application. */
    using ImageHandle = Handle<struct ImageTag>;

    /** @brief What else accesses an image besides the CPU and ImGui. */
    enum class ImageSharing
    {
        /** @brief Only the CPU and ImGui. */
        None,
#ifdef SAF_CUDA_INTEROP
        /** @brief CUDA kernels, through Application::getCudaSurface and Application::getCudaTexture. */
        Cuda,
#endif
#ifdef SAF_VULKAN_COMPUTE
        /** @brief Compute shaders, through Application::bind. Kept in VK_IMAGE_LAYOUT_GENERAL for its lifetime. */
        Compute,
#endif
    };

    struct VulkanImage
    {
        U32 width;
        U32 height;
        VkFormat format;
        ImageSharing sharing = ImageSharing::None;

        VkImage image                 = VK_NULL_HANDLE;
        VkImageView imageView         = VK_NULL_HANDLE;
        VkDeviceMemory deviceMemory   = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VkSampler sampler             = VK_NULL_HANDLE;

#ifdef SAF_CUDA_INTEROP
        cudaExternalMemory_t cudaExternalImageMemory = {};

        cudaMipmappedArray_t cudaMipmappedImageArray = {};
        cudaSurfaceObject_t cudaSurfaceObject        = {};
        cudaTextureObject_t cudaTextureObject        = {};
#endif
    };
} // namespace saf

#endif // VULKAN_IMAGE_HPP
