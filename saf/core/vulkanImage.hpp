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

#include <volk.h>

#ifdef SAF_CUDA_INTEROP
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace saf
{
    using ImageHandle = U32;

    struct VulkanImage
    {
        U32 width;
        U32 height;
        VkFormat format;
        ImageHandle handle;

        VkImage image                 = VK_NULL_HANDLE;
        VkImageView imageView         = VK_NULL_HANDLE;
        VkDeviceMemory deviceMemory   = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        VkSampler sampler             = VK_NULL_HANDLE;

#ifdef SAF_CUDA_INTEROP
        bool sharedWithCuda;

        cudaExternalMemory_t cudaExternalImageMemory;

        cudaMipmappedArray_t cudaMipmappedImageArray;
        cudaSurfaceObject_t cudaSurfaceObject;
        cudaTextureObject_t cudaTextureObject;

        VkSemaphore vkWaitSemaphore;
        VkSemaphore vkSignalSemaphore;
        cudaExternalSemaphore_t cudaExternalWaitSemaphore;
        cudaExternalSemaphore_t cudaExternalSignalSemaphore;
#endif
    };
} // namespace saf

#endif // VULKAN_IMAGE_HPP