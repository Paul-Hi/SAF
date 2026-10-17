/**
 * @file      vulkanBuffer.hpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef VULKAN_BUFFER_HPP
#define VULKAN_BUFFER_HPP

#ifdef SAF_VULKAN_COMPUTE

#include <core/types.hpp>
#include <volk.h>

namespace saf
{
    /** @brief A handle to a buffer held by the Application. */
    using BufferHandle = Handle<struct BufferTag>;

    struct VulkanBuffer
    {
        VkDeviceSize size = 0;

        VkBuffer buffer             = VK_NULL_HANDLE;
        VkDeviceMemory deviceMemory = VK_NULL_HANDLE;
    };
} // namespace saf

#endif // SAF_VULKAN_COMPUTE

#endif // VULKAN_BUFFER_HPP
