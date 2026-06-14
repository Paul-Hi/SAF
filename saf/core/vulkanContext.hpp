/**
 * @file      vulkanContext.hpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef VULKAN_CONTEXT_HPP
#define VULKAN_CONTEXT_HPP

#include <volk.h>

constexpr saf::U32 API_VERSION = VK_API_VERSION_1_3;

namespace saf
{
    struct ExtensionInfo
    {
        const char* extensionName = nullptr;
        void* feature             = nullptr;
    };
    struct ContextCreateInfo
    {
        const char* applicationName                 = "SAF Vulkan Application";
        uint32_t apiVersion                         = API_VERSION;
        std::vector<const char*> instanceExtensions = {};
        std::vector<ExtensionInfo> deviceExtensions = {};
        std::vector<VkQueueFlags> queues            = { VK_QUEUE_GRAPHICS_BIT };

#if NDEBUG
        bool enableValidationLayers = false;
        bool verbose                = false;
#else
        bool enableValidationLayers = true;
        bool verbose                = true;
#endif

#ifdef SAF_CUDA_INTEROP
        bool enableCudaInteroperability = true;
#endif
    };

    struct VulkanQueue
    {
        U32 familyIndex = 0;
        U32 queueIndex  = 0;
        VkQueue queue   = VK_NULL_HANDLE;
    };

    class VulkanContext
    {
    private:
        VkInstance mInstance             = VK_NULL_HANDLE;
        VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
        VkDevice mLogicalDevice          = VK_NULL_HANDLE;

        std::vector<VulkanQueue> mQueues;
        std::vector<std::vector<F32>> mQueuePriorities;
        std::vector<VkDeviceQueueCreateInfo> mQueueCreateInfos;

        VkDebugUtilsMessengerEXT mDebugMessenger = VK_NULL_HANDLE;

    public:
        VulkanContext() = default;
        ~VulkanContext()
        {
            destroy();
        }

        VkResult create(const ContextCreateInfo& createInfo);

        VkInstance getInstance() const
        {
            return mInstance;
        }
        VkPhysicalDevice getPhysicalDevice() const { return mPhysicalDevice; }
        VkDevice getLogicalDevice() const { return mLogicalDevice; }
        const VulkanQueue& getQueue(U32 idx) const { return mQueues[idx]; }

        void destroy();

    private:
        VkResult createInstance(const ContextCreateInfo& createInfo);
        VkResult selectPhysicalDevice(const ContextCreateInfo& createInfo);
        VkResult selectQueueFamilies(const ContextCreateInfo& createInfo);
        VkResult createLogicalDevice(const ContextCreateInfo& createInfo);

#ifdef SAF_CUDA_INTEROP
        VkResult setupVulkanCudaInteropDevice(const Byte* vkDeviceUUID);
#endif

        VkResult checkInstanceExtensionSupport(const std::vector<const char*>& requiredExtensions);
        VkResult checkDeviceExtensionSupport(const std::vector<const char*>& requiredExtensions);

        VkResult printVulkanVersion();
        VkResult printGpus();
        VkResult printQueues();
    };

} // namespace saf

#endif // VULKAN_CONTEXT_HPP
