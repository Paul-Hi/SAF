/**
 * @file      vulkanHelper.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2024
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef VULKAN_HELPER_HPP
#define VULKAN_HELPER_HPP

#include <vulkan/vulkan.h>

struct ImDrawData;

namespace saf
{
    struct VulkanInitInfo
    {
        VkInstance instance;
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        U32 queueFamily;
        VkQueue queue;
        VkPipelineCache pipelineCache;
        VkDescriptorPool descriptorPool;
        U32 subpass;
        U32 minImageCount;                 // >= 2
        U32 imageCount;                    // >= MinImageCount
        VkSampleCountFlagBits msaaSamples; // >= VK_SAMPLE_COUNT_1_BIT (0 -> default to VK_SAMPLE_COUNT_1_BIT)

        bool useDynamicRendering;
        VkFormat colorAttachmentFormat;

        const VkAllocationCallbacks* allocator;
        void (*checkVkResultFn)(VkResult err);
    };

    // Context
    struct VulkanFrameData
    {
        VkCommandPool commandPool;
        VkCommandBuffer commandBuffer;
        VkFence fence;
        VkImage backbuffer;
        VkImageView backbufferView;
        VkFramebuffer framebuffer;

        VulkanFrameData()
        {
            memset(static_cast<void*>(this), 0, sizeof(*this));
        }
    };

    struct VulkanFrameSemaphores
    {
        VkSemaphore imageAcquiredSemaphore;
        VkSemaphore renderCompleteSemaphore;

        VulkanFrameSemaphores()
        {
            memset(static_cast<void*>(this), 0, sizeof(*this));
        }
    };

    struct VulkanContext
    {
        I32 width;
        I32 height;
        VkSwapchainKHR swapchain;
        VkSurfaceKHR surface;
        VkSurfaceFormatKHR surfaceFormat;
        VkPresentModeKHR presentMode;
        VkRenderPass renderPass;
        VkPipeline pipeline;
        bool useDynamicRendering;
        bool clearEnable;
        VkClearValue clearValue;
        U32 frameIndex;
        U32 framesInFlight;
        U32 semaphoreCount; // framesInFlight + 1
        U32 semaphoreIndex;
        VulkanFrameData* frames;
        VulkanFrameSemaphores* frameSemaphores;

        VkCommandPool ressourceCommandPool;
        VkCommandBuffer ressourceCommandBuffer;

        VulkanContext()
        {
            memset(static_cast<void*>(this), 0, sizeof(*this));
            presentMode = static_cast<VkPresentModeKHR>(~0);
            clearEnable = true;
        }
    };

    bool vkInit(VulkanInitInfo* info, VkRenderPass renderPass);
    void vkShutdown();
    void vkNewFrame();
    void vkRenderImGuiDrawData(ImDrawData* drawData, VkCommandBuffer commandBuffer, VkPipeline pipeline = VK_NULL_HANDLE);
    bool vkCreateImGuiFontsTexture(VkCommandBuffer commandBuffer);
    void vkDestroyImGuiFontUploadObjects();
    void vkSetMinImageCount(U32 minImageCount);

    VkDescriptorSet vkAddTexture(VkSampler sampler, VkImageView imageView, VkImageLayout imageLayout);
    void vkRemoveTexture(VkDescriptorSet descriptorSet);

    // Optional: load Vulkan functions with a custom function loader
    // This is only useful with IMGUI_IMPL_VULKAN_NO_PROTOTYPES / VK_NO_PROTOTYPES
    bool vkLoadFunctions(PFN_vkVoidFunction (*loaderFunc)(const char* functionName, void* userData), void* userData = nullptr);

    void vkCreateOrResizeContext(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VulkanContext* context, U32 queueFamily, const VkAllocationCallbacks* allocator, I32 width, I32 height, U32 minImageCount);
    void vkDestroyContext(VkInstance instance, VkDevice logicalDevice, VulkanContext* context, const VkAllocationCallbacks* allocator);
    VkSurfaceFormatKHR vkSelectSurfaceFormat(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, const VkFormat* requestedFormats, I32 requestedFormatsCount, VkColorSpaceKHR requestedColorSpace);
    VkPresentModeKHR vkSelectPresentMode(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, const VkPresentModeKHR* requestModes, I32 requestedModesCount);
    I32 vkGetMinImageCountFromPresentMode(VkPresentModeKHR presentMode);

} // namespace saf

#endif // VULKAN_HELPER_HPP
