/**
 * @file      vulkanSwapchain.hpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef VULKAN_SWAPCHAIN_HPP
#define VULKAN_SWAPCHAIN_HPP

#include <volk.h>

namespace saf
{
    struct SwapchainCreateInfo
    {
        VkDevice logicalDevice          = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkQueue queue                   = VK_NULL_HANDLE;
        U32 queueFamilyIndex            = 0;
        VkSurfaceKHR surface            = VK_NULL_HANDLE;

        VkSurfaceFormatKHR preferredFormat = { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
        VkImageUsageFlags imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        U32 preferredImageCount     = 3;
        U32 preferredFramesInFlight = 2;
    };

    struct SwapchainImage
    {
        VkImage image                        = VK_NULL_HANDLE;
        VkImageView imageView                = VK_NULL_HANDLE;
        VkSemaphore presentCompleteSemaphore = VK_NULL_HANDLE;
    };

    struct PerFrameResources
    {
        VkFence inFlightFence        = VK_NULL_HANDLE;
        VkSemaphore acquireSemaphore = VK_NULL_HANDLE;
    };

    class VulkanSwapchain
    {
    private:
        VkDevice mLogicalDevice          = VK_NULL_HANDLE;
        VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;

        VkQueue mQueue        = VK_NULL_HANDLE;
        U32 mQueueFamilyIndex = 0;

        VkSurfaceKHR mSurface = VK_NULL_HANDLE;
        VkSurfaceCapabilitiesKHR mSurfaceCapabilities;
        VkSurfaceFormatKHR mPreferredSurfaceFormat;
        VkSurfaceFormatKHR mSurfaceFormat;
        VkImageUsageFlags mImageUsage;

        VkExtent2D mExtent;
        VkPresentModeKHR mPresentMode;
        bool mVSyncEnabled = true;

        U32 mPreferredImageCount = 0;
        U32 mImageCount          = 0;

        U32 mPreferredFramesInFlight = 0;
        U32 mFramesInFlight          = 0;

        U32 mCurrentFrame = 0;
        U32 mCurrentImage = 0;

        VkSwapchainKHR mSwapchain    = VK_NULL_HANDLE;
        VkSwapchainKHR mOldSwapchain = VK_NULL_HANDLE;
        bool mNeedsRebuild           = false;

        std::vector<SwapchainImage> mImages;
        std::vector<PerFrameResources> mPerFrameResources;

    public:
        VulkanSwapchain() = default;
        ~VulkanSwapchain()
        {
            destroy();
        }

        VkResult create(const SwapchainCreateInfo& createInfo);
        void destroy();

        VkResult update(VkExtent2D& extent, bool vSyncEnabled);
        VkResult acquire();
        VkResult present();

        U32 getCurrentImageIndex() const { return mCurrentImage; }
        U32 getCurrentFrameIndex() const { return mCurrentFrame; }

        const SwapchainImage& getCurrentImage() const { return mImages[mCurrentImage]; }
        const PerFrameResources& getCurrentFrameResources() const { return mPerFrameResources[mCurrentFrame]; }
        VkFence getCurrentInFlightFence() const { return mPerFrameResources[mCurrentFrame].inFlightFence; }
        VkSemaphore getCurrentAcquireSemaphore() const { return mPerFrameResources[mCurrentFrame].acquireSemaphore; }
        VkSemaphore getCurrentPresentCompleteSemaphore() const { return mImages[mCurrentImage].presentCompleteSemaphore; }

        VkImage getSwapchainImage(U32 index) const { return mImages[index].image; }
        VkImageView getSwapchainImageView(U32 index) const { return mImages[index].imageView; }

        VkFormat getSurfaceFormat() const { return mSurfaceFormat.format; }
        VkColorSpaceKHR getSurfaceColorSpace() const { return mSurfaceFormat.colorSpace; }
        VkPresentModeKHR getPresentMode() const { return mPresentMode; }
        VkExtent2D getExtent() const { return mExtent; }

        void requestRebuild() { mNeedsRebuild = true; }
        bool needsRebuild() const { return mNeedsRebuild; }

        U32 getImageCount() const { return mImageCount; }
        U32 getFramesInFlight() const { return mFramesInFlight; }

    private:
        VkResult createSwapchain();
        VkResult selectSurfaceFormat();
        VkResult selectPresentMode();
        VkResult selectImageCount();
        VkResult selectSwapExtent();
        VkResult createImageViewsAndPerFrameResources();
    };
} // namespace saf

#endif // VULKAN_SWAPCHAIN_HPP
