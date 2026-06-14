/**
 * @file      vulkanSwapchain.cpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#include "vulkanSwapchain.hpp"

#include <core/helpers.hpp>

using namespace saf;

VkResult VulkanSwapchain::create(const SwapchainCreateInfo& createInfo)
{
    mLogicalDevice           = createInfo.logicalDevice;
    mPhysicalDevice          = createInfo.physicalDevice;
    mQueue                   = createInfo.queue;
    mQueueFamilyIndex        = createInfo.queueFamilyIndex;
    mSurface                 = createInfo.surface;
    mPreferredSurfaceFormat  = createInfo.preferredFormat;
    mImageUsage              = createInfo.imageUsage;
    mPreferredImageCount     = createInfo.preferredImageCount;
    mPreferredFramesInFlight = createInfo.preferredFramesInFlight;

    VkBool32 supportsPresenting = VK_FALSE;

    VK_CHECK_RETURN(vkGetPhysicalDeviceSurfaceSupportKHR(mPhysicalDevice, mQueueFamilyIndex, mSurface, &supportsPresenting));

    if (!supportsPresenting)
    {
        std::cerr << "[vulkan] Error: The selected queue family does not support presenting to the surface." << std::endl;
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    return VK_SUCCESS;
}

void VulkanSwapchain::destroy()
{
    for (auto& swapchainImage : mImages)
    {
        vkDestroyImageView(mLogicalDevice, swapchainImage.imageView, nullptr);
        vkDestroySemaphore(mLogicalDevice, swapchainImage.presentCompleteSemaphore, nullptr);
    }

    mImages.clear();

    for (auto& frameResources : mPerFrameResources)
    {
        vkDestroyFence(mLogicalDevice, frameResources.inFlightFence, nullptr);
        vkDestroySemaphore(mLogicalDevice, frameResources.acquireSemaphore, nullptr);
    }

    mPerFrameResources.clear();

    if (mSwapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(mLogicalDevice, mSwapchain, nullptr);
        mSwapchain = VK_NULL_HANDLE;
    }
}

VkResult VulkanSwapchain::update(VkExtent2D& extent, bool vSyncEnabled)
{
    if (extent.width == 0 || extent.height == 0)
    {
        return VK_SUCCESS; // Ignore zero-sized extents (e.g., when minimizing the window)
    }

    if (extent.width == mExtent.width && extent.height == mExtent.height && vSyncEnabled == mVSyncEnabled)
    {
        return VK_SUCCESS; // No update needed
    }

    mExtent       = extent;
    mVSyncEnabled = vSyncEnabled;

    VK_CHECK_RETURN(createSwapchain());
    VK_CHECK_RETURN(createImageViewsAndPerFrameResources());

    extent = mExtent;

    return VK_SUCCESS;
}

VkResult VulkanSwapchain::acquire()
{
    const VkFence inFlightFence = getCurrentInFlightFence();
    VK_CHECK_RETURN(vkWaitForFences(mLogicalDevice, 1, &inFlightFence, VK_TRUE, std::numeric_limits<U64>::max()));

    const VkSemaphore acquireSemaphore = getCurrentAcquireSemaphore();
    const VkResult result              = vkAcquireNextImageKHR(mLogicalDevice, mSwapchain, std::numeric_limits<U64>::max(), acquireSemaphore, VK_NULL_HANDLE, &mCurrentImage);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
    {
        mNeedsRebuild = true;
    }
    else
    {
        VK_CHECK_RETURN(vkResetFences(mLogicalDevice, 1, &inFlightFence));
    }

    return result;
}

VkResult VulkanSwapchain::present()
{
    const VkSemaphore presentCompleteSemaphore = getCurrentPresentCompleteSemaphore();

    const VkPresentInfoKHR presentInfo{
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = &presentCompleteSemaphore,
        .swapchainCount     = 1,
        .pSwapchains        = &mSwapchain,
        .pImageIndices      = &mCurrentImage
    };

    const VkResult result = vkQueuePresentKHR(mQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
    {
        mNeedsRebuild = true;
    }

    mCurrentFrame = (mCurrentFrame + 1) % mFramesInFlight;

    return result;
}

VkResult VulkanSwapchain::createSwapchain()
{
    VK_CHECK_RETURN(vkDeviceWaitIdle(mLogicalDevice));

    VK_CHECK_RETURN(selectSurfaceFormat());
    VK_CHECK_RETURN(selectPresentMode());

    VK_CHECK_RETURN(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(mPhysicalDevice, mSurface, &mSurfaceCapabilities));

    VK_CHECK_RETURN(selectImageCount());
    mFramesInFlight = std::min(mPreferredFramesInFlight, mImageCount);

    VK_CHECK_RETURN(selectSwapExtent());
    SAF_ASSERT(mExtent.width > 0 && mExtent.height > 0);

    mOldSwapchain = mSwapchain;

    const VkSwapchainCreateInfoKHR swapchainCreateInfo{
        .sType                 = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface               = mSurface,
        .minImageCount         = mImageCount,
        .imageFormat           = mSurfaceFormat.format,
        .imageColorSpace       = mSurfaceFormat.colorSpace,
        .imageExtent           = mExtent,
        .imageArrayLayers      = 1,
        .imageUsage            = mImageUsage,
        .imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 1,
        .pQueueFamilyIndices   = &mQueueFamilyIndex,
        .preTransform          = mSurfaceCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : mSurfaceCapabilities.currentTransform,
        .compositeAlpha        = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode           = mPresentMode,
        .clipped               = VK_TRUE,
        .oldSwapchain          = mOldSwapchain
    };

    VK_CHECK_RETURN(vkCreateSwapchainKHR(mLogicalDevice, &swapchainCreateInfo, nullptr, &mSwapchain));

    if (mOldSwapchain != VK_NULL_HANDLE)
    {
        std::swap(mSwapchain, mOldSwapchain);
        destroy();
        std::swap(mSwapchain, mOldSwapchain);
    }

    return VK_SUCCESS;
}

VkResult VulkanSwapchain::selectSurfaceFormat()
{
    U32 availableCount;
    VK_CHECK_RETURN(vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysicalDevice, mSurface, &availableCount, nullptr));
    std::vector<VkSurfaceFormatKHR> availableFormats;
    availableFormats.resize(static_cast<size_t>(availableCount));
    VK_CHECK_RETURN(vkGetPhysicalDeviceSurfaceFormatsKHR(mPhysicalDevice, mSurface, &availableCount, availableFormats.data()));

    if (availableFormats.size() == 1)
    {
        if (availableFormats[0].format == VK_FORMAT_UNDEFINED)
        {
            VkSurfaceFormatKHR defaultFormat = {};
            defaultFormat.format             = mPreferredSurfaceFormat.format;
            defaultFormat.colorSpace         = mPreferredSurfaceFormat.colorSpace;

            mSurfaceFormat = defaultFormat;

            return VK_SUCCESS;
        }
        else
        {
            mSurfaceFormat = availableFormats[0];
            return VK_SUCCESS;
        }
    }

    for (const auto& availableFormat : availableFormats)
    {
        if (availableFormat.format == mPreferredSurfaceFormat.format && availableFormat.colorSpace == mPreferredSurfaceFormat.colorSpace)
        {
            mSurfaceFormat = availableFormat;
            return VK_SUCCESS;
        }
    }

    for (const auto& availableFormat : availableFormats)
    {
        if (availableFormat.format == mPreferredSurfaceFormat.format)
        {
            mSurfaceFormat = availableFormat;
            return VK_SUCCESS;
        }
    }

    mSurfaceFormat = availableFormats[0];

    return VK_SUCCESS;
}

VkResult VulkanSwapchain::selectPresentMode()
{
    U32 availableCount;
    VK_CHECK_RETURN(vkGetPhysicalDeviceSurfacePresentModesKHR(mPhysicalDevice, mSurface, &availableCount, nullptr));
    std::vector<VkPresentModeKHR> availablePresentModes;
    availablePresentModes.resize(static_cast<size_t>(availableCount));
    VK_CHECK_RETURN(vkGetPhysicalDeviceSurfacePresentModesKHR(mPhysicalDevice, mSurface, &availableCount, availablePresentModes.data()));

    if (!mVSyncEnabled)
    {
        bool immediateAvailable = false;
        bool mailboxAvailable   = false;

        for (const auto& availablePresentMode : availablePresentModes)
        {
            if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
            {
                immediateAvailable = true;
            }
            else if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                mailboxAvailable = true;
            }
        }

        if (mailboxAvailable)
        {
            mPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
            return VK_SUCCESS;
        }
        if (immediateAvailable)
        {
            mPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
            return VK_SUCCESS;
        }
    }

    bool fifoRelaxedAvailable = false;

    for (const auto& availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
        {
            fifoRelaxedAvailable = true;
            break;
        }
    }

    mPresentMode = fifoRelaxedAvailable ? VK_PRESENT_MODE_FIFO_RELAXED_KHR : VK_PRESENT_MODE_FIFO_KHR;

    return VK_SUCCESS;
}

VkResult VulkanSwapchain::selectImageCount()
{
    U32 minImageCount = mSurfaceCapabilities.minImageCount;
    U32 maxImageCount = mSurfaceCapabilities.maxImageCount != 0 ? mSurfaceCapabilities.maxImageCount : std::numeric_limits<U32>::max();

    mImageCount = std::clamp(minImageCount, mPreferredImageCount, maxImageCount);

    return VK_SUCCESS;
}

VkResult VulkanSwapchain::selectSwapExtent()
{
    if (mSurfaceCapabilities.currentExtent.width != std::numeric_limits<U32>::max())
    {
        mExtent = mSurfaceCapabilities.currentExtent;
        return VK_SUCCESS;
    }
    else
    {
        mExtent.width  = std::clamp(mExtent.width, mSurfaceCapabilities.minImageExtent.width, mSurfaceCapabilities.maxImageExtent.width);
        mExtent.height = std::clamp(mExtent.height, mSurfaceCapabilities.minImageExtent.height, mSurfaceCapabilities.maxImageExtent.height);

        return VK_SUCCESS;
    }
}

VkResult VulkanSwapchain::createImageViewsAndPerFrameResources()
{
    VK_CHECK_RETURN(vkGetSwapchainImagesKHR(mLogicalDevice, mSwapchain, &mImageCount, nullptr));
    std::vector<VkImage> swapchainImages;
    swapchainImages.resize(static_cast<size_t>(mImageCount));
    VK_CHECK_RETURN(vkGetSwapchainImagesKHR(mLogicalDevice, mSwapchain, &mImageCount, swapchainImages.data()));

    VkImageViewCreateInfo imageViewCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .viewType         = VK_IMAGE_VIEW_TYPE_2D,
        .format           = mSurfaceFormat.format,
        .components       = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A },
        .subresourceRange = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1 }
    };

    const VkSemaphoreCreateInfo semaphoreCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
    };

    const VkFenceCreateInfo fenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };

    mImages.resize(mImageCount);
    mPerFrameResources.resize(mFramesInFlight);

    for (U32 i = 0; i < mImageCount; i++)
    {
        mImages[i].image = swapchainImages[i];

        imageViewCreateInfo.image = mImages[i].image;

        VK_CHECK_RETURN(vkCreateImageView(mLogicalDevice, &imageViewCreateInfo, nullptr, &mImages[i].imageView));
        VK_CHECK_RETURN(vkCreateSemaphore(mLogicalDevice, &semaphoreCreateInfo, nullptr, &mImages[i].presentCompleteSemaphore));
    }

    for (U32 i = 0; i < mFramesInFlight; i++)
    {
        VK_CHECK_RETURN(vkCreateFence(mLogicalDevice, &fenceCreateInfo, nullptr, &mPerFrameResources[i].inFlightFence));
        VK_CHECK_RETURN(vkCreateSemaphore(mLogicalDevice, &semaphoreCreateInfo, nullptr, &mPerFrameResources[i].acquireSemaphore));
    }

    mCurrentFrame = 0;
    mCurrentImage = 0;
    mNeedsRebuild = false;

    return VK_SUCCESS;
}
