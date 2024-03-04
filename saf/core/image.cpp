/**
 * @file      image.cpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2024
 * @copyright Apache License 2.0
 */

#include "image.hpp"
#include "immediateSubmit.hpp"
#include <core/vulkanHelper.hpp>
#include <ui/imguiBackend.hpp>

using namespace saf;

Image::Image(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer, U32 width, U32 height, VkFormat format, const void* data)
    : mWidth(width)
    , mHeight(height)
    , mFormat(format)
    , mDeviceRef(logicalDevice)
{
    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageCreateInfo.extent        = { width, height, 1 };
    imageCreateInfo.mipLevels     = 1;
    imageCreateInfo.arrayLayers   = 1;
    imageCreateInfo.format        = format;
    imageCreateInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageCreateInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.flags         = 0;

    VkResult err = vkCreateImage(logicalDevice, &imageCreateInfo, nullptr, &mImage);
    checkVkResult(err);

    allocateMemory(physicalDevice, logicalDevice, queue, commandPool, commandBuffer);

    update(physicalDevice, logicalDevice, queue, commandPool, commandBuffer, width, height, format, data);
}

Image::Image(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer, const Str& fileName)
    : mDeviceRef(logicalDevice)
{
    std::cerr << "Unimplemented" << '\n';
    exit(-1);
}

static U32 findMemoryType(const U32 typeFilter, const VkPhysicalDeviceMemoryProperties memoryProperties, const VkMemoryPropertyFlags propertyFlags)
{
    for (U32 i = 0; i < memoryProperties.memoryTypeCount; ++i)
    {
        if ((typeFilter & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & propertyFlags) == propertyFlags)
        {
            return i;
        }
    }

    std::cerr << "No suitable memory format available" << '\n';
    std::abort();
}

static U32 bytesPerPixel(VkFormat format) // only some formats handled
{
    switch (format)
    {
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_UNORM:
        return 4;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return 16;
    }
    return 0;
}

void Image::update(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer, U32 width, U32 height, VkFormat format, const void* data)
{
    mDeviceRef = logicalDevice;
    VkResult err;

    if (mWidth != width || mHeight != height)
    {
        mWidth  = width;
        mHeight = height;
        release();
        allocateMemory(physicalDevice, logicalDevice, queue, commandPool, commandBuffer);
    }

    SAF_ASSERT(mFormat == format);

    PtrSize uploadSizeInBytes = mWidth * mHeight * bytesPerPixel(format);

    if (!mStagingBuffer)
    {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType              = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size               = uploadSizeInBytes;
        bufferInfo.usage              = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;
        err                           = vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &mStagingBuffer);
        checkVkResult(err);
        VkMemoryRequirements bufferRequirements;
        vkGetBufferMemoryRequirements(logicalDevice, mStagingBuffer, &bufferRequirements);
        VkMemoryAllocateInfo alloc_info = {};
        alloc_info.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize       = bufferRequirements.size;
        mAlignedSize                    = bufferRequirements.size;
        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
        alloc_info.memoryTypeIndex = findMemoryType(bufferRequirements.memoryTypeBits, memoryProperties, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        err                        = vkAllocateMemory(logicalDevice, &alloc_info, nullptr, &mStagingBufferMemory);
        checkVkResult(err);
        err = vkBindBufferMemory(logicalDevice, mStagingBuffer, mStagingBufferMemory, 0);
        checkVkResult(err);
    }

    if (!data)
    {
        ImmediateSubmit::execute(
            logicalDevice, queue, commandPool, commandBuffer, [&](VkCommandBuffer commandBuffer)
            {
            VkImageMemoryBarrier usageBarier = {};
            usageBarier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            usageBarier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            usageBarier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            usageBarier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            usageBarier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            usageBarier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            usageBarier.image = mImage;
            usageBarier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            usageBarier.subresourceRange.levelCount = 1;
            usageBarier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &usageBarier); });
        return;
    }

    // Upload to Buffer
    {
        char* map = NULL;
        err       = vkMapMemory(logicalDevice, mStagingBufferMemory, 0, mAlignedSize, 0, (void**)(&map));
        checkVkResult(err);
        memcpy(map, data, uploadSizeInBytes);
        VkMappedMemoryRange range[1] = {};
        range[0].sType               = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range[0].memory              = mStagingBufferMemory;
        range[0].size                = mAlignedSize;
        err                          = vkFlushMappedMemoryRanges(logicalDevice, 1, range);
        checkVkResult(err);
        vkUnmapMemory(logicalDevice, mStagingBufferMemory);
    }

    fillFromStagingBuffer(logicalDevice, queue, commandPool, commandBuffer, mStagingBuffer, VkExtent3D{ mWidth, mHeight, 1 });
}

void Image::allocateMemory(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer)
{
    VkResult err;
    // Image and Device Memory
    {
        const VkMemoryRequirements requirements = getMemoryRequirements();
        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType                = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize       = requirements.size;
        allocInfo.memoryTypeIndex      = findMemoryType(requirements.memoryTypeBits, memoryProperties, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        err                            = vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &mDeviceMemory);
        checkVkResult(err);
        err = vkBindImageMemory(logicalDevice, mImage, mDeviceMemory, 0);
        checkVkResult(err);
    }

    // Image View
    {
        VkImageViewCreateInfo info       = {};
        info.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.image                       = mImage;
        info.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
        info.format                      = mFormat;
        info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.layerCount = 1;
        err                              = vkCreateImageView(logicalDevice, &info, nullptr, &mImageView);
        checkVkResult(err);
    }

    // Sampler
    {
        VkSamplerCreateInfo info = {};
        info.sType               = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter           = VK_FILTER_LINEAR;
        info.minFilter           = VK_FILTER_LINEAR;
        info.mipmapMode          = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.addressModeU        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeV        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeW        = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.minLod              = -1000;
        info.maxLod              = 1000;
        info.maxAnisotropy       = 1.0f;
        err                      = vkCreateSampler(logicalDevice, &info, nullptr, &mSampler);
        checkVkResult(err);
    }

    // Descriptor Set:
    mDescriptorSet = static_cast<VkDescriptorSet>(vkAddTexture(mSampler, mImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
}

void Image::fillFromStagingBuffer(VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkBuffer stagingBuffer, VkExtent3D extent)
{
    ImmediateSubmit::execute(
        logicalDevice, queue, commandPool, commandBuffer, [&](VkCommandBuffer commandBuffer)
        {
            VkImageMemoryBarrier copyBarrier = {};
            copyBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            copyBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            copyBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            copyBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            copyBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            copyBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            copyBarrier.image = mImage;
            copyBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyBarrier.subresourceRange.levelCount = 1;
            copyBarrier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &copyBarrier);

            VkBufferImageCopy copyRegion = {};
            copyRegion.bufferOffset = 0;
            copyRegion.bufferRowLength = 0;
            copyRegion.bufferImageHeight = 0;

            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.imageSubresource.mipLevel = 0;
            copyRegion.imageSubresource.baseArrayLayer = 0;
            copyRegion.imageSubresource.layerCount = 1;

            copyRegion.imageOffset = {0, 0, 0};
            copyRegion.imageExtent = extent;

            vkCmdCopyBufferToImage(commandBuffer, mStagingBuffer, mImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

            VkImageMemoryBarrier usageBarier = {};
            usageBarier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            usageBarier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            usageBarier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            usageBarier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            usageBarier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            usageBarier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            usageBarier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            usageBarier.image = mImage;
            usageBarier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            usageBarier.subresourceRange.levelCount = 1;
            usageBarier.subresourceRange.layerCount = 1;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &usageBarier); });
}

void Image::release()
{
    if (mImage)
    {
        vkDestroySampler(mDeviceRef, mSampler, nullptr);
        vkDestroyImageView(mDeviceRef, mImageView, nullptr);
        vkDestroyImage(mDeviceRef, mImage, nullptr);
        vkFreeMemory(mDeviceRef, mDeviceMemory, nullptr);
        vkDestroyBuffer(mDeviceRef, mStagingBuffer, nullptr);
        vkFreeMemory(mDeviceRef, mStagingBufferMemory, nullptr);
    }
}

Image::~Image()
{
    release();
}

VkMemoryRequirements Image::getMemoryRequirements() const
{
    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(mDeviceRef, mImage, &memoryRequirements);

    return memoryRequirements;
}
