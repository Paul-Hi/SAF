/**
 * @file      image.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2023
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef IMAGE_HPP
#define IMAGE_HPP

namespace saf
{
    class Image
    {
    public:
        Image(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer, U32 width, U32 height, VkFormat format, const void *data = nullptr);
        Image(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer, const Str &fileName);
        ~Image();

        VkDescriptorSet getDescriptorSet() const
        {
            return mDescriptorSet;
        }

        void update(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer, U32 width, U32 height, VkFormat format, const void *data = nullptr);

        inline U32 getWidth()
        {
            return mWidth;
        }

        inline U32 getHeight()
        {
            return mHeight;
        }

    private:
        void allocateMemory(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer);

        void release();

        void fillFromStagingBuffer(VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkBuffer stagingBuffer, VkExtent3D extent);

    private:
        U32 mWidth;
        U32 mHeight;

        VkImage mImage = VK_NULL_HANDLE;
        VkImageView mImageView = VK_NULL_HANDLE;
        VkImageLayout mImageLayout;
        VkDeviceMemory mDeviceMemory = VK_NULL_HANDLE;
        VkDescriptorSet mDescriptorSet = VK_NULL_HANDLE;
        VkSampler mSampler = VK_NULL_HANDLE;
        VkFormat mFormat;
        VkBuffer mStagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory mStagingBufferMemory = VK_NULL_HANDLE;

        VkDevice mDeviceRef;

        Str mFilePath;

        PtrSize mAlignedSize;

        VkMemoryRequirements getMemoryRequirements() const;
    };
} // namespace saf

#endif // IMAGE_HPP
