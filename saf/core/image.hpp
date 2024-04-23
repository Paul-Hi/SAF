/**
 * @file      image.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2024
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef IMAGE_HPP
#define IMAGE_HPP

#ifdef SAF_CUDA_INTEROP
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace saf
{
    class Image
    {
    public:
#ifdef SAF_CUDA_INTEROP
        Image(const std::shared_ptr<class ApplicationContext>& applicationContext, U32 width, U32 height, VkFormat format, const void* data = nullptr, bool shareWithCuda = false);
#else
        Image(const std::shared_ptr<class ApplicationContext>& applicationContext, U32 width, U32 height, VkFormat format, const void* data = nullptr);
#endif

        Image(const std::shared_ptr<class ApplicationContext>& applicationContext, const Str& fileName);
        ~Image();

        VkDescriptorSet getDescriptorSet() const
        {
            return mDescriptorSet;
        }

        VkImage getVkImage() const
        {
            return mImage;
        }

        void update(U32 width, U32 height, VkFormat format, const void* data = nullptr);

        inline U32 getWidth()
        {
            return mWidth;
        }

        inline U32 getHeight()
        {
            return mHeight;
        }

#ifdef SAF_CUDA_INTEROP
        cudaSurfaceObject_t getCudaSurfaceObject() const { return mCudaSurfaceObject; }
        cudaTextureObject_t getCudaTextureObject() const { return mCudaTextureObject; }

        VkSemaphore getVkUpdateCudaSemaphore() const { return mVkUpdateCudaSemaphore; }
        VkSemaphore getCudaUpdateVkSemaphore() const { return mCudaUpdateVkSemaphore; }
        cudaExternalSemaphore_t getCudaExternalVkUpdateCudaSemaphore() const { return mCudaExternalVkUpdateCudaSemaphore; }
        cudaExternalSemaphore_t getCudaExternalCudaUpdateVkSemaphore() const { return mCudaExternalCudaUpdateVkSemaphore; }

        void awaitCudaUpdateClearance(cudaStream_t stream = 0);
        void signalVulkanUpdateClearance(cudaStream_t stream = 0);
#endif

    private:
        void allocateMemory(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer);

        void release();

        void fillFromStagingBuffer(VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkBuffer stagingBuffer, VkExtent3D extent);

    private:
        U32 mWidth;
        U32 mHeight;

        VkImage mImage         = VK_NULL_HANDLE;
        VkImageView mImageView = VK_NULL_HANDLE;
        VkImageLayout mImageLayout;
        VkDeviceMemory mDeviceMemory   = VK_NULL_HANDLE;
        VkDescriptorSet mDescriptorSet = VK_NULL_HANDLE;
        VkSampler mSampler             = VK_NULL_HANDLE;
        VkFormat mFormat;
        VkBuffer mStagingBuffer             = VK_NULL_HANDLE;
        VkDeviceMemory mStagingBufferMemory = VK_NULL_HANDLE;

        std::shared_ptr<class ApplicationContext> mApplicationContext;

#ifdef SAF_CUDA_INTEROP
        bool mShareWithCuda;

        cudaExternalMemory_t mCudaExternalImageMemory;

        cudaMipmappedArray_t mCudaMipmappedImageArray;
        cudaSurfaceObject_t mCudaSurfaceObject;
        cudaTextureObject_t mCudaTextureObject;

        VkSemaphore mVkUpdateCudaSemaphore;
        VkSemaphore mCudaUpdateVkSemaphore;

        cudaExternalSemaphore_t mCudaExternalVkUpdateCudaSemaphore;
        cudaExternalSemaphore_t mCudaExternalCudaUpdateVkSemaphore;

        void createSyncSemaphores();

        void getKhrExtensions();

#ifdef WIN32
        PFN_vkGetSemaphoreWin32HandleKHR getSemaphoreWin32HandleKHR;
        HANDLE getVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType, VkDeviceMemory imageMemory);
        HANDLE getVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType, VkDevice device, VkSemaphore& semaphore);
#else
        PFN_vkGetSemaphoreFdKHR getSemaphoreFdKHR = NULL;
        I32 getVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType, VkDeviceMemory imageMemory);
        I32 getVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType, VkDevice device, VkSemaphore& semaphore);
#endif
#endif

        Str mFilePath;

        PtrSize mAlignedSize;

        VkMemoryRequirements getMemoryRequirements() const;
    };
} // namespace saf

#endif // IMAGE_HPP
