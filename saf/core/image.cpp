/**
 * @file      image.cpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2025
 * @copyright Apache License 2.0
 */

#include "image.hpp"
#include "immediateSubmit.hpp"
#include <app/application.hpp>
#include <core/vulkanHelper.hpp>
#include <ui/imguiBackend.hpp>

#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define NOMINMAX
#include <VersionHelpers.h>
#include <aclapi.h>
#include <dxgi1_2.h>
#include <windows.h>
#define _USE_MATH_DEFINES
#endif

using namespace saf;

#ifndef SAF_CUDA_INTEROP
Image::Image(const std::shared_ptr<ApplicationContext>& applicationContext, U32 width, U32 height, VkFormat format, const void* data)
    : mApplicationContext(applicationContext)
    , mWidth(0)
    , mHeight(0)
    , mFormat(format)
    , mStagingBuffer(nullptr)
{
    update(width, height, format, data);
}
#else
Image::Image(const std::shared_ptr<ApplicationContext>& applicationContext, U32 width, U32 height, VkFormat format, const void* data, bool shareWithCuda)
    : mApplicationContext(applicationContext)
    , mWidth(0)
    , mHeight(0)
    , mFormat(format)
    , mStagingBuffer(nullptr)
    , mShareWithCuda(shareWithCuda)
{
    update(width, height, format, data);
}
#endif

Image::Image(const std::shared_ptr<ApplicationContext>& applicationContext, const Str& fileName)
    : mApplicationContext(applicationContext)
{
    std::cerr << "Unimplemented" << '\n';
    exit(-1);
}

static U32 findMemoryType(const U32 typeFilter, VkPhysicalDeviceMemoryProperties memoryProperties, const VkMemoryPropertyFlags propertyFlags)
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

static U32 bitsPerComponent(VkFormat format) // only some formats handled
{
    switch (format)
    {
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_UNORM:
        return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return 32;
    }
    return 0;
}

void Image::update(U32 width, U32 height, VkFormat format, const void* data)
{
    VkResult err;

    if (mWidth != width || mHeight != height)
    {
#ifdef SAF_CUDA_INTEROP
        mApplicationContext->deregisterImage(mImage);
#endif

        vkDeviceWaitIdle(mApplicationContext->mDeviceRef);
        mWidth  = width;
        mHeight = height;
        release();

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

#ifdef SAF_CUDA_INTEROP
        VkExternalMemoryImageCreateInfo externalMemImageCreateInfo{};
        if (mShareWithCuda)
        {
            externalMemImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
#ifdef WIN32
            externalMemImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
            externalMemImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
            imageCreateInfo.pNext = &externalMemImageCreateInfo;
        }
#endif

        VkResult err = vkCreateImage(mApplicationContext->mDeviceRef, &imageCreateInfo, nullptr, &mImage);
        checkVkResult(err);

#ifdef SAF_CUDA_INTEROP
        if (mShareWithCuda)
        {
            getKhrExtensions();
            createSyncSemaphores();

            ApplicationContext::ImageSemaphores imageSemaphores;

            imageSemaphores.vkUpdateCudaSemaphore             = getVkUpdateCudaSemaphore();
            imageSemaphores.cudaUpdateVkSemaphore             = getCudaUpdateVkSemaphore();
            imageSemaphores.cudaExternalVkUpdateCudaSemaphore = getCudaExternalVkUpdateCudaSemaphore();
            imageSemaphores.cudaExternalCudaUpdateVkSemaphore = getCudaExternalCudaUpdateVkSemaphore();

            mApplicationContext->registerImage(mImage, imageSemaphores);
        }
#endif

        allocateMemory(mApplicationContext->mPhysicalDeviceRef, mApplicationContext->mDeviceRef, mApplicationContext->mQueueRef, mApplicationContext->mCommandPoolRef, mApplicationContext->mCommandBufferRef);
    }

    SAF_ASSERT(mFormat == format);

    PtrSize uploadSizeInBytes = mWidth * mHeight * bytesPerPixel(format);

    if (!mStagingBuffer)
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size        = uploadSizeInBytes;
        bufferInfo.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        err                    = vkCreateBuffer(mApplicationContext->mDeviceRef, &bufferInfo, nullptr, &mStagingBuffer);
        checkVkResult(err);
        VkMemoryRequirements bufferRequirements;
        vkGetBufferMemoryRequirements(mApplicationContext->mDeviceRef, mStagingBuffer, &bufferRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = bufferRequirements.size;
        mAlignedSize             = bufferRequirements.size;
        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(mApplicationContext->mPhysicalDeviceRef, &memoryProperties);
        allocInfo.memoryTypeIndex = findMemoryType(bufferRequirements.memoryTypeBits, memoryProperties, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        err                       = vkAllocateMemory(mApplicationContext->mDeviceRef, &allocInfo, nullptr, &mStagingBufferMemory);
        checkVkResult(err);
        err = vkBindBufferMemory(mApplicationContext->mDeviceRef, mStagingBuffer, mStagingBufferMemory, 0);
        checkVkResult(err);
    }

    if (!data)
    {
        ImmediateSubmit::execute(
            mApplicationContext->mDeviceRef, mApplicationContext->mQueueRef, mApplicationContext->mCommandPoolRef, mApplicationContext->mCommandBufferRef, [&](VkCommandBuffer commandBuffer)
            {
            VkImageMemoryBarrier usageBarier{};
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
        err       = vkMapMemory(mApplicationContext->mDeviceRef, mStagingBufferMemory, 0, mAlignedSize, 0, (void**)(&map));
        checkVkResult(err);
        memcpy(map, data, uploadSizeInBytes);
        VkMappedMemoryRange range[1]{};
        range[0].sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range[0].memory = mStagingBufferMemory;
        range[0].size   = mAlignedSize;
        err             = vkFlushMappedMemoryRanges(mApplicationContext->mDeviceRef, 1, range);
        checkVkResult(err);
        vkUnmapMemory(mApplicationContext->mDeviceRef, mStagingBufferMemory);
    }

    fillFromStagingBuffer(mApplicationContext->mDeviceRef, mApplicationContext->mQueueRef, mApplicationContext->mCommandPoolRef, mApplicationContext->mCommandBufferRef, mStagingBuffer, VkExtent3D{ mWidth, mHeight, 1 });
}

std::vector<Byte> Image::downloadImageData() const
{
    VkDeviceSize imageSize = mWidth * mHeight * bytesPerPixel(mFormat);

    std::vector<Byte> imageData(imageSize);

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageCreateInfo.extent        = { mWidth, mHeight, 1 };
    imageCreateInfo.mipLevels     = 1;
    imageCreateInfo.arrayLayers   = 1;
    imageCreateInfo.format        = mFormat;
    imageCreateInfo.tiling        = VK_IMAGE_TILING_LINEAR;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageCreateInfo.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageCreateInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.flags         = 0;

    VkImage dstImage;
    VkResult err = vkCreateImage(mApplicationContext->mDeviceRef, &imageCreateInfo, nullptr, &dstImage);
    checkVkResult(err);

    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(mApplicationContext->mDeviceRef, dstImage, &requirements);
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(mApplicationContext->mPhysicalDeviceRef, &memoryProperties);

    VkMemoryAllocateInfo memAllocInfo{};
    memAllocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAllocInfo.allocationSize  = requirements.size;
    memAllocInfo.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, memoryProperties, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkDeviceMemory dstImageMemory;
    err = vkAllocateMemory(mApplicationContext->mDeviceRef, &memAllocInfo, nullptr, &dstImageMemory);
    checkVkResult(err);

    err = vkBindImageMemory(mApplicationContext->mDeviceRef, dstImage, dstImageMemory, 0);
    checkVkResult(err);

    ImmediateSubmit::execute(
        mApplicationContext->mDeviceRef, mApplicationContext->mQueueRef, mApplicationContext->mCommandPoolRef, mApplicationContext->mCommandBufferRef, [&](VkCommandBuffer commandBuffer)
        {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.image = dstImage;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);

            barrier.image = mImage;
            barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);

            VkImageCopy copyRegion{};
            copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.srcSubresource.layerCount = 1;
            copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.dstSubresource.layerCount = 1;
            copyRegion.extent = { mWidth, mHeight, 1 };
            copyRegion.srcSubresource.mipLevel = 0;
            copyRegion.srcSubresource.baseArrayLayer = 0;
            copyRegion.dstSubresource.mipLevel = 0;
            copyRegion.dstSubresource.baseArrayLayer = 0;

            vkCmdCopyImage(commandBuffer, mImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

            barrier.image = dstImage;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);

            barrier.image = mImage;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier); });

    VkImageSubresource subResource{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
    VkSubresourceLayout subResourceLayout;
    vkGetImageSubresourceLayout(mApplicationContext->mDeviceRef, dstImage, &subResource, &subResourceLayout);

    const char* mappedData;
    err = vkMapMemory(mApplicationContext->mDeviceRef, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&mappedData);
    checkVkResult(err);

    mappedData += subResourceLayout.offset;
    memcpy(imageData.data(), mappedData, imageData.size());

    vkUnmapMemory(mApplicationContext->mDeviceRef, dstImageMemory);
    vkFreeMemory(mApplicationContext->mDeviceRef, dstImageMemory, nullptr);
    vkDestroyImage(mApplicationContext->mDeviceRef, dstImage, nullptr);

    return imageData;
}

void Image::allocateMemory(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer)
{
    VkResult err;
    // Image and Device Memory
    {
        const VkMemoryRequirements requirements = getMemoryRequirements();
        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.pNext = NULL;

#ifdef SAF_CUDA_INTEROP
        VkExportMemoryAllocateInfoKHR exportMemoryAllocateInfoKHR{};
        if (mShareWithCuda)
        {
#ifdef WIN32
            WindowsSecurityAttributes winSecurityAttributes;

            VkExportMemoryWin32HandleInfoKHR exportMemoryWin32HandleInfoKHR{};
            exportMemoryWin32HandleInfoKHR.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
            exportMemoryWin32HandleInfoKHR.pNext       = NULL;
            exportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
            exportMemoryWin32HandleInfoKHR.dwAccess    = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
            exportMemoryWin32HandleInfoKHR.name        = (LPCWSTR)NULL;
#endif
            exportMemoryAllocateInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#ifdef WIN32
            exportMemoryAllocateInfoKHR.pNext       = IsWindows8OrGreater() ? &exportMemoryWin32HandleInfoKHR : NULL;
            exportMemoryAllocateInfoKHR.handleTypes = IsWindows8OrGreater() ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
            exportMemoryAllocateInfoKHR.pNext       = NULL;
            exportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
#endif
            allocInfo.pNext = &exportMemoryAllocateInfoKHR;
        }
#endif

        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = requirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, memoryProperties, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        err                       = vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &mDeviceMemory);
        checkVkResult(err);
        err = vkBindImageMemory(logicalDevice, mImage, mDeviceMemory, 0);
        checkVkResult(err);

        // ... and import in CUDA
#ifdef SAF_CUDA_INTEROP
        if (mShareWithCuda)
        {
            cudaExternalMemoryHandleDesc cudaExternalMemHandleDesc;
            memset(&cudaExternalMemHandleDesc, 0, sizeof(cudaExternalMemHandleDesc));
#ifdef WIN32
            cudaExternalMemHandleDesc.type                = IsWindows8OrGreater() ? cudaExternalMemoryHandleTypeOpaqueWin32 : cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
            cudaExternalMemHandleDesc.handle.win32.handle = getVkImageMemHandle(IsWindows8OrGreater() ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT, mDeviceMemory);
#else
            cudaExternalMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;

            cudaExternalMemHandleDesc.handle.fd = getVkImageMemHandle(VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR, mDeviceMemory);
#endif
            cudaExternalMemHandleDesc.size = requirements.size;

            CUDA_CHECK(cudaImportExternalMemory(&mCudaExternalImageMemory, &cudaExternalMemHandleDesc));

            cudaExtent extent = make_cudaExtent(mWidth, mHeight, 0);
            cudaChannelFormatDesc formatDesc;
            formatDesc.x = bitsPerComponent(mFormat);
            formatDesc.y = bitsPerComponent(mFormat);
            formatDesc.z = bitsPerComponent(mFormat);
            formatDesc.w = bitsPerComponent(mFormat);
            formatDesc.f = cudaChannelFormatKindUnsigned;

            cudaExternalMemoryMipmappedArrayDesc externalMemoryMipmappedArrayDesc{};
            externalMemoryMipmappedArrayDesc.offset     = 0;
            externalMemoryMipmappedArrayDesc.formatDesc = formatDesc;
            externalMemoryMipmappedArrayDesc.extent     = extent;
            externalMemoryMipmappedArrayDesc.flags      = 0;
            externalMemoryMipmappedArrayDesc.numLevels  = 1;

            CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&mCudaMipmappedImageArray, mCudaExternalImageMemory, &externalMemoryMipmappedArrayDesc));

            cudaArray_t cudaMipLevelArray;
            CUDA_CHECK(cudaGetMipmappedArrayLevel(&cudaMipLevelArray, mCudaMipmappedImageArray, 0));

            cudaResourceDesc resourceDesc{};

            resourceDesc.resType         = cudaResourceTypeArray;
            resourceDesc.res.array.array = cudaMipLevelArray;

            CUDA_CHECK(cudaCreateSurfaceObject(&mCudaSurfaceObject, &resourceDesc));

            cudaResourceDesc resDescr{};

            resDescr.resType           = cudaResourceTypeMipmappedArray;
            resDescr.res.mipmap.mipmap = mCudaMipmappedImageArray;

            cudaTextureDesc texDescr{};
            texDescr.normalizedCoords = true;
            texDescr.filterMode       = cudaFilterModePoint;
            texDescr.mipmapFilterMode = cudaFilterModePoint;

            texDescr.addressMode[0] = cudaAddressModeWrap;
            texDescr.addressMode[1] = cudaAddressModeWrap;

            texDescr.maxMipmapLevelClamp = 0;

            texDescr.readMode = cudaReadModeElementType;

            CUDA_CHECK(cudaCreateTextureObject(&mCudaTextureObject, &resDescr, &texDescr, NULL));
        }
#endif
    }

    // Image View
    {
        VkImageViewCreateInfo info{};
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
        VkSamplerCreateInfo info{};
        info.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.magFilter     = VK_FILTER_LINEAR;
        info.minFilter     = VK_FILTER_LINEAR;
        info.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        info.addressModeU  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeV  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.addressModeW  = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        info.minLod        = -1000;
        info.maxLod        = 1000;
        info.maxAnisotropy = 1.0f;
        err                = vkCreateSampler(logicalDevice, &info, nullptr, &mSampler);
        checkVkResult(err);
    }

    // Descriptor Set:
    mDescriptorSet = static_cast<VkDescriptorSet>(vkAddTexture(mSampler, mImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
}

void Image::fillFromStagingBuffer(VkDevice logicalDevice, VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkBuffer stagingBuffer, VkExtent3D extent)
{
    ImmediateSubmit::execute(
        mApplicationContext->mDeviceRef, mApplicationContext->mQueueRef, mApplicationContext->mCommandPoolRef, mApplicationContext->mCommandBufferRef, [&](VkCommandBuffer commandBuffer)
        {
            VkImageMemoryBarrier copyBarrier{};
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

            VkBufferImageCopy copyRegion{};
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

            VkImageMemoryBarrier usageBarier{};
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
#ifdef SAF_CUDA_INTEROP
        if (mShareWithCuda)
        {
            CUDA_CHECK(cudaDestroyExternalSemaphore(mCudaExternalCudaUpdateVkSemaphore));
            CUDA_CHECK(cudaDestroyExternalSemaphore(mCudaExternalVkUpdateCudaSemaphore));

            vkDestroySemaphore(mApplicationContext->mDeviceRef, mCudaUpdateVkSemaphore, nullptr);
            vkDestroySemaphore(mApplicationContext->mDeviceRef, mVkUpdateCudaSemaphore, nullptr);

            CUDA_CHECK(cudaDestroyExternalMemory(mCudaExternalImageMemory));
        }
#endif
        vkDestroySampler(mApplicationContext->mDeviceRef, mSampler, nullptr);
        vkDestroyImageView(mApplicationContext->mDeviceRef, mImageView, nullptr);
        vkDestroyImage(mApplicationContext->mDeviceRef, mImage, nullptr);
        vkFreeMemory(mApplicationContext->mDeviceRef, mDeviceMemory, nullptr);
        vkDestroyBuffer(mApplicationContext->mDeviceRef, mStagingBuffer, nullptr);
        vkFreeMemory(mApplicationContext->mDeviceRef, mStagingBufferMemory, nullptr);
        mStagingBuffer = nullptr;
    }
}

Image::~Image()
{
#ifdef SAF_CUDA_INTEROP
    mApplicationContext->deregisterImage(mImage);
#endif
    release();
}

#ifdef SAF_CUDA_INTEROP
void Image::getKhrExtensions()
{
#ifdef WIN32
    getSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(mApplicationContext->mDeviceRef, "vkGetSemaphoreWin32HandleKHR");
    if (getSemaphoreWin32HandleKHR == NULL)
    {
        std::cerr << "Vulkan: Proc address for \"vkGetSemaphoreWin32HandleKHR\" not found.\n";
    }
#else
    getSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(mApplicationContext->mDeviceRef, "vkGetSemaphoreFdKHR");
    if (getSemaphoreFdKHR == NULL)
    {
        std::cerr << "Vulkan: Proc address for \"vkGetSemaphoreFdKHR\" not found.\n";
    }
#endif
}

#ifdef WIN32
HANDLE Image::getVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType, VkDevice device, VkDeviceMemory imageMemory)
{
    HANDLE handle;

    VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR{};
    vkMemoryGetWin32HandleInfoKHR.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    vkMemoryGetWin32HandleInfoKHR.pNext      = NULL;
    vkMemoryGetWin32HandleInfoKHR.memory     = imageMemory;
    vkMemoryGetWin32HandleInfoKHR.handleType = (VkExternalMemoryHandleTypeFlagBitsKHR)externalMemoryHandleType;

    mApplicationContext->getMemoryWin32HandleKHR(&vkMemoryGetWin32HandleInfoKHR, &handle);
    return handle;
}
HANDLE Image::getVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType, VkDevice device, VkSemaphore& semaphore)
{
    HANDLE handle;

    VkSemaphoreGetWin32HandleInfoKHR vulkanSemaphoreGetWin32HandleInfoKHR{};
    vulkanSemaphoreGetWin32HandleInfoKHR.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    vulkanSemaphoreGetWin32HandleInfoKHR.pNext      = NULL;
    vulkanSemaphoreGetWin32HandleInfoKHR.semaphore  = semaphore;
    vulkanSemaphoreGetWin32HandleInfoKHR.handleType = externalSemaphoreHandleType;

    getSemaphoreWin32HandleKHR(device, &vulkanSemaphoreGetWin32HandleInfoKHR,
                               &handle);

    return handle;
}
#else
I32 Image::getVkImageMemHandle(VkExternalMemoryHandleTypeFlagsKHR externalMemoryHandleType, VkDeviceMemory imageMemory)
{
    if (externalMemoryHandleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR)
    {
        I32 fd;

        VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR{};
        vkMemoryGetFdInfoKHR.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        vkMemoryGetFdInfoKHR.pNext      = NULL;
        vkMemoryGetFdInfoKHR.memory     = imageMemory;
        vkMemoryGetFdInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

        mApplicationContext->getMemoryFdKHR(&vkMemoryGetFdInfoKHR, &fd);

        return fd;
    }

    return -1;
}

I32 Image::getVkSemaphoreHandle(VkExternalSemaphoreHandleTypeFlagBitsKHR externalSemaphoreHandleType, VkDevice device, VkSemaphore& semaphore)
{
    if (externalSemaphoreHandleType == VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT)
    {
        I32 fd;

        VkSemaphoreGetFdInfoKHR vulkanSemaphoreGetFdInfoKHR{};
        vulkanSemaphoreGetFdInfoKHR.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
        vulkanSemaphoreGetFdInfoKHR.pNext      = NULL;
        vulkanSemaphoreGetFdInfoKHR.semaphore  = semaphore;
        vulkanSemaphoreGetFdInfoKHR.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

        getSemaphoreFdKHR(device, &vulkanSemaphoreGetFdInfoKHR, &fd);

        return fd;
    }
    return -1;
}
#endif

void Image::awaitCudaUpdateClearance(cudaStream_t stream)
{
    cudaExternalSemaphoreWaitParams externalSemaphoreWaitParams;

    memset(&externalSemaphoreWaitParams, 0, sizeof(externalSemaphoreWaitParams));

    externalSemaphoreWaitParams.params.fence.value = 0;
    externalSemaphoreWaitParams.flags              = 0;

    CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&mCudaExternalVkUpdateCudaSemaphore, &externalSemaphoreWaitParams, 1, stream));
}

void Image::signalVulkanUpdateClearance(cudaStream_t stream)
{
    cudaExternalSemaphoreSignalParams externalSemaphoreSignalParams;
    memset(&externalSemaphoreSignalParams, 0, sizeof(externalSemaphoreSignalParams));

    externalSemaphoreSignalParams.params.fence.value = 0;
    externalSemaphoreSignalParams.flags              = 0;

    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&mCudaExternalCudaUpdateVkSemaphore, &externalSemaphoreSignalParams, 1, stream));
}

void Image::createSyncSemaphores()
{
    // Create Semaphores in Vulkan
    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

#ifdef WIN32
    WindowsSecurityAttributes winSecurityAttributes;

    VkExportSemaphoreWin32HandleInfoKHR exportSemaphoreWin32HandleInfoKHR{};
    exportSemaphoreWin32HandleInfoKHR.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
    exportSemaphoreWin32HandleInfoKHR.pNext       = NULL;
    exportSemaphoreWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
    exportSemaphoreWin32HandleInfoKHR.dwAccess    = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    exportSemaphoreWin32HandleInfoKHR.name        = (LPCWSTR)NULL;
#endif
    VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo{};
    exportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
#ifdef WIN32
    exportSemaphoreCreateInfo.pNext       = IsWindows8OrGreater() ? &exportSemaphoreWin32HandleInfoKHR : NULL;
    exportSemaphoreCreateInfo.handleTypes = IsWindows8OrGreater() ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
    exportSemaphoreCreateInfo.pNext = NULL;
    exportSemaphoreCreateInfo.handleTypes =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    semaphoreCreateInfo.pNext = &exportSemaphoreCreateInfo;

    VkResult err = vkCreateSemaphore(mApplicationContext->mDeviceRef, &semaphoreCreateInfo, nullptr, &mCudaUpdateVkSemaphore);
    checkVkResult(err);
    err = vkCreateSemaphore(mApplicationContext->mDeviceRef, &semaphoreCreateInfo, nullptr, &mVkUpdateCudaSemaphore);
    checkVkResult(err);

    // import semaphores in CUDA
    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
    memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));

#ifdef WIN32
    externalSemaphoreHandleDesc.type                = IsWindows8OrGreater() ? cudaExternalSemaphoreHandleTypeOpaqueWin32 : cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(IsWindows8OrGreater() ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT, mApplicationContext->mDeviceRef, mCudaUpdateVkSemaphore);
#else
    externalSemaphoreHandleDesc.type      = cudaExternalSemaphoreHandleTypeOpaqueFd;
    externalSemaphoreHandleDesc.handle.fd = getVkSemaphoreHandle(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, mApplicationContext->mDeviceRef, mCudaUpdateVkSemaphore);
#endif
    externalSemaphoreHandleDesc.flags = 0;

    CUDA_CHECK(cudaImportExternalSemaphore(&mCudaExternalCudaUpdateVkSemaphore, &externalSemaphoreHandleDesc));

    memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
#ifdef WIN32
    externalSemaphoreHandleDesc.type                = IsWindows8OrGreater() ? cudaExternalSemaphoreHandleTypeOpaqueWin32 : cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    externalSemaphoreHandleDesc.handle.win32.handle = getVkSemaphoreHandle(IsWindows8OrGreater() ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT, mApplicationContext->mDeviceRef, mVkUpdateCudaSemaphore);
#else
    externalSemaphoreHandleDesc.type      = cudaExternalSemaphoreHandleTypeOpaqueFd;
    externalSemaphoreHandleDesc.handle.fd = getVkSemaphoreHandle(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, mApplicationContext->mDeviceRef, mVkUpdateCudaSemaphore);
#endif
    externalSemaphoreHandleDesc.flags = 0;

    CUDA_CHECK(cudaImportExternalSemaphore(&mCudaExternalVkUpdateCudaSemaphore, &externalSemaphoreHandleDesc));
}
#endif

VkMemoryRequirements Image::getMemoryRequirements() const
{
    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(mApplicationContext->mDeviceRef, mImage, &memoryRequirements);

    return memoryRequirements;
}
