/**
 * @file      application.hpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#include <algorithm>
#include <app/layer.hpp>
#include <array>
#include <core/vulkanBuffer.hpp>
#include <core/vulkanComputePipeline.hpp>
#include <core/vulkanContext.hpp>
#include <core/vulkanImage.hpp>
#include <core/vulkanSwapchain.hpp>
#include <functional>
#include <imgui.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <volk.h>

#ifdef SAF_CUDA_INTEROP
#include <cuda.h>
#include <cuda_runtime.h>
#endif

struct GLFWwindow;

namespace saf
{
    /**
     * @brief Settings when creating any Application.
     */
    struct ApplicationSettings
    {
        /** @brief The name of the Application. */
        Str name;
        /** @brief The width of the Application window. Has no default. */
        I32 windowWidth;
        /** @brief The height of the Application window. Has no default. */
        I32 windowHeight;
        /** @brief The font scale of the Application user interface. */
        F32 fontScale = 1.0f;
        /** @brief Theme of the Application user interface: 0 is dark, 1 is light. */
        U32 theme = 0;
        /** @brief The clear color of the Application window. Has no default. */
        Vec4 clearColor;
        /** @brief Whether V-Sync is enabled. */
        bool vSyncEnabled = true;
        /** @brief The maximum number of images the Application can hold at once. Not persisted. */
        U32 maxImages = 64;
#ifdef SAF_VULKAN_COMPUTE
        /** @brief The maximum number of compute descriptor sets per frame, one per set of each dispatch. Each descriptor type gets four descriptors per set. Not persisted. */
        U32 maxComputeDescriptorSets = 256;
        /**
         * @brief Whether compute runs on a queue family without graphics when the device has one. Not persisted.
         * @details Off by default: on an RTX 5090 such a queue refills freed workgroup slots late, so dispatches with uneven work per workgroup ran slower than on the graphics queue.
         */
        bool asyncCompute = false;
#endif
#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
        /** @brief The maximum number of timing pairs (beginTiming calls) per frame, for Vulkan and CUDA each. At least 1. Not persisted. */
        U32 maxTimingPairs = 32;
#endif
    };

    /**
     * @brief Structure for creating Vulkan images.
     */
    struct VulkanImageCreateInfo
    {
        /** @brief The width of the image. */
        U32 width;
        /** @brief The height of the image. */
        U32 height;
        /** @brief The format of the image. */
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        /** @brief The usage flags for the image. */
        VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        /** @brief What else accesses the image besides ImGui: CUDA or Vulkan compute, never both (for now since not required). */
        ImageSharing sharing = ImageSharing::None;
    };

#ifdef SAF_VULKAN_COMPUTE
    /**
     * @brief Structure for creating Vulkan buffers.
     */
    struct VulkanBufferCreateInfo
    {
        /** @brief The size of the buffer in bytes. */
        VkDeviceSize size;
    };

    /**
     * @brief Structure for creating compute pipelines.
     */
    struct ComputePipelineCreateInfo
    {
        /** @brief The SPIR-V words of the shader module. Only read during creation. */
        std::span<const U32> spirv;
        /** @brief The name of the compute entry point. */
        const char* entryPoint = "main";
        /** @brief Optional specialization constants, passed through to the pipeline. */
        const VkSpecializationInfo* specialization = nullptr;
    };

    /**
     * @brief Names a binding of a compute shader, by its reflected name or by (set, binding): `"outImage"` or `{ 0, 1 }`.
     */
    struct BindingKey
    {
        /**
         * @brief Names a binding by its reflected name.
         * @param[in] name The reflected name.
         */
        BindingKey(const char* name)
            : name(name)
        {
        }
        /**
         * @brief Names a binding by its set and binding number.
         * @param[in] set The descriptor set number.
         * @param[in] binding The binding number in the set.
         */
        BindingKey(U32 set, U32 binding)
            : set(set)
            , binding(binding)
        {
        }

        /** @brief The reflected name, or nullptr when the binding is named by (set, binding). */
        const char* name = nullptr;
        /** @brief The descriptor set number. */
        U32 set = 0;
        /** @brief The binding number in the set. */
        U32 binding = 0;
    };

    /**
     * @brief Reads a SPIR-V file.
     * @param[in] path The path of the .spv file.
     * @param[out] outWords The SPIR-V words.
     * @return VkResult VK_ERROR_INITIALIZATION_FAILED if the file cannot be read or its size is not a multiple of 4, else VK_SUCCESS.
     */
    VkResult loadSpirv(const Str& path, std::vector<U32>& outWords);

    /**
     * @brief The path where saf_add_shaders puts the SPIR-V of a shader: shaders/ next to the running executable.
     * @details Asks the operating system for the executable's location, so it holds whatever the working directory and however the program was started.
     * @param[in] fileName The SPIR-V file name, the shader source's file name plus .spv, e.g. "template_slang.slang.spv".
     * @return Str The absolute path of the file.
     */
    Str shaderPath(const Str& fileName);
#endif

    /**
     * @brief Application.
     */
    class Application
    {
    public:
        /**
         * @brief Constructs an Application.
         * @param[in] settings Settings to setup the application.
         * @param[in] loadFromPersistedSettings Whether to load settings from a persisted source, e.g. an ini file. If true, every persisted field (name, window size, font scale, theme, clear color, V-Sync) overrides the provided settings, which then only act as first-run defaults. `maxImages`, `maxComputeDescriptorSets`, `asyncCompute` and `maxTimingPairs` are not persisted.
         */
        Application(const ApplicationSettings& settings, bool loadFromPersistedSettings = true);

        ~Application();

        /**
         * @brief Runs the Application.
         * @return bool True if the application ran successfully, else False.
         */
        bool run();

        /**
         * @brief Closes the Application.
         */
        void close();

        /**
         * @brief Pushes a Layer onto the layer stack held by the Application.
         * @tparam T The layer's type, derived from Layer.
         */
        template <typename T>
        void pushLayer()
        {
            static_assert(std::is_base_of<Layer, T>::value, "Pushed type is not a Layer!");
            pushLayer(std::make_unique<T>());
        }

        /**
         * @brief Pops a Layer off the layer stack held by the Application.
         * @details Called from a layer callback, the change is applied at the start of the next frame. Does nothing on an empty stack.
         */
        void popLayer();

        /**
         * @brief Sets a callback function creating the menu bar of the Application.
         * @param[in] callback A callback function creating the menu bar of the Application.
         */
        inline void setMenubarCallback(std::function<void()> callback)
        {
            mMenubarCallback = callback;
        }

        /**
         * @brief Creates a new Vulkan image.
         * @param[in] createInfo The creation info for the image.
         * @param[out] outImage The handle to the created image.
         * @return VkResult The result of the operation.
         */
        VkResult createImage(const VulkanImageCreateInfo& createInfo, ImageHandle& outImage);

        /**
         * @brief Destroys a Vulkan image.
         * @details Waits until the device is idle. Does nothing for an unknown handle. Reports and aborts on an image shared with compute after getComputeCommandBuffer() in the same Layer::onUpdate.
         * @param[in] image The handle of the image to destroy.
         */
        void destroyImage(ImageHandle image);

        /**
         * @brief Uploads data to a Vulkan image.
         * @param[in] image The handle of the image to upload data to.
         * @param[in] data The data to upload.
         * @param[in] size The size of the data to upload. Has to match the image size, width × height × bytes per pixel.
         * @return VkResult VK_ERROR_UNKNOWN if the handle is unknown or @a size does not match, else the result of the operation.
         */
        VkResult uploadImage(ImageHandle image, const void* data, size_t size);

        /**
         * @brief Downloads data from a Vulkan image.
         * @param[in] image The handle of the image to download data from.
         * @param[out] data The buffer to store the downloaded data. Has to be allocated by the caller.
         * @param[in] size The size of @a data. Has to match the image size, width × height × bytes per pixel.
         * @return VkResult VK_ERROR_UNKNOWN if the handle is unknown or @a size does not match, else the result of the operation.
         */
        VkResult downloadImage(ImageHandle image, void* data, size_t size);

        /**
         * @brief Gets the ImGui texture of an image, for ImGui::Image and the like.
         * @param[in] image The handle of the image.
         * @return ImTextureID The texture ID ImGui samples the image through.
         * @throws std::runtime_error If the handle is unknown.
         */
        ImTextureID getTextureId(ImageHandle image) const
        {
            return reinterpret_cast<ImTextureID>(getImage(image).descriptorSet);
        }

#ifdef SAF_CUDA_INTEROP
        /**
         * @brief Gets the CUDA surface object of an image, for writing it from a kernel.
         * @details Reports and aborts if the image does not exist or was not created with ImageSharing::Cuda.
         * @param[in] image The handle of the image.
         * @return cudaSurfaceObject_t The surface object, valid until the image is destroyed.
         */
        cudaSurfaceObject_t getCudaSurface(ImageHandle image) const;

        /**
         * @brief Gets the CUDA texture object of an image, for sampling it in a kernel.
         * @details Samples with normalized coordinates, point filtering, wrap addressing and element-type reads. Reports and aborts if the image does not exist or was not created with ImageSharing::Cuda.
         * @param[in] image The handle of the image.
         * @return cudaTextureObject_t The texture object, valid until the image is destroyed.
         */
        cudaTextureObject_t getCudaTexture(ImageHandle image) const;
#endif

#ifdef SAF_VULKAN_COMPUTE
        /**
         * @brief Creates a new Vulkan buffer, shared with compute.
         * @param[in] createInfo The creation info for the buffer.
         * @param[out] outBuffer The handle to the created buffer.
         * @return VkResult VK_ERROR_UNKNOWN if the size is 0, else the result of the operation.
         */
        VkResult createBuffer(const VulkanBufferCreateInfo& createInfo, BufferHandle& outBuffer);

        /**
         * @brief Destroys a Vulkan buffer.
         * @details Waits until the device is idle. Does nothing for an unknown handle. Reports and aborts after getComputeCommandBuffer() in the same Layer::onUpdate.
         * @param[in] buffer The handle of the buffer to destroy.
         */
        void destroyBuffer(BufferHandle buffer);

        /**
         * @brief Uploads data to a Vulkan buffer.
         * @param[in] buffer The handle of the buffer to upload data to.
         * @param[in] data The data to upload.
         * @param[in] size The size of the data to upload. Has to match the buffer size.
         * @return VkResult VK_ERROR_UNKNOWN if the handle is unknown or @a size does not match, else the result of the operation.
         */
        VkResult uploadBuffer(BufferHandle buffer, const void* data, size_t size);

        /**
         * @brief Downloads data from a Vulkan buffer.
         * @param[in] buffer The handle of the buffer to download data from.
         * @param[out] data The buffer to store the downloaded data. Has to be allocated by the caller.
         * @param[in] size The size of @a data. Has to match the buffer size.
         * @return VkResult VK_ERROR_UNKNOWN if the handle is unknown or @a size does not match, else the result of the operation.
         */
        VkResult downloadBuffer(BufferHandle buffer, void* data, size_t size);

        /**
         * @brief Gets the compute command buffer of the current frame slot.
         * @details Valid only during Layer::onUpdate. Misuse reports and aborts. The first call in a frame waits until the slot's previous submission has finished, reads the slot's timestamps, resets the slot's descriptor pool, resets and begins the command buffer and records the reset of the slot's timestamp queries. SAF submits it right after onUpdate.
         * @return VkCommandBuffer The compute command buffer, recording.
         */
        VkCommandBuffer getComputeCommandBuffer();

        /**
         * @brief Creates a compute pipeline, with descriptor set layouts and push constant range reflected from the shader.
         * @param[in] createInfo The creation info for the pipeline.
         * @param[out] outPipeline The handle to the created pipeline.
         * @return VkResult VK_ERROR_INITIALIZATION_FAILED if reflection fails or the entry point is missing or not a compute shader, VK_ERROR_FEATURE_NOT_PRESENT if the shader uses an unsupported descriptor type, an array of descriptors, or a capability whose device feature is not enabled (Int64, Int16, Float64), else the result of the operation.
         */
        VkResult createComputePipeline(const ComputePipelineCreateInfo& createInfo, ComputePipelineHandle& outPipeline);

        /**
         * @brief Destroys a compute pipeline.
         * @param[in] pipeline The handle of the pipeline to destroy.
         */
        void destroyComputePipeline(ComputePipelineHandle pipeline);

        /**
         * @brief Binds a shared-with-compute image to a binding, until rebound.
         * @details Misuse reports and aborts.
         * @param[in] pipeline The pipeline whose binding state is changed.
         * @param[in] key The binding, by reflected name or (set, binding); bindings without a debug name only by (set, binding).
         * @param[in] image The image, created with ImageSharing::Compute.
         */
        void bind(ComputePipelineHandle pipeline, const BindingKey& key, ImageHandle image);

        /**
         * @brief Binds a buffer to a binding, until rebound.
         * @details Misuse reports and aborts.
         * @param[in] pipeline The pipeline whose binding state is changed.
         * @param[in] key The binding, by reflected name or (set, binding); bindings without a debug name only by (set, binding).
         * @param[in] buffer The buffer.
         */
        void bind(ComputePipelineHandle pipeline, const BindingKey& key, BufferHandle buffer);

        /**
         * @brief Stores the push constants of a pipeline, until set again.
         * @details Reports and aborts if sizeof(T) differs from the reflected block size.
         * @tparam T The push constant struct, laid out as the shader's block.
         * @param[in] pipeline The pipeline whose push constants are set.
         * @param[in] data The push constant data.
         */
        template <typename T>
        void setPushConstants(ComputePipelineHandle pipeline, const T& data)
        {
            setPushConstantData(pipeline, &data, static_cast<U32>(sizeof(T)));
        }

        /**
         * @brief Records a dispatch of @a pipeline with its current bindings and push constants.
         * @details Valid only during Layer::onUpdate, since it records into getComputeCommandBuffer(). Misuse reports and aborts. Allocates fresh descriptor sets from the current frame slot, so rebinding between two dispatches in one frame behaves as written. Call computeBarrier() between dependent dispatches; SAF orders work across frames.
         * @param[in] pipeline The pipeline to dispatch.
         * @param[in] groupsX The number of workgroups in x.
         * @param[in] groupsY The number of workgroups in y.
         * @param[in] groupsZ The number of workgroups in z.
         */
        void dispatch(ComputePipelineHandle pipeline, U32 groupsX, U32 groupsY, U32 groupsZ);

        /**
         * @brief Makes the writes of all previously recorded dispatches visible to later dispatches in this frame.
         * @details Valid only during Layer::onUpdate. Misuse reports and aborts. Records a global memory barrier, compute shader writes to compute shader reads and writes, into getComputeCommandBuffer(). Covers buffers and shared-with-compute images alike. Work in earlier frames is already ordered by SAF.
         */
        void computeBarrier();

        /**
         * @brief Begins the timing scope @a name with a timestamp after all commands previously recorded into getComputeCommandBuffer().
         * @details Valid only during Layer::onUpdate. Misuse reports and aborts. Several pairs of one name in a frame are summed. Does nothing but the misuse checks if the compute queue writes no timestamps.
         * @param[in] name The name of the scope. Names are unique across Vulkan and CUDA, so one already used by a CUDA scope reports and aborts.
         */
        void beginTiming(const char* name);

        /**
         * @brief Ends the open timing scope @a name with a timestamp after all commands previously recorded into getComputeCommandBuffer().
         * @details Valid only during Layer::onUpdate. Misuse reports and aborts.
         * @param[in] name The name of the scope.
         */
        void endTiming(const char* name);
#endif

#ifdef SAF_CUDA_INTEROP
        /**
         * @brief Begins the timing scope @a name with an event recorded on @a stream.
         * @details Valid only during Layer::onUpdate. Misuse reports and aborts. Several pairs of one name in a frame are summed.
         * @param[in] stream The CUDA stream.
         * @param[in] name The name of the scope. Names are unique across Vulkan and CUDA, so one already used by a Vulkan scope reports and aborts.
         */
        void beginTiming(cudaStream_t stream, const char* name);

        /**
         * @brief Ends the open timing scope @a name with an event recorded on @a stream.
         * @details Valid only during Layer::onUpdate. Misuse reports and aborts.
         * @param[in] stream The CUDA stream.
         * @param[in] name The name of the scope.
         */
        void endTiming(cudaStream_t stream, const char* name);
#endif

#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
        /** @brief The number of finished durations each timing scope keeps. */
        static constexpr U32 timingHistoryLength = 128;

        /** @brief What a timing scope is timed on. */
        enum class TimingApi
        {
            Vulkan,
            Cuda
        };

        /** @brief A named timing scope. */
        struct TimingScope
        {
            /** @brief The name, unique across Vulkan and CUDA. */
            Str name;
            /** @brief Whether the scope is timed on the compute command buffer or CUDA streams. */
            TimingApi api;
            /** @brief The summed durations in milliseconds of the last finished frames that timed the scope, oldest first, at most timingHistoryLength; read about a frame after the work finished. */
            std::vector<F32> history;
        };

        /**
         * @brief Gets all timing scopes.
         * @return std::span<const TimingScope> The scopes, in order of first use. Valid until the next frame.
         */
        std::span<const TimingScope> getTimings() const { return mTimingScopes; }
#endif

    private:
        /**
         * @brief Initializes Vulkan and GLFW.
         * @return bool True if initialization was successful, else False.
         */
        bool init();

        /**
         * @brief Destroys Vulkan and GLFW resources.
         */
        void destroy();

        /**
         * @brief Pushes a layer onto the layer stack.
         * @details The internal implementation handles synchronization. Called from a layer callback, the push is applied at the start of the next frame.
         *
         * @param[in] layer The layer to push onto the stack.
         */
        void pushLayer(std::unique_ptr<Layer>&& layer);

        /**
         * @brief Executes a single-time command buffer.
         *
         * @param[in] immediateFunction The function to execute within the command buffer.
         * @return VkResult The result of the operation.
         */
        VkResult executeSingleTimeCommandBuffer(const std::function<VkResult(VkCommandBuffer)>& immediateFunction);

#ifdef SAF_VULKAN_COMPUTE
        /**
         * @brief Whether compute runs asynchronously to graphics.
         *
         * @return True if ApplicationSettings::asyncCompute is set and the compute queue's family differs from the graphics family.
         */
        inline bool hasAsyncCompute() const { return mAsyncComputeRequested && mQueueC.familyIndex != mQueueGCT.familyIndex; }
#endif

        /**
         * @brief Finds a suitable memory type.
         * @param[in] memoryTypeBits The memory type bits to search for.
         * @param[in] propertyFlags The property flags to match.
         * @return U32 The type of the found memory, or an error if not found.
         */
        U32 findMemoryType(U32 memoryTypeBits, VkMemoryPropertyFlags propertyFlags);

        /**
         * @brief Creates a Vulkan buffer and binds newly allocated memory to it. On failure, the out handles hold whatever was created so far.
         * @param[in] bufferCreateInfo The buffer creation info.
         * @param[in] propertyFlags The property flags the memory has to have.
         * @param[out] buffer The buffer, to be destroyed by the caller.
         * @param[out] bufferMemory The buffer memory, to be freed by the caller.
         * @return VkResult The result of the first failing call, or VK_SUCCESS.
         */
        VkResult allocateBuffer(const VkBufferCreateInfo& bufferCreateInfo, VkMemoryPropertyFlags propertyFlags, VkBuffer& buffer, VkDeviceMemory& bufferMemory);

        /**
         * @brief Gets the Image object for a given image handle.
         * @param[in] image The handle of the image to retrieve.
         * @return const VulkanImage& The requested image object.
         * @throws std::runtime_error If the handle is unknown.
         */
        const VulkanImage& getImage(ImageHandle image) const
        {
            const VulkanImage* found = mImages.find(image);
            if (found == nullptr)
            {
                throw std::runtime_error("Image not found");
            }
            return *found;
        }

        /**
         * @brief Destroys the resources associated with a Vulkan image.
         * @param[in] image The Vulkan image for which to destroy resources.
         */
        void destroyImageResources(const VulkanImage& image);

        /**
         * @brief Creates the Vulkan (and CUDA) resources of an image. On failure, @a image holds whatever was created so far.
         * @param[in,out] image The image to fill in.
         * @param[in] createInfo The image creation info.
         * @return VkResult The result of the first failing call, or VK_SUCCESS.
         */
        VkResult createImageResources(VulkanImage& image, const VulkanImageCreateInfo& createInfo);

        /**
         * @brief Copies @a data into a staging buffer, then runs @a recordCopy in a single-time command buffer.
         * @param[in] data The data to upload.
         * @param[in] size The size of @a data in bytes.
         * @param[in] recordCopy Records the copy from the staging buffer into the resource, with the resource's barriers.
         * @return VkResult The result of the first failing call, or VK_SUCCESS.
         */
        VkResult uploadThroughStagingBuffer(const void* data, size_t size, const std::function<void(VkCommandBuffer, VkBuffer)>& recordCopy);

        /**
         * @brief Runs @a recordCopy in a single-time command buffer, then copies the staging buffer into @a data.
         * @param[out] data The destination of the downloaded data.
         * @param[in] size The size of @a data in bytes.
         * @param[in] recordCopy Records the copy from the resource into the staging buffer, with the resource's barriers.
         * @return VkResult The result of the first failing call, or VK_SUCCESS.
         */
        VkResult downloadThroughStagingBuffer(void* data, size_t size, const std::function<void(VkCommandBuffer, VkBuffer)>& recordCopy);

        /**
         * @brief Waits on the host until the CUDA or compute work submitted so far on a shared image has finished; returns at once for other images.
         * @param[in] image The image.
         * @return VkResult The result of the operation.
         */
        VkResult waitForSharedImageWork(const VulkanImage& image);

#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
        /**
         * @brief Waits on the host until @a timeline reaches @a value.
         * @param[in] timeline The timeline semaphore.
         * @param[in] value The timeline value to wait for.
         * @return VkResult The result of the operation.
         */
        VkResult waitForTimeline(VkSemaphore timeline, U64 value);
#endif

#ifdef SAF_VULKAN_COMPUTE
        /**
         * @brief Destroys the resources associated with a Vulkan buffer.
         * @param[in] buffer The Vulkan buffer for which to destroy resources.
         */
        void destroyBufferResources(const VulkanBuffer& buffer);

        /**
         * @brief Reflects the shader and creates the Vulkan objects of a compute pipeline. On failure, @a pipeline holds whatever was created so far.
         * @param[in,out] pipeline The pipeline to fill in.
         * @param[in] createInfo The pipeline creation info.
         * @return VkResult The result of the first failing call, or VK_SUCCESS.
         */
        VkResult createComputePipelineResources(VulkanComputePipeline& pipeline, const ComputePipelineCreateInfo& createInfo);

        /**
         * @brief Destroys the Vulkan objects of a compute pipeline.
         * @param[in] pipeline The pipeline for which to destroy resources.
         */
        void destroyComputePipelineResources(const VulkanComputePipeline& pipeline);

        /**
         * @brief Finds a compute pipeline, reports and aborts if it does not exist.
         * @param[in] pipeline The handle of the pipeline.
         * @return VulkanComputePipeline& The pipeline.
         */
        VulkanComputePipeline& findComputePipeline(ComputePipelineHandle pipeline);

        /**
         * @brief Finds the binding @a key names, reports and aborts if the shader does not have it or a name is ambiguous across sets.
         * @param[in] pipeline The pipeline to search.
         * @param[in] key The binding, by reflected name or (set, binding).
         * @return ComputeBinding& The binding.
         */
        ComputeBinding& findComputeBinding(VulkanComputePipeline& pipeline, const BindingKey& key);

        /**
         * @brief Finds a shared-with-compute image, reports and aborts if it does not exist or is not shared with compute.
         * @param[in] image The handle of the image.
         * @return const VulkanImage& The image.
         */
        const VulkanImage& findSharedWithComputeImage(ImageHandle image) const;

        /**
         * @brief Finds a buffer, reports and aborts if it does not exist.
         * @param[in] buffer The handle of the buffer.
         * @return const VulkanBuffer& The buffer.
         */
        const VulkanBuffer& findComputeBuffer(BufferHandle buffer) const;

        /**
         * @brief Stores @a image in an image binding, reports and aborts on a buffer binding or an invalid image.
         * @param[in,out] binding The binding.
         * @param[in] image The image.
         */
        void bindImage(ComputeBinding& binding, ImageHandle image);

        /**
         * @brief Stores @a buffer in a buffer binding, reports and aborts on an image binding or an invalid buffer.
         * @param[in,out] binding The binding.
         * @param[in] buffer The buffer.
         */
        void bindBuffer(ComputeBinding& binding, BufferHandle buffer);

        /**
         * @brief Stores push constant bytes, reports and aborts if @a size differs from the reflected block size.
         * @param[in] pipeline The pipeline whose push constants are set.
         * @param[in] data The push constant data.
         * @param[in] size The size of @a data in bytes.
         */
        void setPushConstantData(ComputePipelineHandle pipeline, const void* data, U32 size);
#endif

#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
        /** @brief One begin/end pair of a timing scope in a frame slot; its queries or events are at 2 × its index and the one after. */
        struct TimingPair
        {
            /** @brief The index of the scope in mTimingScopes. */
            U32 scope;
            /** @brief Whether the end is still missing. */
            bool open;
        };

        /**
         * @brief Opens a pair of the scope @a name, creating the scope on first use. Reports and aborts over the cap, on an open pair of the same name, or on a name used from the other API.
         * @param[in,out] pairs The pairs of the current frame slot.
         * @param[in] name The name of the scope.
         * @param[in] api What the pair is timed on.
         * @return U32 The index of the pair.
         */
        U32 beginTimingPair(std::vector<TimingPair>& pairs, const char* name, TimingApi api);

        /**
         * @brief Closes the open pair of the scope @a name, reports and aborts if there is none.
         * @param[in,out] pairs The pairs of the current frame slot.
         * @param[in] name The name of the scope.
         * @return U32 The index of the pair.
         */
        U32 endTimingPair(std::vector<TimingPair>& pairs, const char* name);

        /**
         * @brief Reports and aborts if a pair is still open at the end of Layer::onUpdate.
         * @param[in] pairs The pairs of the current frame slot.
         */
        void checkTimingPairsClosed(const std::vector<TimingPair>& pairs);

        /**
         * @brief Stores the durations of a frame slot's pairs, summed per scope, appends each sum to its scope's history and clears the pairs.
         * @param[in,out] pairs The pairs of a frame slot.
         * @param[in] milliseconds The duration of each pair.
         */
        void storeTimings(std::vector<TimingPair>& pairs, std::span<const F32> milliseconds);

        /**
         * @brief Reads the timings of every frame slot whose work has finished. Called once per loop iteration, before Layer::onUpdate.
         */
        void collectTimings();
#endif

#ifdef SAF_VULKAN_COMPUTE
        /**
         * @brief Reads the timestamps of a compute frame slot whose submission has finished, then clears its pairs.
         * @param[in] slotIndex The index of the slot.
         */
        void readComputeTimings(U32 slotIndex);
#endif

#ifdef SAF_CUDA_INTEROP
        /**
         * @brief Reads the events of a CUDA timing frame slot once all have completed, then clears its pairs.
         * @param[in] slotIndex The index of the slot.
         * @param[in] wait Whether to wait for the events, else the slot is left as is while one is pending.
         */
        void readCudaTimings(U32 slotIndex, bool wait);
#endif

        /**
         * @brief Gets the number of bytes per pixel for a given Vulkan format.
         * @param[in] format The Vulkan format.
         * @return U32 The number of bytes per pixel.
         */
        U32 getFormatBytesPerPixel(VkFormat format);

        /**
         * @brief Transitions the layout of a Vulkan image.
         * @param[in] image The Vulkan image to transition.
         * @param[in] oldLayout The old layout of the image.
         * @param[in] newLayout The new layout of the image.
         * @param[in] srcAccessMask The source access mask for the transition.
         * @param[in] dstAccessMask The destination access mask for the transition.
         * @param[in] srcStage The source pipeline stage for the transition.
         * @param[in] dstStage The destination pipeline stage for the transition.
         * @param[in] commandBuffer The command buffer to record the transition commands into (optional, if not provided a single-time command buffer will be used).
         * @return VkResult The result of the operation.
         */
        VkResult transitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkCommandBuffer commandBuffer = VK_NULL_HANDLE);

#ifdef SAF_CUDA_INTEROP

        /** @brief Gets the handle for a Vulkan semaphore.
         * @param[in] semaphore The Vulkan semaphore.
         * @return void* The retrieved external semaphore handle.
         */
        void* getSemaphoreHandle(VkSemaphore semaphore);

        /** @brief Gets the handle for a Vulkan device memory object.
         * @param[in] deviceMemory The Vulkan device memory object.
         * @return void* The retrieved external memory handle.
         */
        void* getMemoryHandle(VkDeviceMemory deviceMemory);

        /** @brief Gets the CUDA channel format descriptor for a given Vulkan format.
         * @param[in] format The Vulkan format.
         * @return cudaChannelFormatDesc The corresponding CUDA channel format descriptor.
         */
        cudaChannelFormatDesc getFormatDescriptor(VkFormat format);

        /** @brief Waits for Vulkan completion.
         * @param[in] semaphore The cuda external timeline semaphore to wait on.
         * @param[in] value The timeline value to wait for.
         * @param[in] stream The CUDA stream to use.
         */
        void waitForVulkanCompletion(cudaExternalSemaphore_t semaphore, U64 value, cudaStream_t stream = 0);

        /** @brief Signals CUDA completion.
         * @param[in] semaphore The cuda external timeline semaphore to signal.
         * @param[in] value The timeline value to signal.
         * @param[in] stream The CUDA stream to use.
         */
        void signalCudaCompletion(cudaExternalSemaphore_t semaphore, U64 value, cudaStream_t stream = 0);

        /** @brief Whether any image is shared with CUDA; the CUDA handshake runs only then. */
        bool hasCudaSharedImage() const;
#endif

    private:
        /** @brief The Application's name. */
        Str mName;
        /** @brief The Application's window width. */
        I32 mWindowWidth;
        /** @brief The Application's window height. */
        I32 mWindowHeight;
        /** @brief The Application's font scale */
        F32 mFontScale;
        /** @brief The Application's theme. */
        U32 mTheme;
        /** @brief The Application's clear color. */
        Vec4 mClearColor;
        /** @brief Whether V-Sync is enabled. */
        bool mVSyncEnabled;

        /** @brief The maximum number of images, sizes the ImGui descriptor pool. */
        U32 mMaxImages;
        /** @brief True, if the Application is running, else False. */
        bool mRunning = false;
        /** @brief True, if init() completed successfully, else False. */
        bool mInitialized = false;

        /** @brief The Application's menubar callback. */
        std::function<void()> mMenubarCallback;

        /** @brief The Application's window handle. */
        GLFWwindow* mWindow = nullptr;
        /** @brief The Application's vulkan context. */
        std::unique_ptr<VulkanContext> mVulkanContext;
        /** @brief The Application's vulkan swapchain. */
        std::unique_ptr<VulkanSwapchain> mVulkanSwapchain;

        /** @brief The Application's window surface. */
        VkSurfaceKHR mSurface = VK_NULL_HANDLE;

        /** @brief The Application's layer stack. */
        std::vector<std::unique_ptr<Layer>> mLayerStack;

        /** @brief Which layer loop is running. */
        enum class LayerPhase
        {
            None,
            Update,
            UIRender
        };

        /** @brief The running layer loop; layer stack changes are deferred during any, the compute command buffer and CUDA timings are valid only during Update. */
        LayerPhase mLayerPhase = LayerPhase::None;

        /** @brief Deferred layer stack changes applied at the start of the next frame, a nullptr pops. */
        std::vector<std::unique_ptr<Layer>> mPendingLayerOps;

        /** @brief The Application's Vulkan instance. */
        VkInstance mInstance = VK_NULL_HANDLE;
        /** @brief The Application's physical device. */
        VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
        /** @brief The Application's logical device. */
        VkDevice mLogicalDevice = VK_NULL_HANDLE;

        /** @brief The Application's graphics queue. */
        VulkanQueue mQueueGCT;
        /** @brief The Application's transfer queue. */
        VulkanQueue mQueueT;
        /** @brief The Application's compute queue. */
        VulkanQueue mQueueC;

        /** @brief The Application's command pool. */
        VkCommandPool mCommandPool = VK_NULL_HANDLE;
        /** @brief The Application's transient command pool. */
        VkCommandPool mTransientCommandPool = VK_NULL_HANDLE;
        /** @brief The Application's command buffers. */
        std::vector<VkCommandBuffer> mCommandBuffers;

        /** @brief Resources of one kind by handle, handing out ids from 1 in order. */
        template <typename H, typename T>
        class HandleRegistry
        {
        public:
            H add(const T& resource)
            {
                const H handle{ mNextId++ };
                mEntries.push_back({ handle, resource });
                return handle;
            }

            /** @brief The resource of @a handle, or nullptr if there is none. */
            T* find(H handle)
            {
                const auto it = std::ranges::find(mEntries, handle, &std::pair<H, T>::first);
                return it == mEntries.end() ? nullptr : &it->second;
            }

            const T* find(H handle) const { return const_cast<HandleRegistry*>(this)->find(handle); }

            void erase(H handle)
            {
                std::erase_if(mEntries, [handle](const auto& entry)
                              { return entry.first == handle; });
            }

            auto begin() const { return mEntries.begin(); }
            auto end() const { return mEntries.end(); }
            void clear() { mEntries.clear(); }

        private:
            // No unordered_map: there are few resources, and hashing would cost more than the search
            std::vector<std::pair<H, T>> mEntries;
            U32 mNextId = 1;
        };

        /** @brief The Application's registered images. */
        HandleRegistry<ImageHandle, VulkanImage> mImages;

#ifdef SAF_VULKAN_COMPUTE
        /** @brief The Application's registered buffers. */
        HandleRegistry<BufferHandle, VulkanBuffer> mBuffers;

        /** @brief The compute resources of one frame slot. */
        struct ComputeSlot
        {
            /** @brief The slot's compute command buffer. */
            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
            /** @brief The compute timeline value last signalled from this slot. */
            U64 timelineValue = 0;
            /** @brief The slot's descriptor pool, reset when the slot is reused. */
            VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
            /** @brief The slot's timestamp queries, two per timing pair, reset when the slot is reused. */
            VkQueryPool queryPool = VK_NULL_HANDLE;
            /** @brief The timing pairs written in the slot's last use, cleared once read. */
            std::vector<TimingPair> timingPairs;
        };

        /** @brief The Application's registered compute pipelines. */
        HandleRegistry<ComputePipelineHandle, VulkanComputePipeline> mComputePipelines;

        /** @brief The device's maxComputeWorkGroupCount, checked by dispatch. */
        std::array<U32, 3> mMaxComputeWorkGroupCount{};

        /** @brief The maximum number of compute descriptor sets per frame slot. */
        U32 mMaxComputeDescriptorSets;
        /** @brief ApplicationSettings::asyncCompute. */
        bool mAsyncComputeRequested;
        /** @brief The compute command pool, on the compute queue's family. */
        VkCommandPool mComputeCommandPool = VK_NULL_HANDLE;
        /** @brief One entry per frame slot. */
        std::vector<ComputeSlot> mComputeSlots;
        /** @brief Whether the current slot's command buffer is recording, i.e. it was requested this frame. */
        bool mComputeRecording = false;
        /** @brief The mask of the valid timestamp bits of the compute queue, 0 if it writes no timestamps. */
        U64 mTimestampMask = 0;
        /** @brief The nanoseconds per timestamp tick. */
        F32 mTimestampPeriod = 0.0f;

        /** @brief Timeline semaphore signalled only by compute submits. */
        VkSemaphore mComputeTimeline = VK_NULL_HANDLE;
        /** @brief The last value submitted to signal the compute timeline. */
        U64 mComputeTimelineValue = 0;
#endif

#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
        /** @brief The maximum number of timing pairs per frame slot, for Vulkan and CUDA each. */
        U32 mMaxTimingPairs;
        /** @brief The timing scopes, in order of first use. */
        std::vector<TimingScope> mTimingScopes;
        /** @brief The current frame slot of compute and timing, advanced once per loop iteration. */
        U32 mFrameSlotIndex = 0;

        /** @brief Timeline semaphore signalled only by the frame submit; compute submits and CUDA wait on it. */
        VkSemaphore mGraphicsTimeline = VK_NULL_HANDLE;
        /** @brief The last value submitted to signal the graphics timeline. */
        U64 mGraphicsTimelineValue = 0;
#endif

#ifdef SAF_CUDA_INTEROP
        /** @brief Timeline semaphore signalled only by CUDA, after onUpdate; the frame submit waits on it. */
        VkSemaphore mCudaTimeline = VK_NULL_HANDLE;
        /** @brief The last value CUDA was asked to signal on the CUDA timeline. */
        U64 mCudaTimelineValue = 0;
        /** @brief mGraphicsTimeline imported into CUDA. */
        cudaExternalSemaphore_t mGraphicsTimelineCuda = {};
        /** @brief mCudaTimeline imported into CUDA. */
        cudaExternalSemaphore_t mCudaTimelineCuda = {};

        /** @brief The CUDA timing resources of one frame slot. */
        struct CudaTimingSlot
        {
            /** @brief The slot's events, two per timing pair. */
            std::vector<cudaEvent_t> events;
            /** @brief The timing pairs recorded in the slot's last use, cleared once read. */
            std::vector<TimingPair> timingPairs;
        };

        /** @brief One entry per frame slot. */
        std::vector<CudaTimingSlot> mCudaTimingSlots;
#endif
    };
} // namespace saf

#endif // APPLICATION_HPP
