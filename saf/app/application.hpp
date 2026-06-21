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

#include "layer.hpp"
#include <core/vulkanContext.hpp>
#include <core/vulkanImage.hpp>
#include <core/vulkanSwapchain.hpp>
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
     * @brief Settings when creating any @a Application.
     */
    struct ApplicationSettings
    {
        /** @brief The name of the @a Application. */
        Str name;
        /** @brief The width of the @a Application window. */
        I32 windowWidth;
        /** @brief The height of the @a Application window. */
        I32 windowHeight;
        /** @brief The font scale of the @a Application user interface. */
        F32 fontScale = 1.0f;
        /** @brief Theme of the @a Application user interface. */
        U32 theme = 0;
        /** @brief The clear color of the @a Application window. */
        Vec4 clearColor;
        /** @brief Whether V-Sync is enabled. */
        bool vSyncEnabled = true;
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

#ifdef SAF_CUDA_INTEROP
        /** @brief Whether the image should be shared with CUDA. */
        bool shareWithCUDA = false;
#endif
    };

    /**
     * @brief @a Application.
     */
    class Application
    {
    public:
        /**
         * @brief Constructs a @a Application.
         * @param[in] settings Settings to setup the application.
         * @param[in] loadFromPersistedSettings Whether to load settings from a persisted source, e.g. an ini file. If true, the provided settings will be overridden by the persisted settings.
         */
        Application(const ApplicationSettings& settings, bool loadFromPersistedSettings = true);

        ~Application();

        /**
         * @brief Run the @a Application.
         * @return True if the application ran successfully, else False.
         */
        bool run();

        /**
         * @brief Close the @a Application.
         */
        void close();

        /**
         * @brief Pushes a @a Layer to the layer stack hold by the @a Application.
         * @tparam T The layers type derived from the @a Layer type.
         */
        template <typename T>
        void pushLayer()
        {
            static_assert(std::is_base_of<Layer, T>::value, "Pushed type is not a Layer!");
            pushLayer(std::make_unique<T>());
        }

        /**
         * @brief Pops a @a Layer of the layer stack hold by the @a Application.
         */
        void popLayer();

        /**
         * @brief Sets a callback function creating the menu bar of the @a Application.
         * @param[in] callback A callback function creating the menu bar of the @a Application.
         */
        inline void setMenubarCallback(std::function<void()> callback)
        {
            mMenubarCallback = callback;
        }

        /**
         * @brief Sets the callback function for handling window resize events.
         * @param[in] callback A callback function for handling window resize events.
         */
        inline void setOnResizeCallback(std::function<void(I32, I32)> callback)
        {
            mOnResizeCallback = callback;
        }

        /**
         * @brief Get the Instance object
         *
         * @return VkInstance
         */
        inline VkInstance getInstance() const { return mInstance; }

        /**
         * @brief Get the Physical Device object
         *
         * @return VkPhysicalDevice
         */
        inline VkPhysicalDevice getPhysicalDevice() const { return mPhysicalDevice; }

        /**
         * @brief Get the Logical Device object
         *
         * @return VkDevice
         */
        inline VkDevice getLogicalDevice() const { return mLogicalDevice; }

        /**
         * @brief Get a queue from the application context.
         *
         * @param index The index of the queue to retrieve.
         * @return VulkanQueue The requested queue.
         */
        inline VulkanQueue getQueue(U32 index) const
        {
            switch (index)
            {
            case 0:
                return mQueueGCT;
            case 1:
                return mQueueT;
            case 2:
                return mQueueC;
            default:
                std::cerr << "Error: Invalid queue index requested from ApplicationContext. Returning graphics queue." << std::endl;
                return mQueueGCT;
            }
        }

        /**
         * @brief Get the Transient Command Pool object
         *
         * @return VkCommandPool
         */
        inline VkCommandPool getTransientCommandPool() const { return mTransientCommandPool; }
        /**
         * @brief Get the Command Pool object
         *
         * @return VkCommandPool
         */
        inline VkCommandPool getCommandPool() const { return mCommandPool; }
        /**
         * @brief Get the Command Buffer object
         *
         * @param index The index of the command buffer to retrieve.
         * @return VkCommandBuffer The requested command buffer.
         */
        inline VkCommandBuffer getCommandBuffer(U32 index) const { return mCommandBuffers[index]; }

        /**
         * @brief Executes a single-time command buffer.
         *
         * @param immediateFunction The function to execute within the command buffer.
         * @return VkResult The result of the operation.
         */
        VkResult executeSingleTimeCommandBuffer(const std::function<VkResult(VkCommandBuffer)>& immediateFunction);

        /**
         * @brief Creates a new Vulkan image.
         * @param createInfo The creation info for the image.
         * @param outImage The handle to the created image.
         * @return VkResult The result of the operation.
         */
        VkResult createImage(const VulkanImageCreateInfo& createInfo, ImageHandle& outImage);

        /**
         * @brief Destroys a Vulkan image.
         * @param image The handle of the image to destroy.
         */
        void destroyImage(ImageHandle image);

        /**
         * @brief Uploads data to a Vulkan image.
         * @param image The handle of the image to upload data to.
         * @param data The data to upload.
         * @param size The size of the data to upload.
         * @return VkResult The result of the operation.
         */
        VkResult uploadImage(ImageHandle image, const void* data, size_t size);

        /**
         * @brief Downloads data from a Vulkan image.
         * @param image The handle of the image to download data from.
         * @param data The buffer to store the downloaded data. Has to be allocated by the caller.
         * @param size The size of the buffer.
         * @return VkResult The result of the operation.
         */
        VkResult downloadImage(ImageHandle image, void* data, size_t size);

        /**
         * @brief Get the Image object for a given image handle.
         * @param image The handle of the image to retrieve.
         * @return const VulkanImage& The requested image object.
         */
        const VulkanImage& getImage(ImageHandle image) const
        {
            for (const auto& pair : mImages)
            {
                if (pair.first == image)
                {
                    return pair.second;
                }
            }
            throw std::runtime_error("Image not found");
        }

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
         * @details The internal implementation handles synchronization.
         *
         * @param layer The layer to push onto the stack.
         */
        void pushLayer(std::unique_ptr<Layer>&& layer);

        /**
         * @brief Finds a suitable memory type.
         * @param memoryTypeBits The memory type bits to search for.
         * @param propertyFlags The property flags to match.
         * @return U32 The type of the found memory, or an error if not found.
         */
        U32 findMemoryType(U32 memoryTypeBits, VkMemoryPropertyFlags propertyFlags);

        /**
         * @brief Destroys the resources associated with a Vulkan image.
         * @param image The Vulkan image for which to destroy resources.
         */
        void destroyImageResources(const VulkanImage& image);

        /**
         * @brief Gets the number of bytes per pixel for a given Vulkan format.
         * @param format The Vulkan format.
         * @return U32 The number of bytes per pixel.
         */
        U32 getFormatBytesPerPixel(VkFormat format);

        /**
         * @brief Transitions the layout of a Vulkan image.
         * @param image The Vulkan image to transition.
         * @param oldLayout The old layout of the image.
         * @param newLayout The new layout of the image.
         * @param srcAccessMask The source access mask for the transition.
         * @param dstAccessMask The destination access mask for the transition.
         * @param srcStage The source pipeline stage for the transition.
         * @param dstStage The destination pipeline stage for the transition.
         * @param commandBuffer The command buffer to record the transition commands into (optional, if not provided a single-time command buffer will be used).
         * @return VkResult The result of the operation.
         */
        VkResult transitionImageLayout(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkCommandBuffer commandBuffer = VK_NULL_HANDLE);

#ifdef SAF_CUDA_INTEROP

        /** @brief Gets the handle for a Vulkan semaphore.
         * @param semaphore The Vulkan semaphore.
         * @return void* The retrieved external semaphore handle.
         */
        void* getSemaphoreHandle(VkSemaphore semaphore);

        /** @brief Gets the handle for a Vulkan device memory object.
         * @param deviceMemory The Vulkan device memory object.
         * @return void* The retrieved external memory handle.
         */
        void* getMemoryHandle(VkDeviceMemory deviceMemory);

        /** @brief Gets the CUDA channel format descriptor for a given Vulkan format.
         * @param format The Vulkan format.
         * @return cudaChannelFormatDesc The corresponding CUDA channel format descriptor.
         */
        cudaChannelFormatDesc getFormatDescriptor(VkFormat format);

        /** @brief Waits for Vulkan completion.
         * @param semaphore The cuda external semaphore to wait on.
         * @param stream The CUDA stream to use.
         */
        void waitForVulkanCompletion(cudaExternalSemaphore_t semaphore, cudaStream_t stream = 0);

        /** @brief Signals CUDA completion.
         * @param semaphore The cuda external semaphore to signal.
         * @param stream The CUDA stream to use.
         */
        void signalCudaCompletion(cudaExternalSemaphore_t semaphore, cudaStream_t stream = 0);
#endif

    private:
        /** @brief The @a Applications name. */
        Str mName;
        /** @brief The @a Applications window width. */
        I32 mWindowWidth;
        /** @brief The @a Applications window height. */
        I32 mWindowHeight;
        /** @brief The @a Applications font scale */
        F32 mFontScale;
        /** @brief The @a Applications theme. */
        U32 mTheme;
        /** @brief The @a Applications clear color. */
        Vec4 mClearColor;
        /** @brief Whether V-Sync is enabled. */
        bool mVSyncEnabled;
        /** @brief True, if the @a Application is running, else False. */
        bool mRunning;

        /** @brief The @a Applications menubar callback. */
        std::function<void()> mMenubarCallback;
        /** @brief The @a Applications on resize callback. */
        std::function<void(I32, I32)> mOnResizeCallback;

        /** @brief The @a Applications window handle. */
        GLFWwindow* mWindow;
        /** @brief The @a Applications vulkan context. */
        std::unique_ptr<VulkanContext> mVulkanContext;
        /** @brief The @a Applications vulkan swapchain. */
        std::unique_ptr<VulkanSwapchain> mVulkanSwapchain;

        /** @brief The @a Applications window surface. */
        VkSurfaceKHR mSurface;

        /** @brief The @a Applications layer stack. */
        std::vector<std::unique_ptr<Layer>> mLayerStack;

        /** @brief The @a Applications Vulkan instance. */
        VkInstance mInstance;
        /** @brief The @a Applications physical device. */
        VkPhysicalDevice mPhysicalDevice;
        /** @brief The @a Applications logical device. */
        VkDevice mLogicalDevice;

        /** @brief The @a Applications graphics queue. */
        VulkanQueue mQueueGCT;
        /** @brief The @a Applications transfer queue. */
        VulkanQueue mQueueT;
        /** @brief The @a Applications compute queue. */
        VulkanQueue mQueueC;

        /** @brief The @a Applications command pool. */
        VkCommandPool mCommandPool;
        /** @brief The @a Applications transient command pool. */
        VkCommandPool mTransientCommandPool;
        /** @brief The @a Applications command buffers. */
        std::vector<VkCommandBuffer> mCommandBuffers;

        /** @brief The @a Applications registered images. */
        std::vector<std::pair<ImageHandle, VulkanImage>> mImages; // NOTE: No unordered_map since we expect small number of images and want to avoid the overhead of hashing

        /** @brief The next image handle to be assigned. */
        U32 mNextImageHandle = 1; // Start from 1 to avoid using 0 as a valid handle // TODO: Consider using a more robust handle management system
    };
} // namespace saf

#endif // APPLICATION_HPP
