/**
 * @file      application.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2024
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#include "layer.hpp"
#include <vector>

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
        /** @brief The font size of the @a Application user interface. */
        F32 fontSize = 18.0f;
        /** @brief The clear color of the @a Application window. */
        Vec4 clearColor;
    };

    class ApplicationContext
    {
    public:
        ApplicationContext() = default;

        ~ApplicationContext() = default;

#ifdef SAF_CUDA_INTEROP
        struct ImageSemaphores
        {
            VkSemaphore vkUpdateCudaSemaphore;
            VkSemaphore cudaUpdateVkSemaphore;

            cudaExternalSemaphore_t cudaExternalVkUpdateCudaSemaphore;
            cudaExternalSemaphore_t cudaExternalCudaUpdateVkSemaphore;
        };
#endif
    private:
        friend class Application;
        friend class Image;

        VkInstance mInstanceRef;
        VkPhysicalDevice mPhysicalDeviceRef;
        VkDevice mDeviceRef;

        VkQueue mQueueRef;
        VkCommandPool mCommandPoolRef;
        VkCommandBuffer mCommandBufferRef;

#ifdef SAF_CUDA_INTEROP
        struct ContextSemaphores
        {
            std::unordered_map<VkImage, ImageSemaphores> imageSemaphores;
        } mContextSemaphores;

        void registerImage(VkImage image);
        void registerImage(VkImage image, const ImageSemaphores& imageSemaphores);
        void deregisterImage(VkImage image);

#ifdef WIN32
        inline void getMemoryWin32HandleKHR(const VkMemoryGetWin32HandleInfoKHR* pGetWin32HandleInfo, HANDLE* pHandle)
        {
            auto ptrGetMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetInstanceProcAddr(mInstanceRef, "vkGetMemoryWin32HandleKHR"));
            SAF_ASSERT(ptrGetMemoryWin32HandleKHR != nullptr);
            ptrGetMemoryWin32HandleKHR(mDeviceRef, pGetWin32HandleInfo, pHandle);
        }
#else
        inline void getMemoryFdKHR(const VkMemoryGetFdInfoKHR* pGetFdInfo, int* pFd)
        {
            auto ptrGetMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetInstanceProcAddr(mInstanceRef, "vkGetMemoryFdKHR"));
            SAF_ASSERT(ptrGetMemoryFdKHR != nullptr);
            ptrGetMemoryFdKHR(mDeviceRef, pGetFdInfo, pFd);
        }
#endif
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
         */
        Application(const ApplicationSettings& settings);

        ~Application();

        /**
         * @brief Run the @a Application.
         */
        void run();

        /**
         * @brief Close the @a Application.
         */
        void close();

        inline std::shared_ptr<ApplicationContext> getApplicationContext() const
        {
            return mApplicationContext;
        }

        /**
         * @brief Pushes a @a Layer to the layer stack hold by the @a Application.
         * @tparam T The layers type derived from the @a Layer type.
         */
        template <typename T>
        void pushLayer()
        {
            static_assert(std::is_base_of<Layer, T>::value, "Pushed type is not a Layer!");
            mLayerStack.emplace_back(std::make_unique<T>())->onAttach(this);
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

#ifdef SAF_SCRIPTING
        /**
         * @brief Renders information about all active lua scripts.
         */
        void uiRenderActiveScripts();
#endif

    private:
        /**
         * @brief Initializes Vulkan and GLFW.
         */
        bool initVulkanGLFW();
        /**
         * @brief Shuts down Vulkan and GLFW.
         */
        void shutdownVulkanGLFW();

    private:
        /** @brief The @a Applications name. */
        Str mName;
        /** @brief The @a Applications window width. */
        I32 mWindowWidth;
        /** @brief The @a Applications window height. */
        I32 mWindowHeight;
        /** @brief The @a Applications font size. */
        F32 mFontSize;
        /** @brief The @a Applications clear color. */
        Vec4 mClearColor;
        /** @brief True, if the @a Application is running, else False. */
        bool mRunning;

        /** @brief The @a Applications menubar callback. */
        std::function<void()> mMenubarCallback;

        /** @brief The @a Applications window handle. */
        GLFWwindow* mWindow;
        /** @brief The @a Applications vulkan context. */
        struct VulkanContext* mVulkanContext;

        /** @brief The @a Applications layer stack. */
        std::vector<std::unique_ptr<Layer>> mLayerStack;

        std::shared_ptr<ApplicationContext> mApplicationContext;
    };
} // namespace saf

#endif // APPLICATION_HPP
