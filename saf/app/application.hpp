/**
 * @file      application.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2023
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#include "layer.hpp"
#include <vector>

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
        /** @brief The clear color of the @a Application window. */
        Vec4 clearColor;
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
        Application(const ApplicationSettings &settings);

        ~Application();

        /**
         * @brief Run the @a Application.
         */
        void run();

        void close();

        inline VkPhysicalDevice getPhysicalDevice()
        {
            return mPhysicalDevice;
        }

        inline VkDevice getDevice()
        {
            return mLogicalDevice;
        }

        inline VkQueue getQueue()
        {
            return mQueue;
        }

        inline VkCommandPool getCommandPool()
        {
            return mCommandPool;
        }

        inline VkCommandBuffer getCommandBuffer()
        {
            return mCommandBuffer;
        }

        template <typename T>
        void pushLayer()
        {
            static_assert(std::is_base_of<Layer, T>::value, "Pushed type is not a Layer!");
            mLayerStack.emplace_back(std::make_unique<T>())->onAttach(this);
        }

        void popLayer();

        inline void setMenubarCallback(std::function<void()> callback)
        {
            mMenubarCallback = callback;
        }

    private:
        bool initVulkanGLFW();
        void shutdownVulkanGLFW();

    private:
        Str mName;
        I32 mWindowWidth;
        I32 mWindowHeight;
        Vec4 mClearColor;
        bool mRunning;

        std::function<void()> mMenubarCallback;

        GLFWwindow *mWindow;
        struct VulkanContext *mVulkanContext;

        VkPhysicalDevice mPhysicalDevice;
        VkDevice mLogicalDevice;
        VkQueue mQueue;
        VkCommandPool mCommandPool;
        VkCommandBuffer mCommandBuffer;

        std::vector<std::unique_ptr<Layer>> mLayerStack;
    };
} // namespace saf

#endif // APPLICATION_HPP
