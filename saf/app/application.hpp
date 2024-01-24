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

        /**
         * @brief Close the @a Application.
         */
        void close();

        /**
         * @brief Retrieves the @a VkPhysicalDevice of the @a Application.
         * @return The @a VkPhysicalDevice of the @a Application.
         */
        VkPhysicalDevice getPhysicalDevice();

        /**
         * @brief Retrieves the @a VkDevice of the @a Application.
         * @return The @a VkDevice of the @a Application.
         */
        VkDevice getDevice();

        /**
         * @brief Retrieves the @a VkQueue of the @a Application.
         * @return The @a VkQueue of the @a Application.
         */
        VkQueue getQueue();

        /**
         * @brief Retrieves the @a VkCommandPool of the @a Application.
         * @return The @a VkCommandPool of the @a Application.
         */
        VkCommandPool getCommandPool();

        /**
         * @brief Retrieves the @a VkCommandBuffer of the @a Application.
         * @return The @a VkCommandBuffer of the @a Application.
         */
        VkCommandBuffer getCommandBuffer();

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

    private:
        bool initVulkanGLFW();
        void shutdownVulkanGLFW();

    private:
        Str mName;
        I32 mWindowWidth;
        I32 mWindowHeight;
        F32 mFontSize;
        Vec4 mClearColor;
        bool mRunning;

        std::function<void()> mMenubarCallback;

        GLFWwindow *mWindow;
        struct VulkanContext *mVulkanContext;

        std::vector<std::unique_ptr<Layer>> mLayerStack;
    };
} // namespace saf

#endif // APPLICATION_HPP
