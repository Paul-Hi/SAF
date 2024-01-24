/**
 * @file      layer.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2024
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef LAYER_HPP
#define LAYER_HPP

struct GLFWwindow;
struct ImGui_ImplVulkanH_Window;

namespace saf
{
    /**
     * @brief @a Layer.
     */
    class Layer
    {
    public:
        /**
         * @brief Constructs a @a Layer.
         */
        Layer() noexcept = default;

        virtual ~Layer() = default;

        /**
         * @brief Called when @a Layer is attached to the application.
         * @details Can be used to initialized required @a Layer members.
         * @param[in] application A pointer to the @a Application the @a Layer is attached to.
         */
        virtual void onAttach(class Application *application) { (void)application; }

        /**
         * @brief Called when @a Layer is detached from the application.
         * @details Can be used to cleanup @a Layer members.
         */
        virtual void onDetach() {}

        /**
         * @brief Called when @a Layer is updated by the application.
         * @param[in] application A pointer to the @a Application the @a Layer is updated by.
         */
        virtual void onUpdate(class Application *application) { (void)application; }

        /**
         * @brief Called when the application is rendering the user interface.
         * @details Can be used to submit custom ImGui windows.
         */
        virtual void onUIRender() {}
    };
} // namespace saf

#endif // LAYER_HPP
