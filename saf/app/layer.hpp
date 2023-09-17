/**
 * @file      layer.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2023
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

        virtual void onAttach(class Application *application) {}
        virtual void onDetach() {}
        virtual void onUpdate(class Application *application) {}
        virtual void onUIRender() {}
    };
} // namespace saf

#endif // LAYER_HPP
