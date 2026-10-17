/**
 * @file      layer.hpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef LAYER_HPP
#define LAYER_HPP

#include <app/parameter.hpp>

struct GLFWwindow;

namespace saf
{
    /**
     * @brief Layer.
     */
    class Layer
    {
    public:
        /**
         * @brief Constructs a Layer.
         */
        Layer() noexcept = default;

        virtual ~Layer() = default;

        /**
         * @brief Called when the Layer is attached to the application.
         * @details Can be used to initialize required Layer members.
         * @param[in] application A pointer to the Application the Layer is attached to.
         */
        virtual void onAttach(class Application* application) { (void)application; }

        /**
         * @brief Called when the Layer is detached from the application.
         * @details Can be used to clean up Layer members.
         * @param[in] application A pointer to the Application the Layer is detached from.
         */
        virtual void onDetach(Application* application) { (void)application; }

        /**
         * @brief Called when the Layer is updated by the application.
         * @param[in] application A pointer to the Application the Layer is updated by.
         * @param[in] dt The time since the previous frame in seconds.
         */
        virtual void onUpdate(class Application* application, float dt) { (void)application; }

        /**
         * @brief Called when the application is rendering the user interface.
         * @details Can be used to submit custom ImGui windows.
         * @param[in] application A pointer to the Application the Layer is rendered by.
         */
        virtual void onUIRender(Application* application) {}
    };
} // namespace saf

#endif // LAYER_HPP
