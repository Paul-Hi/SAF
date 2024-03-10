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

#include "parameter.hpp"
#include <fstream>

#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

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
        virtual void onAttach(class Application* application) { (void)application; }

        /**
         * @brief Called when @a Layer is detached from the application.
         * @details Can be used to cleanup @a Layer members.
         */
        virtual void onDetach() {}

        /**
         * @brief Called when @a Layer is updated by the application.
         * @param[in] application A pointer to the @a Application the @a Layer is updated by.
         */
        virtual void onUpdate(class Application* application, float dt) { (void)application; }

        /**
         * @brief Called when the application is rendering the user interface.
         * @param[in] application A pointer to the @a Application the @a Layer is updated by.
         * @details Can be used to submit custom ImGui windows.
         */
        virtual void onUIRender(Application* application) {}
        struct Script
        {
            Str scriptName;
            Str fileName;

            bool running;
            sol::state state;
            sol::protected_function onAttach;
            sol::protected_function onUpdate;
            sol::protected_function onDetach;
            std::function<void(sol::state&)> setup;
            std::function<void(sol::state&)> cleanup;
            std::function<void(const Str&)> log;
        };

        template <typename SetupFn, typename CleanupFn>
        inline void loadScript(const Str& scriptName, const Str& fileName, const SetupFn& setup, const CleanupFn cleanup, const std::function<void(const Str&)>& log)
        {
            Script script;

            try
            {
                script.state.open_libraries(sol::lib::base);

                setupParametersInLuaState(script.state);

                setup(script.state);

                std::stringstream sstream;
                sstream << std::ifstream(fileName).rdbuf();
                std::string scriptSource = sstream.str();

                auto result = script.state.safe_script(scriptSource);

                if (!result.valid())
                {
                    sol::error err = result;
                    log(err.what());
                    log("Unrecoverable error - removing script.");
                    return;
                }

                script.scriptName = scriptName;
                script.fileName   = fileName;
                script.log        = log;

                script.running = false;
                script.setup   = setup;
                script.cleanup = cleanup;

                script.state["print"] = [log](sol::object v)
                {
                    log(v.as<std::string>());
                };

                script.onAttach = sol::protected_function(script.state["onAttach"], script.state["print"]);
                script.onUpdate = sol::protected_function(script.state["onUpdate"], script.state["print"]);
                script.onDetach = sol::protected_function(script.state["onDetach"], script.state["print"]);

                mScripts[scriptName] = std::move(script);
            }
            catch (const sol::error& e)
            {
                log(e.what());
            }
        }

        inline void unloadScript(const Str& scriptName)
        {
            auto it = mScripts.find(scriptName);
            if (it != mScripts.end())
            {
                unloadScript(it);
            }
        }

        inline void reloadScript(const Str& scriptName)
        {
            auto it = mScripts.find(scriptName);
            if (it != mScripts.end())
            {
                reloadScript(it);
            }
        }

        inline void startScript(const Str& scriptName)
        {
            auto it = mScripts.find(scriptName);
            if (it != mScripts.end())
            {
                startScript(it);
            }
        }

        inline void stopScript(const Str& scriptName)
        {
            auto it = mScripts.find(scriptName);
            if (it != mScripts.end())
            {
                stopScript(it);
            }
        }

    private:
        friend class Application;
        inline void unloadScript(std::map<Str, Layer::Script>::iterator& it)
        {
            Script& script = it->second;

            stopScript(it);

            script.cleanup(script.state);
            script.state.collect_garbage();
            it = mScripts.erase(it);
        }

        inline void reloadScript(std::map<Str, Layer::Script>::iterator& it)
        {
            Script& script = it->second;
            if (script.running)
            {
                script.onDetach();
            }
            script.cleanup(script.state);
            script.state.collect_garbage();
            Script backup = std::move(script);
            it            = mScripts.erase(it);

            std::cout << typeid(decltype(this)).name();

            loadScript(backup.scriptName, backup.fileName, backup.setup, backup.cleanup, backup.log);
        }

        inline void startScript(std::map<Str, Layer::Script>::iterator& it)
        {
            Script& script = it->second;
            if (script.running)
            {
                stopScript(it);
            }
            auto callResult = script.onAttach();

            if (!callResult.valid())
            {
                callResult.abandon();
                return;
            }

            script.running = true;
        }

        inline void stopScript(std::map<Str, Layer::Script>::iterator& it)
        {
            Script& script = it->second;
            if (!script.running)
            {
                return;
            }
            auto callResult = script.onDetach();

            if (!callResult.valid())
            {
                callResult.abandon();
            }

            script.running = false;
        }

        inline void updateScript(std::map<Str, Layer::Script>::iterator& it, F32 dt)
        {
            Script& script = it->second;
            if (!script.running)
            {
                return;
            }
            auto callResult = script.onUpdate(dt);

            if (!callResult.valid())
            {
                callResult.abandon();
                stopScript(it);
            }
            else
            {
                bool change = callResult;
                if (!change)
                {
                    stopScript(it);
                }
            }
        }

        std::map<Str, Script> mScripts;
    };
} // namespace saf

#endif // LAYER_HPP
