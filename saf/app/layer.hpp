/**
 * @file      layer.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2025
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef LAYER_HPP
#define LAYER_HPP

#include "parameter.hpp"
#include <fstream>

#ifdef SAF_SCRIPTING
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>
#endif
#ifdef SAF_FILE_WATCH
#include <efsw/FileSystem.hpp>
#include <efsw/efsw.hpp>
#endif

struct GLFWwindow;
struct VulkanWindow;

namespace saf
{

#ifdef SAF_SCRIPTING
    /**
     * @brief Data for a lua script.
     */
    struct Script
    {
        /** @brief Name (ID) of the script. */
        Str scriptName;
        /** @brief Filename the script was loaded from. */
        Str fileName;

        /** @brief True if the script is running, else False. */
        bool running;
        /** @brief The scripts lua state. */
        sol::state state;
        /** @brief A lua function called on attach. */
        sol::protected_function onAttach;
        /** @brief A lua function called on update. */
        sol::protected_function onUpdate;
        /** @brief A lua function called on detach. */
        sol::protected_function onDetach;
        /** @brief A function used to setup the lua state. */
        std::function<void(sol::state&)> setup;
        /** @brief A function used to cleanup the lua state. */
        std::function<void(sol::state&)> cleanup;
        /** @brief A function used to log script output. */
        std::function<void(const Str&)> log;

#ifdef SAF_FILE_WATCH
        efsw::WatchID watchID;
#endif
    };
#endif

    /**
     * @brief @a Layer.
     */
    class Layer
    {
    public:
        /**
         * @brief Constructs a @a Layer.
         */
        Layer() noexcept
#ifdef SAF_FILE_WATCH
            : mScriptUpdateListener(this)
        {
            mScriptWatcher.watch();
        }
#else
            = default;
#endif

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

#ifdef SAF_SCRIPTING
        int isSubstring(const saf::Str& s1, const saf::Str& s2)
        {
            if (s2.find(s1) != saf::Str::npos)
                return s2.find(s1);
            return -1;
        }

        /**
         * @brief Loads a lua script from a file.
         * @tparam SetupFn The type of @a setup.
         * @tparam CleanupFn The type of @a cleanup.
         * @param scriptName The name of the script used as ID.
         * @param fileName The filename to load the code from.
         * @param setup Function setting up the lua state.
         * @param cleanup Function cleaning up the lua script.
         * @param log A logging callback.
         */
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

                auto tostring = sol::protected_function(script.state["tostring"]);

                script.state["print"] = [tostring, log](sol::variadic_args args)
                {
                    std::stringstream ss;
                    for (auto arg : args)
                    {
                        ss << tostring(arg.as<sol::object>()).get<std::string>() << "\t";
                    }

                    log(ss.str());
                };

                script.onAttach = sol::protected_function(script.state["onAttach"], script.state["print"]);
                script.onUpdate = sol::protected_function(script.state["onUpdate"], script.state["print"]);
                script.onDetach = sol::protected_function(script.state["onDetach"], script.state["print"]);

#ifdef SAF_FILE_WATCH
                // watch for updates
                auto dirName = efsw::FileSystem::getCurrentWorkingDirectory();
                efsw::FileSystem::dirAddSlashAtEnd(dirName);
                if (isSubstring(dirName, fileName) >= 0)
                {
                    dirName = "";
                }
                auto completeFilename = dirName + fileName;

                dirName = efsw::FileSystem::pathRemoveFileName(completeFilename);

                const auto& watchedDirs = mScriptWatcher.directories();
                if (std::find(watchedDirs.begin(), watchedDirs.end(), dirName) == watchedDirs.end())
                {
                    script.watchID = checkWatchID(mScriptWatcher.addWatch(dirName, &mScriptUpdateListener, false));
                }
#endif

                mScripts[scriptName] = std::move(script);
            }
            catch (const sol::error& e)
            {
                log(e.what());
            }
        }

        /**
         * @brief Unload a lua script.
         * @param scriptName The name (ID) of the script.
         */
        inline void
        unloadScript(const Str& scriptName)
        {
            auto it = mScripts.find(scriptName);
            if (it != mScripts.end())
            {
                unloadScript(it);
            }
        }

        /**
         * @brief Reload a lua script.
         * @param scriptName The name (ID) of the script.
         */
        inline void reloadScript(const Str& scriptName)
        {
            auto it = mScripts.find(scriptName);
            if (it != mScripts.end())
            {
                reloadScript(it);
            }
        }

        /**
         * @brief Start the execution of a lua script.
         * @param scriptName The name (ID) of the script.
         */
        inline void startScript(const Str& scriptName)
        {
            auto it = mScripts.find(scriptName);
            if (it != mScripts.end())
            {
                startScript(it);
            }
        }

        /**
         * @brief Stop the execution of a lua script.
         * @param scriptName The name (ID) of the script.
         */
        inline void stopScript(const Str& scriptName)
        {
            auto it = mScripts.find(scriptName);
            if (it != mScripts.end())
            {
                stopScript(it);
            }
        }
#endif

    private:
        friend class Application;

#ifdef SAF_SCRIPTING
        /**
         * @brief Unload a lua script by an iterator.
         * @param it The iterator pointing to a script.
         */
        inline void unloadScript(std::map<Str, Script>::iterator& it)
        {
            Script& script = it->second;

            stopScript(it);

            script.cleanup(script.state);
            script.state.collect_garbage();

#ifdef SAF_FILE_WATCH
            mScriptWatcher.removeWatch(script.watchID);
#endif

            it = mScripts.erase(it);
        }

        /**
         * @brief Reload a lua script by an iterator.
         * @param it The iterator pointing to a script.
         */
        inline void reloadScript(std::map<Str, Script>::iterator& it)
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

            loadScript(backup.scriptName, backup.fileName, backup.setup, backup.cleanup, backup.log);
        }

        /**
         * @brief Start the execution of a lua script identified by an iterator.
         * @param it The iterator pointing to a script.
         */
        inline void startScript(std::map<Str, Script>::iterator& it)
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

        /**
         * @brief Stop the execution of a lua script identified by an iterator.
         * @param it The iterator pointing to a script.
         */
        inline void stopScript(std::map<Str, Script>::iterator& it)
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

        /**
         * @brief Call the update function of a lua script identified by an iterator.
         * @param it The iterator pointing to a script.
         * @param dt Time in seconds since last call.
         */
        inline void updateScript(std::map<Str, Script>::iterator& it, F32 dt)
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

        /** @brief Map of loaded scripts, name to script data.*/
        std::map<Str, Script> mScripts;

#ifdef SAF_FILE_WATCH
        class ScriptUpdateListener : public efsw::FileWatchListener
        {
        public:
            ScriptUpdateListener(Layer* layerPtr)
                : mLayerPtr(layerPtr)
            {
            }

            void handleFileAction(efsw::WatchID watchid, const std::string& dir, const std::string& filename, efsw::Action action, std::string oldFilename) override
            {
                mLayerPtr->scriptUpdateCallback(watchid, dir, filename, action, oldFilename);
            }

        private:
            Layer* mLayerPtr;
        };

        inline efsw::WatchID checkWatchID(efsw::WatchID watchid)
        {
            switch (watchid)
            {
            case efsw::Errors::FileNotFound:
            case efsw::Errors::FileRepeated:
            case efsw::Errors::FileOutOfScope:
            case efsw::Errors::FileRemote:
            case efsw::Errors::WatcherFailed:
            case efsw::Errors::Unspecified:
            {
                std::cerr << efsw::Errors::Log::getLastErrorLog().c_str() << std::endl;
                break;
            }
            default:
                break;
            }

            return watchid;
        }

        inline void scriptUpdateCallback(efsw::WatchID watchid, const std::string& dir, const std::string& filename, efsw::Action action, std::string oldFilename)
        {
            switch (action)
            {
            case efsw::Actions::Add:
                // Ignore - we watch only the script file atm.
                break;
            case efsw::Actions::Delete:
                // Remove Script
                {
                    auto it = std::find_if(mScripts.begin(), mScripts.end(),
                                           [this, &filename](const auto& pair) -> bool
                                           {
                                               return efsw::FileSystem::fileNameFromPath(pair.second.fileName) == filename;
                                           });
                    if (it != mScripts.end())
                    {
                        unloadScript(it);
                    }
                }
                break;
            case efsw::Actions::Modified:
                // Reload Script
                {
                    auto it = std::find_if(mScripts.begin(), mScripts.end(),
                                           [this, &filename](const auto& pair) -> bool
                                           {
                                               return efsw::FileSystem::fileNameFromPath(pair.second.fileName) == filename;
                                           });
                    if (it != mScripts.end())
                    {
                        reloadScript(it);
                    }
                }
                break;
            case efsw::Actions::Moved:
                // Ignore - we watch only the script file atm.
                break;
            default:
                break;
            }
        }

        efsw::FileWatcher mScriptWatcher;
        ScriptUpdateListener mScriptUpdateListener;
#endif
#endif
    };
} // namespace saf

#endif // LAYER_HPP
