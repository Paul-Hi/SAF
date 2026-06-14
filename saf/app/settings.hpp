/**
 * @file      settings.hpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef SETTINGS_HPP
#define SETTINGS_HPP

#include <fstream>
#include <sstream>

namespace saf
{
    struct PersistentSettings
    {
        Str name;
        I32 windowWidth;
        I32 windowHeight;
        F32 fontScale;
        U32 theme;
        Vec4 clearColor;
        bool vSyncEnabled;
    };

    namespace detail
    {
        inline Str trim(std::string_view sv)
        {
            const auto start = sv.find_first_not_of(" \t\r\n");
            if (start == std::string_view::npos)
            {
                return {};
            }
            const auto end = sv.find_last_not_of(" \t\r\n");
            return Str(sv.substr(start, end - start + 1));
        }

        inline bool parseBool(const Str& s)
        {
            Str lower = s;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            return lower == "true" || lower == "1" || lower == "yes";
        }

        inline I32 parseInt(const Str& s, I32 fallback = 0)
        {
            try
            {
                return std::stoi(s);
            }
            catch (...)
            {
                return fallback;
            }
        }

        inline U32 parseUInt(const Str& s, U32 fallback = 0u)
        {
            try
            {
                return static_cast<U32>(std::stoul(s));
            }
            catch (...)
            {
                return fallback;
            }
        }

        inline F32 parseFloat(const Str& s, F32 fallback = 0.0f)
        {
            try
            {
                return std::stof(s);
            }
            catch (...)
            {
                return fallback;
            }
        }

    } // namespace detail

    inline bool loadPersistentSettings(PersistentSettings& settings, const Str& filePath)
    {
        std::ifstream file(filePath);
        if (!file.is_open())
            return false;

        bool hasName         = false;
        bool hasWindowWidth  = false;
        bool hasWindowHeight = false;
        bool hasFontScale    = false;
        bool hasTheme        = false;
        bool hasClearColorR  = false;
        bool hasClearColorG  = false;
        bool hasClearColorB  = false;
        bool hasClearColorA  = false;
        bool hasVSyncEnabled = false;

        Str line;
        while (std::getline(file, line))
        {
            const Str trimmed = detail::trim(line);

            if (trimmed.empty() || trimmed.front() == '[' || trimmed.front() == ';' || trimmed.front() == '#')
                continue;

            const auto sep = trimmed.find('=');
            if (sep == Str::npos)
                continue;

            const Str key   = detail::trim(trimmed.substr(0, sep));
            const Str value = detail::trim(trimmed.substr(sep + 1));

            if (key == "name")
            {
                settings.name = value;
                hasName       = true;
            }
            else if (key == "windowWidth")
            {
                settings.windowWidth = detail::parseInt(value, 1280);
                hasWindowWidth       = true;
            }
            else if (key == "windowHeight")
            {
                settings.windowHeight = detail::parseInt(value, 720);
                hasWindowHeight       = true;
            }
            else if (key == "fontScale")
            {
                settings.fontScale = detail::parseFloat(value, 1.0f);
                hasFontScale       = true;
            }
            else if (key == "theme")
            {
                settings.theme = detail::parseUInt(value, 0u);
                hasTheme       = true;
            }
            else if (key == "clearColorR")
            {
                settings.clearColor.x() = detail::parseFloat(value, 0.0f);
                hasClearColorR          = true;
            }
            else if (key == "clearColorG")
            {
                settings.clearColor.y() = detail::parseFloat(value, 0.0f);
                hasClearColorG          = true;
            }
            else if (key == "clearColorB")
            {
                settings.clearColor.z() = detail::parseFloat(value, 0.0f);
                hasClearColorB          = true;
            }
            else if (key == "clearColorA")
            {
                settings.clearColor.w() = detail::parseFloat(value, 1.0f);
                hasClearColorA          = true;
            }
            else if (key == "vSyncEnabled")
            {
                settings.vSyncEnabled = detail::parseBool(value);
                hasVSyncEnabled       = true;
            }
        }

        return hasName &&
               hasWindowWidth &&
               hasWindowHeight &&
               hasFontScale &&
               hasTheme &&
               hasClearColorR &&
               hasClearColorG &&
               hasClearColorB &&
               hasClearColorA &&
               hasVSyncEnabled;
    }

    // -----------------------------------------------------------------------------
    inline bool storePersistentSettings(const PersistentSettings& settings, const Str& filePath)
    {
        std::ofstream file(filePath);
        if (!file.is_open())
            return false;

        file << "[Application]\n";
        file << "name         = " << settings.name << '\n';
        file << "windowWidth  = " << settings.windowWidth << '\n';
        file << "windowHeight = " << settings.windowHeight << '\n';
        file << "fontScale    = " << settings.fontScale << '\n';
        file << "theme        = " << settings.theme << '\n';
        file << "clearColorR  = " << settings.clearColor.x() << '\n';
        file << "clearColorG  = " << settings.clearColor.y() << '\n';
        file << "clearColorB  = " << settings.clearColor.z() << '\n';
        file << "clearColorA  = " << settings.clearColor.w() << '\n';
        file << "vSyncEnabled = " << (settings.vSyncEnabled ? "true" : "false") << '\n';

        return true;
    }

} // namespace saf

#endif // SETTINGS_HPP