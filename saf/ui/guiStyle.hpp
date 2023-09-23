/**
 * @file      guiStyle.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2023
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef GUI_STYLE_HPP
#define GUI_STYLE_HPP

#include "interVariableFont.hpp"
#include <imgui.h>

namespace saf
{
    namespace theme
    {
        const ImColor black(0, 0, 0, 255);
        const ImColor white(255, 255, 255, 255);
        const ImColor brighten(255, 255, 255, 127);
        const ImColor darken(0, 0, 0, 127);
        const ImColor header(47, 47, 47, 255);
        const ImColor text(192, 192, 192, 255);
        const ImColor titlebar(21, 21, 21, 255);
        const ImColor titlebarCollapsed(9, 9, 9, 255);
        const ImColor background(36, 36, 36, 255);
        const ImColor backgroundDark(26, 26, 26, 255);
        const ImColor backgroundProperty(15, 15, 15, 255);
        const ImColor backgroundGrab(32, 32, 32, 255);
        const ImColor backgroundPopup(50, 50, 50, 255);
        const ImColor accentClick(104, 104, 104, 255);

        const ImColor textError(236, 50, 50, 255);
        const ImColor highlight(36, 180, 241, 255);
        const ImColor activeGrab(42, 125, 157, 255);
    }

    inline void setupImGuiStyle()
    {
        ImGuiIO &io = ImGui::GetIO();

        ImFontConfig fontConfig;
        fontConfig.FontDataOwnedByAtlas = false;
        io.FontDefault = io.Fonts->AddFontFromMemoryCompressedTTF(reinterpret_cast<const void *>(gInterVariableFontCompressedData), gInterVariableFontCompressedSize, 18.0f, &fontConfig);

        io.ConfigWindowsMoveFromTitleBarOnly = true; // Only move from title bar

        // Colors
        ImVec4 *colors = ImGui::GetStyle().Colors;

        colors[ImGuiCol_Text] = theme::text;
        colors[ImGuiCol_TextDisabled] = theme::brighten;
        colors[ImGuiCol_WindowBg] = theme::titlebar;       // Background of normal windows
        colors[ImGuiCol_ChildBg] = theme::background;      // Background of child windows
        colors[ImGuiCol_PopupBg] = theme::backgroundPopup; // Background of popups, menus, tooltips windows
        colors[ImGuiCol_Border] = theme::backgroundDark;
        colors[ImGuiCol_BorderShadow] = theme::black;
        colors[ImGuiCol_FrameBg] = theme::backgroundProperty; // Background of checkbox, radio button, plot, slider, text input
        colors[ImGuiCol_FrameBgHovered] = theme::backgroundProperty;
        colors[ImGuiCol_FrameBgActive] = theme::backgroundProperty;
        colors[ImGuiCol_TitleBg] = theme::titlebar;
        colors[ImGuiCol_TitleBgActive] = theme::titlebar;
        colors[ImGuiCol_TitleBgCollapsed] = theme::titlebarCollapsed;
        colors[ImGuiCol_MenuBarBg] = theme::black;
        colors[ImGuiCol_ScrollbarBg] = theme::black;
        colors[ImGuiCol_ScrollbarGrab] = theme::backgroundGrab;
        colors[ImGuiCol_ScrollbarGrabHovered] = theme::backgroundDark;
        colors[ImGuiCol_ScrollbarGrabActive] = theme::activeGrab;
        colors[ImGuiCol_CheckMark] = theme::highlight;
        colors[ImGuiCol_SliderGrab] = theme::backgroundGrab;
        colors[ImGuiCol_SliderGrabActive] = theme::activeGrab;
        colors[ImGuiCol_Button] = theme::backgroundGrab;
        colors[ImGuiCol_ButtonHovered] = theme::backgroundPopup;
        colors[ImGuiCol_ButtonActive] = theme::accentClick;
        colors[ImGuiCol_Header] = theme::header; // Header* colors are used for CollapsingHeader, TreeNode, Selectable, MenuItem
        colors[ImGuiCol_HeaderHovered] = theme::header;
        colors[ImGuiCol_HeaderActive] = theme::header;
        colors[ImGuiCol_Separator] = theme::backgroundPopup;
        colors[ImGuiCol_SeparatorHovered] = theme::backgroundDark;
        colors[ImGuiCol_SeparatorActive] = theme::highlight;
        colors[ImGuiCol_ResizeGrip] = theme::backgroundGrab; // Resize grip in lower-right and lower-left corners of windows.
        colors[ImGuiCol_ResizeGripHovered] = theme::backgroundDark;
        colors[ImGuiCol_ResizeGripActive] = theme::activeGrab;
        colors[ImGuiCol_Tab] = theme::titlebar; // TabItem in a TabBar
        colors[ImGuiCol_TabHovered] = theme::titlebar;
        colors[ImGuiCol_TabActive] = theme::titlebar;
        colors[ImGuiCol_TabUnfocused] = theme::titlebarCollapsed;
        colors[ImGuiCol_TabUnfocusedActive] = theme::titlebar;
        colors[ImGuiCol_DockingPreview] = theme::brighten;   // Preview overlay color when about to docking something
        colors[ImGuiCol_DockingEmptyBg] = theme::background; // Background color for empty node (e.g. CentralNode with no window docked into it)
        colors[ImGuiCol_PlotLines] = theme::text;
        colors[ImGuiCol_PlotLinesHovered] = theme::text;
        colors[ImGuiCol_PlotHistogram] = theme::text;
        colors[ImGuiCol_PlotHistogramHovered] = theme::text;
        colors[ImGuiCol_TableHeaderBg] = theme::header;             // Table header background
        colors[ImGuiCol_TableBorderStrong] = theme::backgroundDark; // Table outer and header borders (prefer using Alpha=1.0 here)
        colors[ImGuiCol_TableBorderLight] = theme::backgroundDark;  // Table inner borders (prefer using Alpha=1.0 here)
        colors[ImGuiCol_TableRowBg] = theme::backgroundDark;        // Table row background (even rows)
        colors[ImGuiCol_TableRowBgAlt] = theme::background;         // Table row background (odd rows)
        colors[ImGuiCol_TextSelectedBg] = theme::brighten;
        colors[ImGuiCol_DragDropTarget] = theme::highlight;        // Rectangle highlighting a drop target
        colors[ImGuiCol_NavHighlight] = theme::highlight;          // Gamepad/keyboard: current highlighted item
        colors[ImGuiCol_NavWindowingHighlight] = theme::highlight; // Highlight window when using CTRL+TAB
        colors[ImGuiCol_NavWindowingDimBg] = theme::darken;        // Darken/colorize entire screen behind the CTRL+TAB window list, when active
        colors[ImGuiCol_ModalWindowDimBg] = theme::darken;         // Darken/colorize entire screen behind a modal window, when one is active

        // Style
        ImGuiStyle &style = ImGui::GetStyle();
        style.FrameRounding = 3.5f;
        style.WindowRounding = 0.0f;
        style.ChildBorderSize = 1.0f;
        style.FrameBorderSize = 1.0f;
        style.PopupBorderSize = 1.0f;
        style.WindowBorderSize = 0.0f;
        style.IndentSpacing = 11.0f;
        style.Alpha = 1.0f;
        style.DisabledAlpha = 0.5f;
    }

} // namespace saf

#endif // GUI_STYLE_HPP
