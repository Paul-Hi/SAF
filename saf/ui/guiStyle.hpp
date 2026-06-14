/**
 * @file      guiStyle.hpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef GUI_STYLE_HPP
#define GUI_STYLE_HPP

#include "interVariableFont.hpp"
#include <imgui.h>
#include <regex>

namespace saf
{
    inline void setupImGuiStyleDark(F32 fontScaleMain, F32 fontScaleDpi)
    {
        ImGuiIO& io = ImGui::GetIO();

        ImFontConfig fontConfig;
        fontConfig.FontDataOwnedByAtlas = false;
        fontConfig.GlyphOffset          = ImVec2(0.0f, -1.0f); // optical centering for Inter
        io.FontDefault                  = io.Fonts->AddFontFromMemoryCompressedTTF(
            reinterpret_cast<const void*>(gInterVariableFontCompressedData),
            gInterVariableFontCompressedSize, 14.0f * fontScaleMain * fontScaleDpi, &fontConfig);

        io.ConfigWindowsMoveFromTitleBarOnly = true;

        ImVec4* colors = ImGui::GetStyle().Colors;

        const ImVec4 bgBase      = ImVec4(0.039f, 0.043f, 0.059f, 1.00f); // #0a0b0f — deepest
        const ImVec4 bgPrimary   = ImVec4(0.059f, 0.067f, 0.090f, 1.00f); // #0f1117 — window bg
        const ImVec4 bgSecondary = ImVec4(0.075f, 0.082f, 0.110f, 1.00f); // #131520 — elevated surface
        const ImVec4 bgTertiary  = ImVec4(0.086f, 0.094f, 0.122f, 1.00f); // #16182e — menubar
        const ImVec4 bgPopup     = ImVec4(0.051f, 0.059f, 0.082f, 1.00f); // #0d0f15 — popup/tooltip

        const ImVec4 frameBg     = ImVec4(0.051f, 0.059f, 0.086f, 1.00f); // #0d0f16
        const ImVec4 frameHover  = ImVec4(0.094f, 0.106f, 0.153f, 1.00f); // #181b27
        const ImVec4 frameActive = ImVec4(0.102f, 0.118f, 0.184f, 1.00f); // #1a1e2f — blue-tinted

        const ImVec4 borderSubtle = ImVec4(0.118f, 0.129f, 0.188f, 1.00f); // #1e2130
        const ImVec4 borderMid    = ImVec4(0.161f, 0.176f, 0.255f, 1.00f); // #292d41
        const ImVec4 borderStrong = ImVec4(0.212f, 0.231f, 0.333f, 1.00f); // #363b55

        const ImVec4 textPrimary   = ImVec4(0.886f, 0.894f, 0.925f, 1.00f); // #e2e4ec
        const ImVec4 textSecondary = ImVec4(0.533f, 0.569f, 0.659f, 1.00f); // #8891a8
        const ImVec4 textDisabled  = ImVec4(0.353f, 0.380f, 0.459f, 1.00f); // #5a6175

        const ImVec4 accentHover   = ImVec4(0.420f, 0.530f, 1.000f, 1.00f); // lighter — hover feedback
        const ImVec4 accent        = ImVec4(0.310f, 0.431f, 0.969f, 1.00f); // #4f6ef7 — rest
        const ImVec4 accentPressed = ImVec4(0.239f, 0.361f, 0.878f, 1.00f); // darker — pressed/active
        const ImVec4 accentDim     = ImVec4(0.310f, 0.431f, 0.969f, 0.20f); // 20% — text selection
        const ImVec4 accentSubtle  = ImVec4(0.310f, 0.431f, 0.969f, 0.25f); // 25% — docking preview

        const ImVec4 headerBg     = ImVec4(0.075f, 0.098f, 0.176f, 1.00f); // #13192d
        const ImVec4 headerHover  = ImVec4(0.094f, 0.122f, 0.212f, 1.00f); // #181f36
        const ImVec4 headerActive = ImVec4(0.110f, 0.145f, 0.247f, 1.00f); // #1c253f

        // -------------------------------------------------------------------------
        // Text
        // -------------------------------------------------------------------------
        colors[ImGuiCol_Text]           = textPrimary;
        colors[ImGuiCol_TextDisabled]   = textDisabled;
        colors[ImGuiCol_TextSelectedBg] = accentDim;

        // -------------------------------------------------------------------------
        // Backgrounds
        // -------------------------------------------------------------------------
        colors[ImGuiCol_WindowBg] = bgPrimary;
        colors[ImGuiCol_ChildBg]  = bgSecondary;
        colors[ImGuiCol_PopupBg]  = bgPopup;

        // -------------------------------------------------------------------------
        // Borders
        // -------------------------------------------------------------------------
        colors[ImGuiCol_Border]       = borderSubtle;
        colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

        // -------------------------------------------------------------------------
        // Frames (inputs, sliders, checkboxes)
        // -------------------------------------------------------------------------
        colors[ImGuiCol_FrameBg]        = frameBg;
        colors[ImGuiCol_FrameBgHovered] = frameHover;
        colors[ImGuiCol_FrameBgActive]  = frameActive;

        // -------------------------------------------------------------------------
        // Title bars
        // -------------------------------------------------------------------------
        colors[ImGuiCol_TitleBg]          = bgBase;
        colors[ImGuiCol_TitleBgActive]    = bgBase;
        colors[ImGuiCol_TitleBgCollapsed] = bgPrimary;

        // -------------------------------------------------------------------------
        // Menus & scrollbars
        // -------------------------------------------------------------------------
        colors[ImGuiCol_MenuBarBg]            = bgTertiary;
        colors[ImGuiCol_ScrollbarBg]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_ScrollbarGrab]        = borderSubtle;
        colors[ImGuiCol_ScrollbarGrabHovered] = borderMid;
        colors[ImGuiCol_ScrollbarGrabActive]  = borderStrong;

        // -------------------------------------------------------------------------
        // Interactive controls
        // -------------------------------------------------------------------------
        colors[ImGuiCol_CheckMark]        = accent;
        colors[ImGuiCol_SliderGrab]       = accent;
        colors[ImGuiCol_SliderGrabActive] = accentHover;

        // -------------------------------------------------------------------------
        // Buttons
        // -------------------------------------------------------------------------
        colors[ImGuiCol_Button]        = frameBg;
        colors[ImGuiCol_ButtonHovered] = frameHover;
        colors[ImGuiCol_ButtonActive]  = accentPressed;

        // -------------------------------------------------------------------------
        // Headers (collapsing headers, selectables, tree nodes, menu items)
        // -------------------------------------------------------------------------
        colors[ImGuiCol_Header]        = headerBg;
        colors[ImGuiCol_HeaderHovered] = headerHover;
        colors[ImGuiCol_HeaderActive]  = headerActive;

        // -------------------------------------------------------------------------
        // Separators
        // -------------------------------------------------------------------------
        colors[ImGuiCol_Separator]        = borderSubtle;
        colors[ImGuiCol_SeparatorHovered] = borderMid;
        colors[ImGuiCol_SeparatorActive]  = accent;

        // -------------------------------------------------------------------------
        // Resize grips — invisible at rest, accent on hover, pressed on active
        // -------------------------------------------------------------------------
        colors[ImGuiCol_ResizeGrip]        = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_ResizeGripHovered] = accent;
        colors[ImGuiCol_ResizeGripActive]  = accentHover;

        // -------------------------------------------------------------------------
        // Tabs (docking branch)
        // -------------------------------------------------------------------------
        colors[ImGuiCol_Tab]                       = bgBase;
        colors[ImGuiCol_TabHovered]                = frameHover;
        colors[ImGuiCol_TabSelected]               = bgSecondary;
        colors[ImGuiCol_TabSelectedOverline]       = accent;
        colors[ImGuiCol_TabDimmed]                 = bgBase;
        colors[ImGuiCol_TabDimmedSelected]         = bgPrimary;
        colors[ImGuiCol_TabDimmedSelectedOverline] = borderMid;
        // colors[ImGuiCol_TabBarBg]                  = bgBase;

        // -------------------------------------------------------------------------
        // Docking (docking branch)
        // -------------------------------------------------------------------------
        colors[ImGuiCol_DockingPreview] = accentSubtle;
        colors[ImGuiCol_DockingEmptyBg] = bgSecondary;

        // -------------------------------------------------------------------------
        // Plots
        // -------------------------------------------------------------------------
        colors[ImGuiCol_PlotLines]            = accent;
        colors[ImGuiCol_PlotLinesHovered]     = accentHover;
        colors[ImGuiCol_PlotHistogram]        = accentPressed;
        colors[ImGuiCol_PlotHistogramHovered] = accent;

        // -------------------------------------------------------------------------
        // Tables
        // -------------------------------------------------------------------------
        colors[ImGuiCol_TableHeaderBg]     = bgBase;
        colors[ImGuiCol_TableBorderStrong] = borderMid;
        colors[ImGuiCol_TableBorderLight]  = borderSubtle;
        colors[ImGuiCol_TableRowBg]        = ImVec4(0.059f, 0.067f, 0.090f, 0.50f);
        colors[ImGuiCol_TableRowBgAlt]     = ImVec4(0.075f, 0.082f, 0.110f, 0.50f);

        // -------------------------------------------------------------------------
        // Drag & drop
        // -------------------------------------------------------------------------
        colors[ImGuiCol_DragDropTarget] = accent;

        // -------------------------------------------------------------------------
        // Navigation
        // -------------------------------------------------------------------------
        colors[ImGuiCol_NavCursor]             = accent;
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.020f, 0.024f, 0.059f, 0.45f);
        colors[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.020f, 0.024f, 0.059f, 0.55f);

        // =========================================================================
        // Style metrics
        // =========================================================================
        ImGuiStyle& style = ImGui::GetStyle();

        style.Alpha         = 1.0f;
        style.DisabledAlpha = 0.6f;

        // Windows
        style.WindowPadding            = ImVec2(12.0f, 12.0f);
        style.WindowRounding           = 8.0f;
        style.WindowBorderSize         = 1.0f;
        style.WindowMinSize            = ImVec2(120.0f, 32.0f);
        style.WindowTitleAlign         = ImVec2(0.5f, 0.5f);
        style.WindowMenuButtonPosition = ImGuiDir_Right;

        // Child windows
        style.ChildRounding   = 8.0f;
        style.ChildBorderSize = 1.0f;

        // Popups
        style.PopupRounding   = 6.0f;
        style.PopupBorderSize = 1.0f;

        // Frames
        style.FramePadding    = ImVec2(10.0f, 6.0f);
        style.FrameRounding   = 6.0f;
        style.FrameBorderSize = 1.0f;

        // Item
        style.ItemSpacing      = ImVec2(8.0f, 8.0f);
        style.ItemInnerSpacing = ImVec2(6.0f, 5.0f);
        style.CellPadding      = ImVec2(10.0f, 6.0f);

        // Indent / columns
        style.IndentSpacing     = 12.0f;
        style.ColumnsMinSpacing = 4.0f;

        // Scrollbar
        style.ScrollbarSize     = 10.0f;
        style.ScrollbarRounding = 10.0f;

        // Grab
        style.GrabMinSize  = 10.0f;
        style.GrabRounding = std::round(10.0f) / 2.0f;

        // Tabs
        style.TabRounding                      = 6.0f;
        style.TabBorderSize                    = 0.0f;
        style.TabBarBorderSize                 = 1.0f;
        style.TabCloseButtonMinWidthUnselected = 16.0f;

        // Separator text
        style.SeparatorTextBorderSize = 1.0f;
        style.SeparatorTextAlign      = ImVec2(0.0f, 0.5f);
        style.SeparatorTextPadding    = ImVec2(20.0f, 3.0f);

        // Docking
        style.DockingSeparatorSize = 1.0f;

        // Misc
        style.ColorButtonPosition = ImGuiDir_Right;
        style.ButtonTextAlign     = ImVec2(0.5f, 0.5f);
        style.SelectableTextAlign = ImVec2(0.0f, 0.5f);
        style.LogSliderDeadzone   = 4.0f;
    }

    inline void setupImGuiStyleLight(F32 fontScaleMain, F32 fontScaleDpi)
    {
        ImGuiIO& io = ImGui::GetIO();

        ImFontConfig fontConfig;
        fontConfig.FontDataOwnedByAtlas = false;
        fontConfig.GlyphOffset          = ImVec2(0.0f, -1.0f); // optical centering for Inter
        io.FontDefault                  = io.Fonts->AddFontFromMemoryCompressedTTF(
            reinterpret_cast<const void*>(gInterVariableFontCompressedData),
            gInterVariableFontCompressedSize, 14.0f * fontScaleMain * fontScaleDpi, &fontConfig);

        io.ConfigWindowsMoveFromTitleBarOnly = true;

        ImVec4* colors = ImGui::GetStyle().Colors;

        // -------------------------------------------------------------------------
        // Base palette
        // -------------------------------------------------------------------------
        const ImVec4 bgBase      = ImVec4(0.996f, 0.996f, 1.000f, 1.00f); // #fefeff — pure surface
        const ImVec4 bgPrimary   = ImVec4(0.973f, 0.976f, 0.984f, 1.00f); // #f8f9fb — window bg
        const ImVec4 bgSecondary = ImVec4(0.945f, 0.949f, 0.961f, 1.00f); // #f1f2f5 — child/elevated
        const ImVec4 bgTertiary  = ImVec4(0.922f, 0.929f, 0.945f, 1.00f); // #ebedf1 — menubar
        const ImVec4 bgPopup     = ImVec4(0.996f, 0.996f, 1.000f, 1.00f); // #fefeff — popup/tooltip

        // Frames
        const ImVec4 frameBg     = ImVec4(0.918f, 0.922f, 0.937f, 1.00f); // #eaebef
        const ImVec4 frameHover  = ImVec4(0.882f, 0.890f, 0.914f, 1.00f); // #e1e3e9
        const ImVec4 frameActive = ImVec4(0.839f, 0.855f, 0.918f, 1.00f); // #d6daea — blue-tinted

        // Borders
        const ImVec4 borderSubtle = ImVec4(0.820f, 0.831f, 0.867f, 1.00f); // #d1d4dd
        const ImVec4 borderMid    = ImVec4(0.729f, 0.749f, 0.804f, 1.00f); // #babfcd
        const ImVec4 borderStrong = ImVec4(0.608f, 0.635f, 0.706f, 1.00f); // #9ba2b4

        // Text
        const ImVec4 textPrimary   = ImVec4(0.094f, 0.102f, 0.141f, 1.00f); // #181a24 — near black
        const ImVec4 textSecondary = ImVec4(0.380f, 0.408f, 0.490f, 1.00f); // #61687d
        const ImVec4 textDisabled  = ImVec4(0.588f, 0.612f, 0.678f, 1.00f); // #969cad

        // Accent — theme continuity
        const ImVec4 accentHover   = ImVec4(0.420f, 0.530f, 1.000f, 1.00f); // lighter — hover
        const ImVec4 accent        = ImVec4(0.310f, 0.431f, 0.969f, 1.00f); // #4f6ef7 — rest
        const ImVec4 accentPressed = ImVec4(0.239f, 0.361f, 0.878f, 1.00f); // darker — pressed
        const ImVec4 accentDim     = ImVec4(0.310f, 0.431f, 0.969f, 0.15f); // 15% — text selection
        const ImVec4 accentSubtle  = ImVec4(0.310f, 0.431f, 0.969f, 0.20f); // 20% — docking preview

        // Headers — blue-tinted light, progressive darkening rest → hover → active
        const ImVec4 headerBg     = ImVec4(0.847f, 0.867f, 0.941f, 1.00f); // #d8ddf0
        const ImVec4 headerHover  = ImVec4(0.800f, 0.824f, 0.922f, 1.00f); // #ccd2eb
        const ImVec4 headerActive = ImVec4(0.749f, 0.780f, 0.902f, 1.00f); // #bfc7e6

        // -------------------------------------------------------------------------
        // Text
        // -------------------------------------------------------------------------
        colors[ImGuiCol_Text]           = textPrimary;
        colors[ImGuiCol_TextDisabled]   = textDisabled;
        colors[ImGuiCol_TextSelectedBg] = accentDim;

        // -------------------------------------------------------------------------
        // Backgrounds
        // -------------------------------------------------------------------------
        colors[ImGuiCol_WindowBg] = bgPrimary;
        colors[ImGuiCol_ChildBg]  = bgSecondary;
        colors[ImGuiCol_PopupBg]  = bgPopup;

        // -------------------------------------------------------------------------
        // Borders
        // -------------------------------------------------------------------------
        colors[ImGuiCol_Border]       = borderSubtle;
        colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);

        // -------------------------------------------------------------------------
        // Frames
        // -------------------------------------------------------------------------
        colors[ImGuiCol_FrameBg]        = frameBg;
        colors[ImGuiCol_FrameBgHovered] = frameHover;
        colors[ImGuiCol_FrameBgActive]  = frameActive;

        // -------------------------------------------------------------------------
        // Title bars
        // -------------------------------------------------------------------------
        colors[ImGuiCol_TitleBg]          = bgTertiary;
        colors[ImGuiCol_TitleBgActive]    = bgSecondary;
        colors[ImGuiCol_TitleBgCollapsed] = bgTertiary;

        // -------------------------------------------------------------------------
        // Menus & scrollbars
        // -------------------------------------------------------------------------
        colors[ImGuiCol_MenuBarBg]            = bgTertiary;
        colors[ImGuiCol_ScrollbarBg]          = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_ScrollbarGrab]        = borderSubtle;
        colors[ImGuiCol_ScrollbarGrabHovered] = borderMid;
        colors[ImGuiCol_ScrollbarGrabActive]  = borderStrong;

        // -------------------------------------------------------------------------
        // Interactive controls
        // -------------------------------------------------------------------------
        colors[ImGuiCol_CheckMark]        = accent;
        colors[ImGuiCol_SliderGrab]       = accent;
        colors[ImGuiCol_SliderGrabActive] = accentHover;

        // -------------------------------------------------------------------------
        // Buttons
        // -------------------------------------------------------------------------
        colors[ImGuiCol_Button]        = frameBg;
        colors[ImGuiCol_ButtonHovered] = frameHover;
        colors[ImGuiCol_ButtonActive]  = accentPressed;

        // -------------------------------------------------------------------------
        // Header
        // -------------------------------------------------------------------------
        colors[ImGuiCol_Header]        = headerBg;
        colors[ImGuiCol_HeaderHovered] = headerHover;
        colors[ImGuiCol_HeaderActive]  = headerActive;

        // -------------------------------------------------------------------------
        // Separators
        // -------------------------------------------------------------------------
        colors[ImGuiCol_Separator]        = borderSubtle;
        colors[ImGuiCol_SeparatorHovered] = borderMid;
        colors[ImGuiCol_SeparatorActive]  = accent;

        // -------------------------------------------------------------------------
        // Resize grips
        // -------------------------------------------------------------------------
        colors[ImGuiCol_ResizeGrip]        = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_ResizeGripHovered] = accent;
        colors[ImGuiCol_ResizeGripActive]  = accentHover;

        // -------------------------------------------------------------------------
        // Tabs (docking branch)
        // -------------------------------------------------------------------------
        colors[ImGuiCol_Tab]                       = bgTertiary;
        colors[ImGuiCol_TabHovered]                = bgSecondary;
        colors[ImGuiCol_TabSelected]               = bgBase;
        colors[ImGuiCol_TabSelectedOverline]       = accent;
        colors[ImGuiCol_TabDimmed]                 = bgTertiary;
        colors[ImGuiCol_TabDimmedSelected]         = bgPrimary;
        colors[ImGuiCol_TabDimmedSelectedOverline] = borderMid;
        // colors[ImGuiCol_TabBarBg]                  = bgTertiary;

        // -------------------------------------------------------------------------
        // Docking (docking branch)
        // -------------------------------------------------------------------------
        colors[ImGuiCol_DockingPreview] = accentSubtle;
        colors[ImGuiCol_DockingEmptyBg] = bgSecondary;

        // -------------------------------------------------------------------------
        // Plots
        // -------------------------------------------------------------------------
        colors[ImGuiCol_PlotLines]            = accent;
        colors[ImGuiCol_PlotLinesHovered]     = accentHover;
        colors[ImGuiCol_PlotHistogram]        = accentPressed;
        colors[ImGuiCol_PlotHistogramHovered] = accent;

        // -------------------------------------------------------------------------
        // Tables
        // -------------------------------------------------------------------------
        colors[ImGuiCol_TableHeaderBg]     = bgTertiary;
        colors[ImGuiCol_TableBorderStrong] = borderMid;
        colors[ImGuiCol_TableBorderLight]  = borderSubtle;
        colors[ImGuiCol_TableRowBg]        = ImVec4(0.973f, 0.976f, 0.984f, 0.60f);
        colors[ImGuiCol_TableRowBgAlt]     = ImVec4(0.945f, 0.949f, 0.961f, 0.60f);

        // -------------------------------------------------------------------------
        // Drag & drop
        // -------------------------------------------------------------------------
        colors[ImGuiCol_DragDropTarget] = accent;

        // -------------------------------------------------------------------------
        // Navigation
        // -------------------------------------------------------------------------
        colors[ImGuiCol_NavCursor]             = accent;
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.10f, 0.10f, 0.10f, 0.70f);
        colors[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.800f, 0.820f, 0.900f, 0.35f);
        colors[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.800f, 0.820f, 0.900f, 0.45f);

        // =========================================================================
        // Style metrics
        // =========================================================================
        ImGuiStyle& style = ImGui::GetStyle();

        style.Alpha         = 1.0f;
        style.DisabledAlpha = 0.6f;

        // Windows
        style.WindowPadding            = ImVec2(12.0f, 12.0f);
        style.WindowRounding           = 8.0f;
        style.WindowBorderSize         = 1.0f;
        style.WindowMinSize            = ImVec2(120.0f, 32.0f);
        style.WindowTitleAlign         = ImVec2(0.5f, 0.5f);
        style.WindowMenuButtonPosition = ImGuiDir_Right;

        // Child windows
        style.ChildRounding   = 8.0f;
        style.ChildBorderSize = 1.0f;

        // Popups
        style.PopupRounding   = 6.0f;
        style.PopupBorderSize = 1.0f;

        // Frames
        style.FramePadding    = ImVec2(10.0f, 6.0f);
        style.FrameRounding   = 6.0f;
        style.FrameBorderSize = 1.0f;

        // Item
        style.ItemSpacing      = ImVec2(8.0f, 8.0f);
        style.ItemInnerSpacing = ImVec2(6.0f, 5.0f);
        style.CellPadding      = ImVec2(10.0f, 6.0f);

        // Indent / columns
        style.IndentSpacing     = 12.0f;
        style.ColumnsMinSpacing = 4.0f;

        // Scrollbar
        style.ScrollbarSize     = 10.0f;
        style.ScrollbarRounding = 10.0f;

        // Grab
        style.GrabMinSize  = 10.0f;
        style.GrabRounding = std::round(10.0f) / 2.0f;

        // Tabs
        style.TabRounding                      = 6.0f;
        style.TabBorderSize                    = 0.0f;
        style.TabBarBorderSize                 = 1.0f;
        style.TabCloseButtonMinWidthUnselected = 16.0f;

        // Separator text
        style.SeparatorTextBorderSize = 1.0f;
        style.SeparatorTextAlign      = ImVec2(0.0f, 0.5f);
        style.SeparatorTextPadding    = ImVec2(20.0f, 3.0f);

        // Docking
        style.DockingSeparatorSize = 1.0f;

        // Misc
        style.ColorButtonPosition = ImGuiDir_Right;
        style.ButtonTextAlign     = ImVec2(0.5f, 0.5f);
        style.SelectableTextAlign = ImVec2(0.0f, 0.5f);
        style.LogSliderDeadzone   = 4.0f;
    }

} // namespace saf

#endif // GUI_STYLE_HPP
