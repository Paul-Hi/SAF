/**
 * @file      guiStyle.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2024
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
    inline void setupImGuiStyle(F32 fontSize)
    {
        ImGuiIO& io = ImGui::GetIO();

        ImFontConfig fontConfig;
        fontConfig.FontDataOwnedByAtlas = false;
        io.FontDefault                  = io.Fonts->AddFontFromMemoryCompressedTTF(reinterpret_cast<const void*>(gInterVariableFontCompressedData), gInterVariableFontCompressedSize, fontSize, &fontConfig);

        io.ConfigWindowsMoveFromTitleBarOnly = true; // Only move from title bar

        // Colors
        ImVec4* colors = ImGui::GetStyle().Colors;

        colors[ImGuiCol_Text]                  = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        colors[ImGuiCol_TextDisabled]          = ImVec4(0.422921746969223f, 0.466429591178894f, 0.6008583307266235f, 1.0f);
        colors[ImGuiCol_WindowBg]              = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.1019607856869698f, 1.0f);
        colors[ImGuiCol_ChildBg]               = ImVec4(0.09411764889955521f, 0.1019607856869698f, 0.1176470592617989f, 1.0f);
        colors[ImGuiCol_PopupBg]               = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.1019607856869698f, 1.0f);
        colors[ImGuiCol_Border]                = ImVec4(0.1568627506494522f, 0.168627455830574f, 0.1921568661928177f, 1.0f);
        colors[ImGuiCol_BorderShadow]          = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.1019607856869698f, 1.0f);
        colors[ImGuiCol_FrameBg]               = ImVec4(0.1137254908680916f, 0.125490203499794f, 0.1529411822557449f, 1.0f);
        colors[ImGuiCol_FrameBgHovered]        = ImVec4(0.1568627506494522f, 0.168627455830574f, 0.1921568661928177f, 1.0f);
        colors[ImGuiCol_FrameBgActive]         = ImVec4(0.1568627506494522f, 0.168627455830574f, 0.1921568661928177f, 1.0f);
        colors[ImGuiCol_TitleBg]               = ImVec4(0.0470588244497776f, 0.05490196123719215f, 0.07058823853731155f, 1.0f);
        colors[ImGuiCol_TitleBgActive]         = ImVec4(0.0470588244497776f, 0.05490196123719215f, 0.07058823853731155f, 1.0f);
        colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.1019607856869698f, 1.0f);
        colors[ImGuiCol_MenuBarBg]             = ImVec4(0.09803921729326248f, 0.105882354080677f, 0.1215686276555061f, 1.0f);
        colors[ImGuiCol_ScrollbarBg]           = ImVec4(0.0470588244497776f, 0.05490196123719215f, 0.07058823853731155f, 1.0f);
        colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
        colors[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.1568627506494522f, 0.168627455830574f, 0.1921568661928177f, 1.0f);
        colors[ImGuiCol_ScrollbarGrabActive]   = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
        colors[ImGuiCol_CheckMark]             = ImVec4(1.0f, 0.7607843279838562f, 0.4980392158031464f, 1.0f);
        colors[ImGuiCol_SliderGrab]            = ImVec4(1.0f, 0.7607843279838562f, 0.4980392158031464f, 1.0f);
        colors[ImGuiCol_SliderGrabActive]      = ImVec4(1.0f, 0.6117647290229797f, 0.4980392158031464f, 1.0f);
        colors[ImGuiCol_Button]                = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
        colors[ImGuiCol_ButtonHovered]         = ImVec4(0.1803921610116959f, 0.1882352977991104f, 0.196078434586525f, 1.0f);
        colors[ImGuiCol_ButtonActive]          = ImVec4(0.1529411822557449f, 0.1529411822557449f, 0.1529411822557449f, 1.0f);
        colors[ImGuiCol_Header]                = ImVec4(0.1411764770746231f, 0.1647058874368668f, 0.2078431397676468f, 1.0f);
        colors[ImGuiCol_HeaderHovered]         = ImVec4(0.105882354080677f, 0.105882354080677f, 0.105882354080677f, 1.0f);
        colors[ImGuiCol_HeaderActive]          = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.1019607856869698f, 1.0f);
        colors[ImGuiCol_Separator]             = ImVec4(0.1294117718935013f, 0.1490196138620377f, 0.1921568661928177f, 1.0f);
        colors[ImGuiCol_SeparatorHovered]      = ImVec4(0.1568627506494522f, 0.1843137294054031f, 0.250980406999588f, 1.0f);
        colors[ImGuiCol_SeparatorActive]       = ImVec4(0.1568627506494522f, 0.1843137294054031f, 0.250980406999588f, 1.0f);
        colors[ImGuiCol_ResizeGrip]            = ImVec4(0.1450980454683304f, 0.1450980454683304f, 0.1450980454683304f, 1.0f);
        colors[ImGuiCol_ResizeGripHovered]     = ImVec4(1.0f, 0.7607843279838562f, 0.4980392158031464f, 1.0f);
        colors[ImGuiCol_ResizeGripActive]      = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        colors[ImGuiCol_Tab]                   = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.1019607856869698f, 1.0f);
        colors[ImGuiCol_TabHovered]            = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
        colors[ImGuiCol_TabActive]             = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
        colors[ImGuiCol_TabUnfocused]          = ImVec4(0.0784313753247261f, 0.08627451211214066f, 0.1019607856869698f, 1.0f);
        colors[ImGuiCol_TabUnfocusedActive]    = ImVec4(0.125490203499794f, 0.2745098173618317f, 0.572549045085907f, 1.0f);
        colors[ImGuiCol_DockingPreview]        = ImVec4(0.9372549057006836f, 0.9372549057006836f, 0.9372549057006836f, 1.0f);
        colors[ImGuiCol_DockingEmptyBg]        = ImVec4(0.09411764889955521f, 0.1019607856869698f, 0.1176470592617989f, 1.0f);
        colors[ImGuiCol_PlotLines]             = ImVec4(0.5215686559677124f, 0.6000000238418579f, 0.7019608020782471f, 1.0f);
        colors[ImGuiCol_PlotLinesHovered]      = ImVec4(0.03921568766236305f, 0.9803921580314636f, 0.9803921580314636f, 1.0f);
        colors[ImGuiCol_PlotHistogram]         = ImVec4(1.0f, 0.8363728523254395f, 0.6566523313522339f, 1.0f);
        colors[ImGuiCol_PlotHistogramHovered]  = ImVec4(0.95686274766922f, 0.95686274766922f, 0.95686274766922f, 1.0f);
        colors[ImGuiCol_TableHeaderBg]         = ImVec4(0.0470588244497776f, 0.05490196123719215f, 0.07058823853731155f, 1.0f);
        colors[ImGuiCol_TableBorderStrong]     = ImVec4(0.0470588244497776f, 0.05490196123719215f, 0.07058823853731155f, 1.0f);
        colors[ImGuiCol_TableBorderLight]      = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
        colors[ImGuiCol_TableRowBg]            = ImVec4(0.1176470592617989f, 0.1333333402872086f, 0.1490196138620377f, 1.0f);
        colors[ImGuiCol_TableRowBgAlt]         = ImVec4(0.09803921729326248f, 0.105882354080677f, 0.1215686276555061f, 1.0f);
        colors[ImGuiCol_TextSelectedBg]        = ImVec4(0.9372549057006836f, 0.9372549057006836f, 0.9372549057006836f, 1.0f);
        colors[ImGuiCol_DragDropTarget]        = ImVec4(0.4980392158031464f, 0.5137255191802979f, 1.0f, 1.0f);
        colors[ImGuiCol_NavHighlight]          = ImVec4(0.2666666805744171f, 0.2901960909366608f, 1.0f, 1.0f);
        colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.4980392158031464f, 0.5137255191802979f, 1.0f, 1.0f);
        colors[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.196078434586525f, 0.1764705926179886f, 0.5450980663299561f, 0.08627451211214066f);
        colors[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.196078434586525f, 0.1764705926179886f, 0.5450980663299561f, 0.08627451211214066f);

        // Style
        ImGuiStyle& style               = ImGui::GetStyle();
        style.Alpha                     = 1.0f;
        style.DisabledAlpha             = 0.5f;
        style.WindowPadding             = ImVec2(8.0f, 8.0f);
        style.WindowRounding            = 6.0f;
        style.WindowBorderSize          = 0.0f;
        style.WindowMinSize             = ImVec2(20.0f, 20.0f);
        style.WindowTitleAlign          = ImVec2(0.5f, 0.5f);
        style.WindowMenuButtonPosition  = ImGuiDir_Right;
        style.ChildRounding             = 6.0f;
        style.ChildBorderSize           = 1.0f;
        style.PopupRounding             = 6.0f;
        style.PopupBorderSize           = 1.0f;
        style.FramePadding              = ImVec2(10.0f, 4.0f);
        style.FrameRounding             = 6.0f;
        style.FrameBorderSize           = 0.0f;
        style.ItemSpacing               = ImVec2(8.0f, 6.0f);
        style.ItemInnerSpacing          = ImVec2(6.0f, 4.0f);
        style.CellPadding               = ImVec2(12.0f, 9.0f);
        style.IndentSpacing             = 0.0f;
        style.ColumnsMinSpacing         = 4.0f;
        style.ScrollbarSize             = 12.0f;
        style.ScrollbarRounding         = 6.0f;
        style.GrabMinSize               = 8.0f;
        style.GrabRounding              = 6.0f;
        style.TabRounding               = 6.0f;
        style.TabBorderSize             = 0.0f;
        style.TabMinWidthForCloseButton = 0.0f;
        style.ColorButtonPosition       = ImGuiDir_Right;
        style.ButtonTextAlign           = ImVec2(0.5f, 0.5f);
        style.SelectableTextAlign       = ImVec2(0.0f, 0.0f);

        style.DockingSeparatorSize     = 1.0f;
    }

} // namespace saf

#endif // GUI_STYLE_HPP
