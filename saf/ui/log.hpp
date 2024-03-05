/**
 * @file      log.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2024
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef LOG_HPP
#define LOG_HPP

#include <imgui.h>

namespace saf
{
    // Simple Log, similar to ImGui demo
    class UILog
    {
    public:
        static UILog& get()
        {
            static UILog log;

            return log;
        }

        ~UILog() = default;

        void clearLog()
        {
            mBuffer.clear();
            mLineOffsets.clear();
            mLineOffsets.push_back(0);
        }

        void add(const char* fmt, ...) IM_FMTARGS(2)
        {
            int oldSize = mBuffer.size();
            va_list args;
            va_start(args, fmt);
            mBuffer.appendfv(fmt, args);
            if (*(mBuffer.end() - 1) != '\n')
            {
                mBuffer.append("\n"); // Auto Newline
            }
            va_end(args);
            for (int newSize = mBuffer.size(); oldSize < newSize; oldSize++)
            {
                if (mBuffer[oldSize] == '\n')
                {
                    mLineOffsets.push_back(oldSize + 1);
                }
            }
        }

        void render(const char* title, bool* pOpen = NULL)
        {
            if (!ImGui::Begin(title, pOpen))
            {
                ImGui::End();
                return;
            }

            if (ImGui::BeginPopup("Options"))
            {
                ImGui::Checkbox("Auto-scroll", &mAutoScroll);
                ImGui::EndPopup();
            }

            if (ImGui::Button("Options"))
            {
                ImGui::OpenPopup("Options");
            }
            ImGui::SameLine();
            bool clear = ImGui::Button("Clear");
            ImGui::SameLine();
            bool copy = ImGui::Button("Copy");
            ImGui::SameLine();
            mFilter.Draw("Filter", -100.0f);

            ImGui::Separator();

            if (ImGui::BeginChild("Scrolling", ImVec2(0, 0), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar))
            {
                if (clear)
                {
                    clearLog();
                }
                if (copy)
                {
                    ImGui::LogToClipboard();
                }

                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
                const char* bufferStart = mBuffer.begin();
                const char* bufferEnd   = mBuffer.end();
                if (mFilter.IsActive())
                {
                    for (int nr = 0; nr < mLineOffsets.Size; nr++)
                    {
                        const char* lineStart = bufferStart + mLineOffsets[nr];
                        const char* lineEnd   = (nr + 1 < mLineOffsets.Size) ? (bufferStart + mLineOffsets[nr + 1] - 1) : bufferEnd;
                        if (mFilter.PassFilter(lineStart, lineEnd))
                        {
                            ImGui::TextUnformatted(lineStart, lineEnd);
                        }
                    }
                }
                else
                {
                    ImGuiListClipper clipper;
                    clipper.Begin(mLineOffsets.Size);
                    while (clipper.Step())
                    {
                        for (int nr = clipper.DisplayStart; nr < clipper.DisplayEnd; nr++)
                        {
                            const char* lineStart = bufferStart + mLineOffsets[nr];
                            const char* lineEnd   = (nr + 1 < mLineOffsets.Size) ? (bufferStart + mLineOffsets[nr + 1] - 1) : bufferEnd;
                            ImGui::TextUnformatted(lineStart, lineEnd);
                        }
                    }
                    clipper.End();
                }
                ImGui::PopStyleVar();

                if (mAutoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
                {
                    ImGui::SetScrollHereY(1.0f);
                }
            }
            ImGui::EndChild();
            ImGui::End();
        }

    private:
        UILog()
        {
            mAutoScroll = true;
            clearLog();
        }

        ImGuiTextBuffer mBuffer;
        ImGuiTextFilter mFilter;
        ImVector<int> mLineOffsets;
        bool mAutoScroll;
    };
} // namespace saf

#endif // LOG_HPP
