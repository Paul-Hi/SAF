/**
 * @file      ui_helper.hpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef UI_HELPER_HPP
#define UI_HELPER_HPP

#include "iconsFA6.hpp"
#include <algorithm>
#include <app/parameter.hpp>
#include <core/types.hpp>

namespace saf
{
    class SampleParameter : public Parameter<std::tuple<U32, bool, U32>> // [samples, running, currentSample]
    {
    public:
        SampleParameter(const Str& name, const U32 samples)
            : Parameter<std::tuple<U32, bool, U32>>(name, std::make_tuple(samples, false, 0))
        {
        }

        ~SampleParameter() = default;

        bool onUIRender() override
        {
            bool changed = false;

            ImGui::BeginChild((mName + "##SampleParameterChild").c_str(), ImVec2(0, 0), ImGuiChildFlags_AutoResizeY);

            ImGui::Spacing();

            ImGui::Text(mName.c_str());

            auto& [samples, running, currentSample] = mValue;

            bool toggleRunning = ImGui::Button(running ? ICON_FA_PAUSE : ICON_FA_PLAY);

            ImGui::SameLine();

            if (toggleRunning)
            {
                running = !running;
                changed = true;
            }

            if (running)
            {
                ImGui::BeginDisabled();
            }

            changed |= ImGui::InputScalar("Samples", ImGuiDataType_U32, &samples);

            samples = std::max(samples, currentSample);

            if (running)
            {
                ImGui::EndDisabled();
            }

            ImGui::SameLine();

            if (ImGui::Button("Reset"))
            {
                running       = false;
                currentSample = 0;
                changed       = true;
            }

            ImGui::Text("Current Sample: %d", currentSample);

            ImGui::Spacing();

            ImGui::EndChild();

            return changed;
        }

        void onUpdate()
        {
            auto& [samples, running, currentSample] = mValue;

            if (running)
            {
                currentSample = std::min(currentSample + 1, samples);
                if (currentSample >= samples)
                {
                    running = false;
                }
            }
        }

        void reset()
        {
            auto& [samples, running, currentSample] = mValue;

            running       = false;
            currentSample = 0;
        }
    };
} // namespace saf

#endif // UI_HELPER_HPP