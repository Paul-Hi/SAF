/**
 * @file      gpuTimings.hpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef GPU_TIMINGS_HPP
#define GPU_TIMINGS_HPP

#include <algorithm>
#include <app/application.hpp>
#include <cstdio>
#include <imgui.h>
#include <implot.h>
#include <numeric>

#if defined(SAF_VULKAN_COMPUTE) || defined(SAF_CUDA_INTEROP)
namespace saf
{
    /**
     * @brief UI plotting the recent durations of every timing scope in a plot of its own, titled with their average.
     */
    class UIGpuTimings
    {
    public:
        /**
         * @brief Renders the ImGui window for the timing scopes of @a app. Call in Layer::onUIRender.
         * @param[in] app The application holding the scopes.
         * @param[in] title The title of the ImGui window.
         * @param[in,out] pOpen Passing pOpen != nullptr shows a window-closing widget in the upper-right corner of the window, which clicking will set the boolean to false when clicked.
         */
        static void render(const Application& app, const char* title, bool* pOpen = nullptr)
        {
            if (!ImGui::Begin(title, pOpen))
            {
                ImGui::End();
                return;
            }

            // One plot per scope with results; they share the window's height, but get at least minPlotHeight and scroll beyond
            constexpr F32 minPlotHeight                            = 100.0f;
            const std::span<const Application::TimingScope> scopes = app.getTimings();
            const auto plotCount                                   = std::count_if(scopes.begin(), scopes.end(), [](const Application::TimingScope& scope)
                                                                                   { return !scope.history.empty(); });
            const F32 spacing = ImGui::GetStyle().ItemSpacing.y;
            const F32 height  = plotCount == 0 ? 0.0f : std::max(minPlotHeight, (ImGui::GetContentRegionAvail().y - spacing * static_cast<F32>(plotCount - 1)) / static_cast<F32>(plotCount));

            for (size_t i = 0; i < scopes.size(); ++i)
            {
                const Str& name                    = scopes[i].name;
                const std::span<const F32> history = scopes[i].history;
                if (history.empty())
                {
                    continue;
                }
                const F32 average = std::accumulate(history.begin(), history.end(), 0.0f) / static_cast<F32>(history.size());
                // The ID after ### stays fixed while the average changes, so ImPlot keeps the plot's state
                char title[128];
                std::snprintf(title, sizeof(title), "%s: %.3f ms (%zu)###%s", name.c_str(), average, history.size(), name.c_str());
                if (ImPlot::BeginPlot(title, ImVec2(-1.0f, height), ImPlotFlags_NoLegend))
                {
                    // The newest result sits at x = 0, older ones to its left
                    ImPlot::SetupAxes("Result", "ms", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
                    ImPlot::SetupAxisLimits(ImAxis_X1, 1.0 - Application::timingHistoryLength, 0.0, ImPlotCond_Always);
                    // Colored by first use, so a scope keeps its color while others come and go
                    ImPlotSpec spec;
                    spec.LineColor = ImPlot::GetColormapColor(static_cast<I32>(i));
                    ImPlot::PlotLine(name.c_str(), history.data(), static_cast<I32>(history.size()), 1.0, 1.0 - static_cast<F64>(history.size()), spec);
                    ImPlot::EndPlot();
                }
            }
            ImGui::End();
        }
    };
} // namespace saf
#endif

#endif // GPU_TIMINGS_HPP
