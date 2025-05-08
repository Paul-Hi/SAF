#include "demo.cuh"
#include "random.hpp"
#include "test.hpp"
#include <app/application.hpp>
#include <app/parameter.hpp>
#include <core/image.hpp>
#include <core/types.hpp>
#include <imgui.h>
#include <implot.h>
#include <ui/imguiBackend.hpp>
#include <ui/log.hpp>

using namespace saf;

class DemoLayer : public Layer
{
public:
    virtual void onAttach(Application* application) override
    {
        mData.resize(720 * 720, Eigen::Vector4<Byte>(255, 0, 0, 255));
#ifdef SAF_CUDA_INTEROP
        mImage = std::make_shared<Image>(application->getApplicationContext(), 720, 720, VK_FORMAT_R8G8B8A8_UNORM, mData.data(), true);
#else
        mImage = std::make_shared<Image>(application->getApplicationContext(), 720, 720, VK_FORMAT_R8G8B8A8_UNORM, mData.data());
#endif
        mRandom.set(RandGen(mSeed));
#if defined(SAF_SCRIPTING) && defined(SAF_FILE_WATCH)
        loadScript(
            "TestScript",
            Application::stringToPath("./examples/scripts/test.lua"), [this](sol::state& state)
            { state.open_libraries(sol::lib::math); state["seed"] = &mSeed; state["update"] = [this] { this->updateLayer(); }; },
            [](sol::state&) {}, [](const Str& msg)
            { UILog::get().println("[Script] %s", msg.c_str()); });
#endif
    }

    inline void updateLayer()
    {
        mUpdate = true;
    }

    virtual void onDetach() override
    {
    }

    virtual void onUpdate(Application* application, float dt) override
    {
#ifdef SAF_CUDA_INTEROP
        mImage->awaitCudaUpdateClearance();
#endif
        if (mUpdate)
        {
            mUpdate = false;

            mRandom.set(RandGen(mSeed));

#ifndef SAF_DEBUG
#pragma omp parallel for
#endif
            for (I32 i = 0; i < mData.size(); ++i)
            {
                mData[i] = Eigen::Vector4<Byte>(
                    static_cast<Byte>(mRandom.get().rand() * 255),
                    static_cast<Byte>(mRandom.get().rand() * 255),
                    static_cast<Byte>(mRandom.get().rand() * 255),
                    255);
            }

            mImage->update(720, 720, VK_FORMAT_R8G8B8A8_UNORM, mData.data());

#ifdef SAF_CUDA_INTEROP
            callGrayScaleKernel(mImage->getCudaSurfaceObject(), mImage->getCudaTextureObject(), mImage->getWidth(), mImage->getHeight());
#endif
        }
#ifdef SAF_CUDA_INTEROP
        mImage->signalVulkanUpdateClearance();
#endif
    }

    virtual void onUIRender(Application* application) override
    {
        ImGui::Begin("Random");
        F64 fr = static_cast<F64>(ImGui::GetIO().Framerate);
        ImGui::Text("Average %.3f ms/frame", 1000.0 / fr);
        ImGui::Spacing();
        ImGui::Image(mImage->getDescriptorSet(), ImVec2(static_cast<F32>(mImage->getWidth()), static_cast<F32>(mImage->getHeight())));
        mUpdate |= mSeed.onUIRender();

        mRandom.onUIRender();

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Spacing();

        mTest.onUIRender();

        static const char* labels[] = { "-.-", ":D", ":)", ":(", "^^" };
        static F32 data[]           = { 0.1, 0.4, 0.2, 0.1, 0.2 };

        if (ImPlot::BeginPlot("##Pie2", ImVec2(350, 250), ImPlotFlags_Equal | ImPlotFlags_NoMouseText))
        {
            ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
            ImPlot::SetupAxesLimits(0, 1, 0, 1);
            ImPlot::PlotPieChart(labels, data, 5, 0.65, 0.5, 0.3, "%.1f", 180);
            ImPlot::EndPlot();
        }

        ImGui::End();

#ifdef SAF_SCRIPTING
        ImGui::Begin("Scripting");
        application->uiRenderActiveScripts();
        ImGui::End();
#endif

        UILog::get().render("Log");
    }

private:
    bool mUpdate = true;
    std::shared_ptr<Image> mImage;
    std::vector<Eigen::Vector4<Byte>> mData;

    RandGenParameter mRandom = RandGenParameter("RandGen", RandGen(UVec2(0, 0)));

    UVec2Parameter mSeed = UVec2Parameter("Noise Seed", UVec2(19, 97), 0, UINT8_MAX);

    TestParameter mTest = TestParameter("Test", Test());
};

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    ApplicationSettings settings;
    settings.windowWidth  = 1920;
    settings.windowHeight = 1080;
    settings.fontSize     = 24.0f;
    settings.name         = "Demo";
    settings.clearColor   = Vec4(0.3f, 0.3f, 0.3f, 1.0f);

    Application app(settings);

    app.setMenubarCallback([&app]()
                           {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Exit"))
            {
                app.close();
            }
            ImGui::EndMenu();
        } });

    app.pushLayer<DemoLayer>();

    app.run();
}
