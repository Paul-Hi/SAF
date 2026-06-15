#include "demo.cuh"
#include "random.hpp"
#include "test.hpp"
#include <app/application.hpp>
#include <app/parameter.hpp>
#include <core/helpers.hpp>
#include <core/types.hpp>
#include <imgui.h>
#include <implot.h>
#include <implot3d.h>
#include <ui/imguiBackend.hpp>
#include <ui/log.hpp>

using namespace saf;

class DemoLayer : public Layer
{
public:
    virtual void onAttach(Application* application) override
    {
        mData.resize(720 * 720, Eigen::Vector4<Byte>(0, 0, 0, 255));
#ifdef SAF_CUDA_INTEROP
        const VulkanImageCreateInfo imageCreateInfo{
            .width         = 720,
            .height        = 720,
            .format        = VK_FORMAT_R8G8B8A8_UNORM,
            .usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .shareWithCUDA = true
        };

        VK_CHECK(application->createImage(imageCreateInfo, mImage));
        VK_CHECK(application->uploadImage(mImage, mData.data(), mData.size() * sizeof(Eigen::Vector4<Byte>)));
#else
        const VulkanImageCreateInfo imageCreateInfo{
            .width  = 720,
            .height = 720,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .usage  = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        };

        VK_CHECK(application->createImage(imageCreateInfo, mImage));
        VK_CHECK(application->uploadImage(mImage, mData.data(), mData.size() * sizeof(Eigen::Vector4<Byte>)));
#endif
        mRandom.set(RandGen(mSeed));
    }

    inline void updateLayer()
    {
        mUpdate = true;
    }

    virtual void onDetach(Application* application) override
    {
        (void)application;
    }

    virtual void onUpdate(Application* application, F32 dt) override
    {
        (void)dt;
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

            application->uploadImage(mImage, mData.data(), mData.size() * sizeof(Eigen::Vector4<Byte>));

#ifdef SAF_CUDA_INTEROP
            const VulkanImage& vulkanImage = application->getImage(mImage);
            callGrayScaleKernel(vulkanImage.cudaSurfaceObject, vulkanImage.cudaTextureObject, vulkanImage.width, vulkanImage.height);
#endif
        }
    }

    virtual void onUIRender(Application* application) override
    {
        ImGui::Begin("Random");
        F64 fr = static_cast<F64>(ImGui::GetIO().Framerate);
        ImGui::Text("Average %.3f ms/frame", 1000.0 / fr);
        ImGui::Spacing();
        const VulkanImage& vulkanImage = application->getImage(mImage);
        ImGui::Image(reinterpret_cast<ImTextureID>(vulkanImage.descriptorSet), ImVec2(static_cast<F32>(vulkanImage.width), static_cast<F32>(vulkanImage.height)));
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

        if (ImPlot::BeginPlot("##Pie2", ImVec2(350, 350), ImPlotFlags_Equal | ImPlotFlags_NoMouseText))
        {
            ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_NoDecorations);
            ImPlot::SetupAxesLimits(0, 1, 0, 1);
            ImPlot::PlotPieChart(labels, data, 5, 0.65, 0.5, 0.3, "%.1f", 180);
            ImPlot::EndPlot();
        }

        // Some 3D surface data - similar to the 3D surface demo of ImPlot3D
        constexpr I32 N = 40;
        static F32 xs[N * N], ys[N * N], zs[N * N];
        static F32 t = 0.0f;
        t += ImGui::GetIO().DeltaTime;

        constexpr F32 minVal = -10.0f;
        constexpr F32 maxVal = 10.0f;
        constexpr F32 step   = (maxVal - minVal) / (N - 1);

        for (I32 i = 0; i < N; i++)
        {
            for (I32 j = 0; j < N; j++)
            {
                I32 idx = i * N + j;
                xs[idx] = minVal + j * step; // X values are constant along rows
                ys[idx] = minVal + i * step; // Y values are constant along columns
                zs[idx] = 0.5f * sin(-4 * t + sqrt((xs[idx] * xs[idx] + ys[idx] * ys[idx])));
            }
        }

        ImGui::SameLine();

        ImPlot3D::PushColormap("Viridis");
        if (ImPlot3D::BeginPlot("##SurfacePlot", ImVec2(350, 350), ImPlot3DFlags_NoClip))
        {
            ImPlot3D::SetupAxesLimits(-10, 10, -10, 10, -1.5, 1.5);
            ImPlot3D::PushStyleVar(ImPlot3DStyleVar_FillAlpha, 0.8f);
            ImPlot3DSpec spec;
            spec.LineColor = ImPlot3D::GetColormapColor(1);

            ImPlot3D::PlotSurface("Wavy Surface", xs, ys, zs, N, N, 0.0, 0.0, spec);

            ImPlot3D::PopStyleVar();
            ImPlot3D::EndPlot();
        }
        ImPlot3D::PopColormap();

        ImGui::End();

        UILog::get().render("Log");
    }

private:
    bool mUpdate = true;
    ImageHandle mImage;
    std::vector<Eigen::Vector4<Byte>> mData;

    RandGenParameter mRandom = RandGenParameter("RandGen", RandGen(UVec2(0, 0)));

    UVec2Parameter mSeed = UVec2Parameter("Noise Seed", UVec2(19, 97), 0, UINT8_MAX);

    TestParameter mTest = TestParameter("Test", Test());
};

I32 main(I32 argc, char** argv)
{
    (void)argc;
    (void)argv;

    ApplicationSettings settings;
    settings.windowWidth  = 1920;
    settings.windowHeight = 1080;
    settings.fontScale    = 1.0f;
    settings.name         = "Demo";
    settings.theme        = 0;
    settings.clearColor   = Vec4(0.3f, 0.3f, 0.3f, 1.0f);
    settings.vSyncEnabled = true;

    Application app(settings);

    app.setMenubarCallback([&app]()
                           {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Test"))
            {
                UILog::get().println("Test menu item clicked!");
            }
            ImGui::EndMenu();
        } });

    app.pushLayer<DemoLayer>();

    app.run();
}
