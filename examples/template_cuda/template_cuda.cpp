#include "template_cuda.cuh"
#include <app/application.hpp>
#include <app/parameter.hpp>
#include <core/helpers.hpp>
#include <imgui.h>
#include <ui/gpuTimings.hpp>
#include <ui/log.hpp>

using namespace saf;

constexpr U32 imageSize = 8192;

class TemplateLayer : public Layer
{
public:
    // Creates the image once. CUDA writes it through its surface object.
    virtual void onAttach(Application* application) override
    {
        const VulkanImageCreateInfo imageCreateInfo{
            .width   = imageSize,
            .height  = imageSize,
            .format  = VK_FORMAT_R8G8B8A8_UNORM,
            .usage   = VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharing = ImageSharing::Cuda
        };
        VK_CHECK(application->createImage(imageCreateInfo, mImage));
    }

    // Launches this frame's kernel on the default stream. SAF makes that stream wait for the previous frame's sampling
    // before onUpdate, and the frame wait for the stream before ImGui samples the image.
    virtual void onUpdate(Application* application, F32 dt) override
    {
        mTime += dt * mSpeed;

        application->beginTiming(cudaStream_t{}, "Template");
        callVoronoiKernel(application->getCudaSurface(mImage), imageSize, imageSize, mTime, mMaxDistance, mSeed);
        application->endTiming(cudaStream_t{}, "Template");
    }

    virtual void onUIRender(Application* application) override
    {
        ImGui::Begin("Settings");
        ImGui::Text("Average %.3f ms/frame", 1000.0 / static_cast<F64>(ImGui::GetIO().Framerate));
        ImGui::Spacing();
        mSpeed.onUIRender();
        mMaxDistance.onUIRender();
        mSeed.onUIRender();
        ImGui::End();

        if (ImGui::Begin("Main Viewport"))
        {
            if (ImPlot::BeginPlot("##Image", ImVec2(-1, -1), ImPlotFlags_Equal))
            {
                ImVec2 bmin(-imageSize * 0.5, -imageSize * 0.5);
                ImVec2 bmax(imageSize * 0.5, imageSize * 0.5);
                ImPlot::PlotImage("Output", application->getTextureId(mImage), bmin, bmax);
                ImPlot::EndPlot();
            }
            ImGui::End();
        }

        UIGpuTimings::render(*application, "GPU Timings");
        UILog::get().render("Log");
    }

private:
    ImageHandle mImage;
    F32 mTime = 0.0f;

    F32Parameter mSpeed       = F32Parameter("Speed", 1.0f, 0.0f, 5.0f, "%.2f");
    F32Parameter mMaxDistance = F32Parameter("Max. Distance", 8.0f, 2.0f, 32.0f, "%.2f");
    U32Parameter mSeed        = U32Parameter("Random Seed", 1);
};

I32 main()
{
    ApplicationSettings settings;
    settings.windowWidth  = 1600;
    settings.windowHeight = 1000;
    settings.name         = "Template CUDA";
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

    app.pushLayer<TemplateLayer>();
    app.run();
}
