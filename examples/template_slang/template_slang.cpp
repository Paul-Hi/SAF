#include <app/application.hpp>
#include <app/parameter.hpp>
#include <core/helpers.hpp>
#include <imgui.h>
#include <ui/gpuTimings.hpp>
#include <ui/log.hpp>

using namespace saf;

constexpr U32 imageSize = 8192;

// RGBA colors the shader picks from.
constexpr F32 palette[][4] = {
    { 0.06f, 0.10f, 0.23f, 1.0f },
    { 0.09f, 0.19f, 0.39f, 1.0f },
    { 0.12f, 0.29f, 0.55f, 1.0f },
    { 0.17f, 0.47f, 0.68f, 1.0f },
    { 0.23f, 0.64f, 0.79f, 1.0f },
    { 0.47f, 0.78f, 0.86f, 1.0f },
    { 0.72f, 0.89f, 0.91f, 1.0f },
    { 0.98f, 0.88f, 0.62f, 1.0f },
    { 0.95f, 0.72f, 0.35f, 1.0f },
    { 0.85f, 0.35f, 0.18f, 1.0f },
    { 0.60f, 0.16f, 0.12f, 1.0f }
};

struct PushConstants
{
    F32 time;
    F32 maxDistance;
    U32 seed;
    U32 padding;
};

class TemplateLayer : public Layer
{
public:
    // Creates the GPU resources once. Bindings stay until rebound, so they are set here, not every frame.
    virtual void onAttach(Application* application) override
    {
        std::vector<U32> spirv;
        VK_CHECK(loadSpirv(shaderPath("template_slang.slang.spv"), spirv));
        VK_CHECK(application->createComputePipeline({ .spirv = spirv }, mPipeline));

        BufferHandle paletteBuffer;
        VK_CHECK(application->createBuffer({ .size = sizeof(palette) }, paletteBuffer));
        VK_CHECK(application->uploadBuffer(paletteBuffer, palette, sizeof(palette)));

        const VulkanImageCreateInfo imageCreateInfo{
            .width   = imageSize,
            .height  = imageSize,
            .format  = VK_FORMAT_R8G8B8A8_UNORM,
            .usage   = VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharing = ImageSharing::Compute
        };
        VK_CHECK(application->createImage(imageCreateInfo, mImage));

        application->bind(mPipeline, "palette", paletteBuffer);
        application->bind(mPipeline, "outImage", mImage);
    }

    // Records this frame's GPU work. SAF submits it after onUpdate, and the frame waits for it before ImGui samples
    // the image. Dependent dispatches within a frame need a computeBarrier() between them.
    virtual void onUpdate(Application* application, F32 dt) override
    {
        mTime += dt * mSpeed;
        application->setPushConstants(mPipeline, PushConstants{ .time = mTime, .maxDistance = mMaxDistance, .seed = mSeed });

        application->beginTiming("Template");
        application->dispatch(mPipeline, (imageSize + 15) / 16, (imageSize + 15) / 16, 1);
        application->endTiming("Template");
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
    ComputePipelineHandle mPipeline;
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
    settings.name         = "Template Slang";
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
