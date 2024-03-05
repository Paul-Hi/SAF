#include "random.hpp"
#include "test.hpp"
#include <app/application.hpp>
#include <app/parameter.hpp>
#include <core/image.hpp>
#include <imgui.h>
#include <ui/imguiBackend.hpp>
#include <ui/log.hpp>

using namespace saf;

class DemoLayer : public Layer
{
public:
    virtual void onAttach(Application* application) override
    {
        mData.resize(720 * 720, Eigen::Vector4<Byte>(255, 0, 0, 255));
        mImage = std::make_shared<Image>(application->getPhysicalDevice(), application->getDevice(), application->getQueue(), application->getCommandPool(), application->getCommandBuffer(), 720, 720, VK_FORMAT_R8G8B8A8_UNORM, mData.data());
    }

    virtual void onDetach() override
    {
    }

    virtual void onUpdate(Application* application) override
    {
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

            mImage->update(application->getPhysicalDevice(), application->getDevice(), application->getQueue(), application->getCommandPool(), application->getCommandBuffer(), 720, 720, VK_FORMAT_R8G8B8A8_UNORM, mData.data());
        }
    }

    virtual void onUIRender() override
    {
        ImGui::Begin("Random");
        F64 fr = static_cast<F64>(ImGui::GetIO().Framerate);
        ImGui::Text("Average %.3f ms/frame", 1000.0 / fr);
        ImGui::Spacing();
        ImGui::Image(mImage->getDescriptorSet(), ImVec2(static_cast<F32>(mImage->getWidth()), static_cast<F32>(mImage->getHeight())));
        mUpdate = mSeed.onUIRender();

        mRandom.onUIRender();

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Spacing();

        mTest.onUIRender();

        ImGui::End();

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
    settings.fontSize     = 20.0f;
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
