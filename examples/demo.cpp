#include <app/application.hpp>
#include <core/image.hpp>
#include <core/vulkanHelper.hpp>
#include <imgui.h>

using namespace saf;

class DemoLayer : public Layer
{
public:
    virtual void onAttach(Application *application) override
    {
        mData.resize(512 * 512, Eigen::Vector4<Byte>(255, 0, 0, 255));
        mImage = std::make_shared<Image>(application->getPhysicalDevice(), application->getDevice(), application->getQueue(), application->getCommandPool(), application->getCommandBuffer(), 512, 512, VK_FORMAT_R8G8B8A8_UNORM, mData.data());
    }

    virtual void onDetach() override
    {
    }

    U32 wang(U32 v)
    {
        v = (v ^ 61u) ^ (v >> 16u);
        v *= 9u;
        v ^= v >> 4u;
        v *= 0x27d4eb2du;
        v ^= v >> 15u;
        return v;
    }

    virtual void onUpdate(Application *application) override
    {
        if (mUpdate)
        {
            mUpdate = false;

#ifndef SAF_DEBUG
#pragma omp parallel for
#endif
            for (PtrSize i = 0; i < mData.size(); ++i)
            {
                mData[i] = Eigen::Vector4<Byte>(
                    static_cast<Byte>(static_cast<F32>(wang(static_cast<U32>(mFrame * i)) / RAND_MAX) * 255),
                    static_cast<Byte>(static_cast<F32>(wang(static_cast<U32>(mFrame * i + mData.size())) / RAND_MAX) * 255),
                    static_cast<Byte>(static_cast<F32>(wang(static_cast<U32>(mFrame * i + 2 * mData.size())) / RAND_MAX) * 255),
                    255);
            }

            mImage->update(application->getPhysicalDevice(), application->getDevice(), application->getQueue(), application->getCommandPool(), application->getCommandBuffer(), 512, 512, VK_FORMAT_R8G8B8A8_UNORM, mData.data());
        }
    }

    virtual void onUIRender() override
    {
        ImGui::Begin("WangHash", nullptr);
        F64 fr = static_cast<F64>(ImGui::GetIO().Framerate);
        ImGui::Text("Average %.3f ms/frame", 1000.0 / fr);
        ImGui::Spacing();
        ImGui::Image(mImage->getDescriptorSet(), ImVec2(static_cast<F32>(mImage->getWidth()), static_cast<F32>(mImage->getHeight())));
        mUpdate = ImGui::SliderInt("Noise Frame", reinterpret_cast<I32 *>(&mFrame), 1, 255);
        ImGui::End();
    }

private:
    bool mUpdate = true;
    U32 mFrame = 1;
    std::shared_ptr<Image> mImage;
    std::vector<Eigen::Vector4<Byte>> mData;
};

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    ApplicationSettings settings;
    settings.windowWidth = 1920;
    settings.windowHeight = 1080;
    settings.name = "Demo";
    settings.clearColor = Vec4(0.3f, 0.3f, 0.3f, 1.0f);

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
