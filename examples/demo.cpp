#include <app/application.hpp>
#include <app/parameter.hpp>
#include <core/image.hpp>
#include <imgui.h>

using namespace saf;

struct Random
{
    // https://prng.di.unimi.it/xoshiro128starstar.c
    // This is a Xoshiro128StarStar initialized with a SplitMix64 similarly executed to Nvidias Falcor
    static UVec4 gState;

    static inline U64 splitMix64Rand(U64 state)
    {
        U64 z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    static void init(UVec2 seed)
    {
        U64 solitMix64State = (U64(seed.x()) << 32) | U64(seed.y());
        U64 s0 = splitMix64Rand(solitMix64State);
        U64 s1 = splitMix64Rand(solitMix64State);
        gState = {U32(s0), U32(s0 >> 32), U32(s1), U32(s1 >> 32)};
    }

    static inline U32 rotl(const U32 x, int k)
    {
        return (x << k) | (x >> (32 - k));
    }

    static U32 nextU32()
    {
        const U32 result = rotl(gState.y() * 5, 7) * 9;

        const U32 t = gState.y() << 9;

        gState.z() ^= gState.x();
        gState.w() ^= gState.y();
        gState.y() ^= gState.z();
        gState.x() ^= gState.w();

        gState.z() ^= t;

        gState.w() = rotl(gState.w(), 11);

        return result;
    }

    static F32 rand()
    {
        // Upper 24 bits + divide by 2^24 to get a random number in [0,1).
        U32 bits = nextU32();
        return (bits >> 8) * 0x1p-24;
    }

    static Vec2 rand2()
    {
        Vec2 result;
        result.x() = rand();
        result.y() = rand();
        return result;
    }
};

UVec4 Random::gState = UVec4(0.0, 0.0, 0.0, 0.0);

class DemoLayer : public Layer
{
public:
    virtual void onAttach(Application *application) override
    {
        Random::init(mSeed);
        mData.resize(720 * 720, Eigen::Vector4<Byte>(255, 0, 0, 255));
        mImage = std::make_shared<Image>(application->getPhysicalDevice(), application->getDevice(), application->getQueue(), application->getCommandPool(), application->getCommandBuffer(), 720, 720, VK_FORMAT_R8G8B8A8_UNORM, mData.data());
    }

    virtual void onDetach() override
    {
    }

    virtual void onUpdate(Application *application) override
    {
        if (mUpdate)
        {
            mUpdate = false;

            Random::init(mSeed);

#ifndef SAF_DEBUG
#pragma omp parallel for
#endif
            for (I32 i = 0; i < mData.size(); ++i)
            {
                mData[i] = Eigen::Vector4<Byte>(
                    static_cast<Byte>(Random::rand() * 255),
                    static_cast<Byte>(Random::rand() * 255),
                    static_cast<Byte>(Random::rand() * 255),
                    255);
            }

            mImage->update(application->getPhysicalDevice(), application->getDevice(), application->getQueue(), application->getCommandPool(), application->getCommandBuffer(), 720, 720, VK_FORMAT_R8G8B8A8_UNORM, mData.data());
        }
    }

    virtual void onUIRender() override
    {
        ImGui::Begin("Random", nullptr);
        F64 fr = static_cast<F64>(ImGui::GetIO().Framerate);
        ImGui::Text("Average %.3f ms/frame", 1000.0 / fr);
        ImGui::Spacing();
        ImGui::Image(mImage->getDescriptorSet(), ImVec2(static_cast<F32>(mImage->getWidth()), static_cast<F32>(mImage->getHeight())));
        mUpdate = mSeed.onUIRender();
        ImGui::End();
    }

private:
    bool mUpdate = true;
    UVec2Parameter mSeed = UVec2Parameter("Noise Seed", UVec2(19, 97), 0, UINT8_MAX);
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
    settings.fontSize = 20.0f;
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
