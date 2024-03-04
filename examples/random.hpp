
#include <app/parameter.hpp>
#include <core/types.hpp>

namespace saf
{
    class RandGen
    {
    public:
        RandGen(UVec2 seed = UVec2(0, 0))
        {
            U64 solitMix64State = (U64(seed.x()) << 32) | U64(seed.y());
            U64 s0              = splitMix64Rand(solitMix64State);
            U64 s1              = splitMix64Rand(solitMix64State);
            mState.set({ U32(s0), U32(s0 >> 32), U32(s1), U32(s1 >> 32) });
        }

        F32 rand()
        {
            // Upper 24 bits + divide by 2^24 to get a random number in [0,1).
            U32 bits = nextU32();
            return (bits >> 8) * 0x1p-24;
        }

        Vec2 rand2()
        {
            Vec2 result;
            result.x() = rand();
            result.y() = rand();
            return result;
        }

    private:
        // https://prng.di.unimi.it/xoshiro128starstar.c
        // This is a Xoshiro128StarStar initialized with a SplitMix64 similarly executed to Nvidias Falcor
        UVec4Parameter mState = UVec4Parameter("State", UVec4(0.0, 0.0, 0.0, 0.0));

        inline U64 splitMix64Rand(U64 state)
        {
            U64 z = (state += 0x9E3779B97F4A7C15ull);
            z     = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z     = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        }

        inline U32 rotl(const U32 x, int k)
        {
            return (x << k) | (x >> (32 - k));
        }

        U32 nextU32()
        {
            UVec4& state     = mState.get();
            const U32 result = rotl(state.y() * 5, 7) * 9;

            const U32 t = state.y() << 9;

            state.z() ^= state.x();
            state.w() ^= state.y();
            state.y() ^= state.z();
            state.x() ^= state.w();

            state.z() ^= t;

            state.w() = rotl(state.w(), 11);

            return result;
        }

        friend class RandGenParameter;
    };

    class RandGenParameter : public ParameterBlock<RandGen, UVec4Parameter>
    {
    public:
        RandGenParameter(const Str& name, const RandGen& randgen)
            : ParameterBlock<RandGen, UVec4Parameter>(name, randgen){

            };

        ~RandGenParameter() = default;

        bool onUIRender() override
        {
            ImGui::BeginDisabled(true);
            mValue.mState.onUIRender();
            ImGui::EndDisabled();

            return false;
        }

        std::tuple<UVec4Parameter*> getParameters() override
        {
            return { &mValue.mState };
        }
    };

} // namespace saf
