
#include <app/parameter.hpp>

namespace saf
{
    class Test
    {
    public:
        Test()
            : mInteger("Integer Member", 10, -1, 1024)
            , mSize("Size", IVec3(0, 0, 0), -10, 10)
            , mPoint("Point", Vec3(0.0, 0.0, 0.0), -36.0, 36.0)
        {
        }

    private:
        I32Parameter mInteger;
        IVec3Parameter mSize;
        Vec3Parameter mPoint;

        friend class TestParameter;
    };

    class TestParameter : public ParameterBlock<Test, I32Parameter, IVec3Parameter, Vec3Parameter>
    {
    public:
        TestParameter(const Str& name, const Test& randgen)
            : ParameterBlock<Test, I32Parameter, IVec3Parameter, Vec3Parameter>(name, randgen){

            };

        ~TestParameter() = default;

        std::tuple<I32Parameter*, IVec3Parameter*, Vec3Parameter*> getParameters() override
        {
            return { &mValue.mInteger, &mValue.mSize, &mValue.mPoint };
        }
    };

} // namespace saf
