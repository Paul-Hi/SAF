/**
 * @file      parameter.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2025
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef PARAMETER_HPP
#define PARAMETER_HPP

#include <imgui.h>

#ifdef SAF_SCRIPTING
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>
#endif
namespace saf
{
    template <typename T>
    class Parameter
    {
    public:
        virtual ~Parameter() = default;

        const Str& name() const { return mName; }

        T& get() { return mValue; }

        const T& get() const { return mValue; }

        operator T&() { return mValue; }

        operator const T&() const { return mValue; }

        void set(const T& value) { mValue = value; }

        virtual bool onUIRender() = 0;

    protected:
        Parameter(const Str& name, const T& value)
            : mName(name)
            , mValue(value)
        {
        }

        Str mName;
        T mValue;
    };
    template <typename T, typename... M>
    class ParameterBlock : public Parameter<T>
    {
    public:
        virtual ~ParameterBlock() = default;

        constexpr U32 getParameterCount() const
        {
            return sizeof...(M);
        }

        virtual std::tuple<M*...> getParameters() = 0;

        template <typename FuncT>
        void forEachParameter(FuncT f)
        {
            std::apply([f](auto&&... args)
                       { ((f(args)), ...); },
                       getParameters());
        }

        bool onUIRender() override
        {
            bool changed = false;

            if (ImGui::TreeNode(this->mName.c_str()))
            {
                ImGui::Spacing();
                forEachParameter([&changed](auto&& p)
                                 { changed |= p->onUIRender(); });

                ImGui::TreePop();
            }

            return changed;
        }

    protected:
        ParameterBlock(const Str& name, const T& value)
            : Parameter<T>(name, value)
        {
        }
    };

    class ByteParameter : public Parameter<Byte>
    {
    public:
        ByteParameter(const Str& name, const Byte& value, const Byte& min, const Byte& max, const char* format = (const char*)0)
            : Parameter<Byte>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~ByteParameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_U8, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        Byte mMin;
        Byte mMax;
        const char* mFormat = (const char*)0;
    };

    class I16Parameter : public Parameter<I16>
    {
    public:
        I16Parameter(const Str& name, const I16& value, const I16& min, const I16& max, const char* format = (const char*)0)
            : Parameter<I16>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~I16Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_S16, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        I16 mMin;
        I16 mMax;
        const char* mFormat = (const char*)0;
    };

    class I32Parameter : public Parameter<I32>
    {
    public:
        I32Parameter(const Str& name, const I32& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<I32>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~I32Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        I32 mMin;
        I32 mMax;
        const char* mFormat = (const char*)0;
    };

    class I64Parameter : public Parameter<I64>
    {
    public:
        I64Parameter(const Str& name, const I64& value, const I64& min, const I64& max, const char* format = (const char*)0)
            : Parameter<I64>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~I64Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_S64, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        I64 mMin;
        I64 mMax;
        const char* mFormat = (const char*)0;
    };

    class U16Parameter : public Parameter<U16>
    {
    public:
        U16Parameter(const Str& name, const U16& value, const U16& min, const U16& max, const char* format = (const char*)0)
            : Parameter<U16>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~U16Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_U16, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        U16 mMin;
        U16 mMax;
        const char* mFormat = (const char*)0;
    };

    class U32Parameter : public Parameter<U32>
    {
    public:
        U32Parameter(const Str& name, const U32& value, const U32& min = 0, const U32& max = UINT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<U32>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~U32Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        U32 mMin;
        U32 mMax;
        const char* mFormat = (const char*)0;
    };

    class U64Parameter : public Parameter<U64>
    {
    public:
        U64Parameter(const Str& name, const U64& value, const U64& min, const U64& max, const char* format = (const char*)0)
            : Parameter<U64>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~U64Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_U64, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        U64 mMin;
        U64 mMax;
        const char* mFormat = (const char*)0;
    };

    class F32Parameter : public Parameter<F32>
    {
    public:
        F32Parameter(const Str& name, const F32& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2, const char* format = (const char*)0)
            : Parameter<F32>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~F32Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        F32 mMin;
        F32 mMax;
        const char* mFormat = (const char*)0;
    };

    class F64Parameter : public Parameter<F64>
    {
    public:
        F64Parameter(const Str& name, const F64& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2, const char* format = (const char*)0)
            : Parameter<F64>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~F64Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        F64 mMin;
        F64 mMax;
        const char* mFormat = (const char*)0;
    };

    class Vec2Parameter : public Parameter<Vec2>
    {
    public:
        Vec2Parameter(const Str& name, const Vec2& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2, const char* format = (const char*)0)
            : Parameter<Vec2>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~Vec2Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(mValue.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        F32 mMin;
        F32 mMax;
        const char* mFormat = (const char*)0;
    };

    class Vec3Parameter : public Parameter<Vec3>
    {
    public:
        Vec3Parameter(const Str& name, const Vec3& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2, const char* format = (const char*)0)
            : Parameter<Vec3>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~Vec3Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(mValue.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        F32 mMin;
        F32 mMax;
        const char* mFormat = (const char*)0;
    };

    class Vec4Parameter : public Parameter<Vec4>
    {
    public:
        Vec4Parameter(const Str& name, const Vec4& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2, const char* format = (const char*)0)
            : Parameter<Vec4>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~Vec4Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(mValue.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        F32 mMin;
        F32 mMax;
        const char* mFormat = (const char*)0;
    };

    class Mat2Parameter : public Parameter<Mat2>
    {
    public:
        Mat2Parameter(const Str& name, const Mat2& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2, const char* format = (const char*)0)
            : Parameter<Mat2>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~Mat2Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<F32, 2, 2, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data() + 2), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            mValue                                       = rM;
            return row0 || row1;
        }

    private:
        F32 mMin;
        F32 mMax;
        const char* mFormat = (const char*)0;
    };

    class Mat3Parameter : public Parameter<Mat3>
    {
    public:
        Mat3Parameter(const Str& name, const Mat3& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2, const char* format = (const char*)0)
            : Parameter<Mat3>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~Mat3Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<F32, 3, 3, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data() + 3), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data() + 6), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            mValue                                       = rM;
            return row0 || row1 || row2;
        }

    private:
        F32 mMin;
        F32 mMax;
        const char* mFormat = (const char*)0;
    };

    class Mat4Parameter : public Parameter<Mat4>
    {
    public:
        Mat4Parameter(const Str& name, const Mat4& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2, const char* format = (const char*)0)
            : Parameter<Mat4>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~Mat4Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<F32, 4, 4, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data() + 4), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data() + 8), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row3                                    = ImGui::SliderScalarN((mName + "3").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data() + 12), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            mValue                                       = rM;
            return row0 || row1 || row2 || row3;
        }

    private:
        F32 mMin;
        F32 mMax;
        const char* mFormat = (const char*)0;
    };

    class QuatParameter : public Parameter<Quat>
    {
    public:
        QuatParameter(const Str& name, const Quat& value, const char* format = (const char*)0)
            : Parameter<Quat>(name, value)
            , mFormat(format)
        {
        }

        ~QuatParameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(mValue.coeffs().data()), 4, nullptr, nullptr, mFormat);
        }

    private:
        const char* mFormat = (const char*)0;
    };

    class DVec2Parameter : public Parameter<DVec2>
    {
    public:
        DVec2Parameter(const Str& name, const DVec2& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2, const char* format = (const char*)0)
            : Parameter<DVec2>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~DVec2Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(mValue.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        F64 mMin;
        F64 mMax;
        const char* mFormat = (const char*)0;
    };

    class DVec3Parameter : public Parameter<DVec3>
    {
    public:
        DVec3Parameter(const Str& name, const DVec3& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2, const char* format = (const char*)0)
            : Parameter<DVec3>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~DVec3Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(mValue.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        F64 mMin;
        F64 mMax;
        const char* mFormat = (const char*)0;
    };

    class DVec4Parameter : public Parameter<DVec4>
    {
    public:
        DVec4Parameter(const Str& name, const DVec4& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2, const char* format = (const char*)0)
            : Parameter<DVec4>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~DVec4Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(mValue.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        F64 mMin;
        F64 mMax;
        const char* mFormat = (const char*)0;
    };

    class DMat2Parameter : public Parameter<DMat2>
    {
    public:
        DMat2Parameter(const Str& name, const DMat2& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2, const char* format = (const char*)0)
            : Parameter<DMat2>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~DMat2Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<F64, 2, 2, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data() + 2), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            mValue                                       = rM;
            return row0 || row1;
        }

    private:
        F64 mMin;
        F64 mMax;
        const char* mFormat = (const char*)0;
    };

    class DMat3Parameter : public Parameter<DMat3>
    {
    public:
        DMat3Parameter(const Str& name, const DMat3& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2, const char* format = (const char*)0)
            : Parameter<DMat3>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~DMat3Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<F64, 3, 3, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data() + 3), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data() + 6), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            mValue                                       = rM;
            return row0 || row1 || row2;
        }

    private:
        F64 mMin;
        F64 mMax;
        const char* mFormat = (const char*)0;
    };

    class DMat4Parameter : public Parameter<DMat4>
    {
    public:
        DMat4Parameter(const Str& name, const DMat4& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2, const char* format = (const char*)0)
            : Parameter<DMat4>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~DMat4Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<F64, 4, 4, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data() + 4), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data() + 8), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row3                                    = ImGui::SliderScalarN((mName + "3").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data() + 12), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            mValue                                       = rM;
            return row0 || row1 || row2 || row3;
        }

    private:
        F64 mMin;
        F64 mMax;
        const char* mFormat = (const char*)0;
    };

    class DQuatParameter : public Parameter<DQuat>
    {
    public:
        DQuatParameter(const Str& name, const DQuat& value, const char* format = (const char*)0)
            : Parameter<DQuat>(name, value)
            , mFormat(format)
        {
        }

        ~DQuatParameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(mValue.coeffs().data()), 4, nullptr, nullptr, mFormat);
        }

    private:
        const char* mFormat = (const char*)0;
    };

    class IVec2Parameter : public Parameter<IVec2>
    {
    public:
        IVec2Parameter(const Str& name, const IVec2& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<IVec2>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~IVec2Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(mValue.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        I32 mMin;
        I32 mMax;
        const char* mFormat = (const char*)0;
    };

    class IVec3Parameter : public Parameter<IVec3>
    {
    public:
        IVec3Parameter(const Str& name, const IVec3& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<IVec3>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~IVec3Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(mValue.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        I32 mMin;
        I32 mMax;
        const char* mFormat = (const char*)0;
    };

    class IVec4Parameter : public Parameter<IVec4>
    {
    public:
        IVec4Parameter(const Str& name, const IVec4& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<IVec4>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~IVec4Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(mValue.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        I32 mMin;
        I32 mMax;
        const char* mFormat = (const char*)0;
    };

    class IMat2Parameter : public Parameter<IMat2>
    {
    public:
        IMat2Parameter(const Str& name, const IMat2& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<IMat2>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~IMat2Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<I32, 2, 2, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data() + 2), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            mValue                                       = rM;
            return row0 || row1;
        }

    private:
        I32 mMin;
        I32 mMax;
        const char* mFormat = (const char*)0;
    };

    class IMat3Parameter : public Parameter<IMat3>
    {
    public:
        IMat3Parameter(const Str& name, const IMat3& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<IMat3>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~IMat3Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<I32, 3, 3, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data() + 3), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data() + 6), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            mValue                                       = rM;
            return row0 || row1 || row2;
        }

    private:
        I32 mMin;
        I32 mMax;
        const char* mFormat = (const char*)0;
    };

    class IMat4Parameter : public Parameter<IMat4>
    {
    public:
        IMat4Parameter(const Str& name, const IMat4& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<IMat4>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~IMat4Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<I32, 4, 4, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data() + 4), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data() + 8), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row3                                    = ImGui::SliderScalarN((mName + "3").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data() + 12), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            mValue                                       = rM;
            return row0 || row1 || row2 || row3;
        }

    private:
        I32 mMin;
        I32 mMax;
        const char* mFormat = (const char*)0;
    };

    class UVec2Parameter : public Parameter<UVec2>
    {
    public:
        UVec2Parameter(const Str& name, const UVec2& value, const U32& min = 0, const U32& max = UINT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<UVec2>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~UVec2Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(mValue.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        U32 mMin;
        U32 mMax;
        const char* mFormat = (const char*)0;
    };

    class UVec3Parameter : public Parameter<UVec3>
    {
    public:
        UVec3Parameter(const Str& name, const UVec3& value, const U32& min = 0, const U32& max = UINT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<UVec3>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~UVec3Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(mValue.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        U32 mMin;
        U32 mMax;
        const char* mFormat = (const char*)0;
    };

    class UVec4Parameter : public Parameter<UVec4>
    {
    public:
        UVec4Parameter(const Str& name, const UVec4& value, const U32& min = 0, const U32& max = UINT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<UVec4>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~UVec4Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(mValue.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
        }

    private:
        U32 mMin;
        U32 mMax;
        const char* mFormat = (const char*)0;
    };

    class UMat2Parameter : public Parameter<UMat2>
    {
    public:
        UMat2Parameter(const Str& name, const UMat2& value, const U32& min = 0, const U32& max = UINT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<UMat2>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~UMat2Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<U32, 2, 2, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data() + 2), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            mValue                                       = rM;
            return row0 || row1;
        }

    private:
        U32 mMin;
        U32 mMax;
        const char* mFormat = (const char*)0;
    };

    class UMat3Parameter : public Parameter<UMat3>
    {
    public:
        UMat3Parameter(const Str& name, const UMat3& value, const U32& min = 0, const U32& max = UINT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<UMat3>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~UMat3Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<U32, 3, 3, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data() + 3), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data() + 6), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            mValue                                       = rM;
            return row0 || row1 || row2;
        }

    private:
        U32 mMin;
        U32 mMax;
        const char* mFormat = (const char*)0;
    };

    class UMat4Parameter : public Parameter<UMat4>
    {
    public:
        UMat4Parameter(const Str& name, const UMat4& value, const U32& min = 0, const U32& max = UINT32_MAX / 2, const char* format = (const char*)0)
            : Parameter<UMat4>(name, value)
            , mMin(min)
            , mMax(max)
            , mFormat(format)
        {
        }

        ~UMat4Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<U32, 4, 4, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data() + 4), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data() + 8), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            bool row3                                    = ImGui::SliderScalarN((mName + "3").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data() + 12), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax), mFormat);
            mValue                                       = rM;
            return row0 || row1 || row2 || row3;
        }

    private:
        U32 mMin;
        U32 mMax;
        const char* mFormat = (const char*)0;
    };

    class BoolParameter : public Parameter<bool>
    {
    public:
        BoolParameter(const Str& name, const bool& value)
            : Parameter<bool>(name, value)
        {
        }

        ~BoolParameter() = default;

        bool onUIRender() override
        {
            return ImGui::Checkbox(mName.c_str(), &mValue);
        }
    };

    class StrParameter : public Parameter<Str>
    {
    public:
        StrParameter(const Str& name, const Str& value, PtrSize maxBufferLength)
            : Parameter<Str>(name, value)
        {
            mBuf.resize(maxBufferLength);
        }

        ~StrParameter() = default;

        bool onUIRender() override
        {
            bool changed = ImGui::InputText(mName.c_str(), mBuf.data(), mBuf.size());

            if (changed)
            {
                mValue = Str(mBuf.data(), mBuf.size());
            }

            return changed;
        }

    private:
        std::vector<char> mBuf;
    };

#ifdef SAF_SCRIPTING
    void setupParametersInLuaState(sol::state& state);
#endif

} // namespace saf

#endif // PARAMETER_HPP
