/**
 * @file      parameter.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2024
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef PARAMETER_HPP
#define PARAMETER_HPP

#include <imgui.h>
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>

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
        ByteParameter(const Str& name, const Byte& value, const Byte& min, const Byte& max)
            : Parameter<Byte>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~ByteParameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_U8, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        Byte mMin;
        Byte mMax;
    };

    class I16Parameter : public Parameter<I16>
    {
    public:
        I16Parameter(const Str& name, const I16& value, const I16& min, const I16& max)
            : Parameter<I16>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~I16Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_S16, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        I16 mMin;
        I16 mMax;
    };

    class I32Parameter : public Parameter<I32>
    {
    public:
        I32Parameter(const Str& name, const I32& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2)
            : Parameter<I32>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~I32Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        I32 mMin;
        I32 mMax;
    };

    class I64Parameter : public Parameter<I64>
    {
    public:
        I64Parameter(const Str& name, const I64& value, const I64& min, const I64& max)
            : Parameter<I64>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~I64Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_S64, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        I64 mMin;
        I64 mMax;
    };

    class U16Parameter : public Parameter<U16>
    {
    public:
        U16Parameter(const Str& name, const U16& value, const U16& min, const U16& max)
            : Parameter<U16>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~U16Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_U16, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        U16 mMin;
        U16 mMax;
    };

    class U32Parameter : public Parameter<U32>
    {
    public:
        U32Parameter(const Str& name, const U32& value, const U32& min = 0, const U32& max = UINT32_MAX / 2)
            : Parameter<U32>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~U32Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        U32 mMin;
        U32 mMax;
    };

    class U64Parameter : public Parameter<U64>
    {
    public:
        U64Parameter(const Str& name, const U64& value, const U64& min, const U64& max)
            : Parameter<U64>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~U64Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_U64, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        U64 mMin;
        U64 mMax;
    };

    class F32Parameter : public Parameter<F32>
    {
    public:
        F32Parameter(const Str& name, const F32& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2)
            : Parameter<F32>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~F32Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        F32 mMin;
        F32 mMax;
    };

    class F64Parameter : public Parameter<F64>
    {
    public:
        F64Parameter(const Str& name, const F64& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2)
            : Parameter<F64>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~F64Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalar(mName.c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(&mValue), reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        F64 mMin;
        F64 mMax;
    };

    class Vec2Parameter : public Parameter<Vec2>
    {
    public:
        Vec2Parameter(const Str& name, const Vec2& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2)
            : Parameter<Vec2>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~Vec2Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(mValue.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        F32 mMin;
        F32 mMax;
    };

    class Vec3Parameter : public Parameter<Vec3>
    {
    public:
        Vec3Parameter(const Str& name, const Vec3& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2)
            : Parameter<Vec3>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~Vec3Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(mValue.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        F32 mMin;
        F32 mMax;
    };

    class Vec4Parameter : public Parameter<Vec4>
    {
    public:
        Vec4Parameter(const Str& name, const Vec4& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2)
            : Parameter<Vec4>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~Vec4Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(mValue.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        F32 mMin;
        F32 mMax;
    };

    class Mat2Parameter : public Parameter<Mat2>
    {
    public:
        Mat2Parameter(const Str& name, const Mat2& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2)
            : Parameter<Mat2>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~Mat2Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<F32, 2, 2, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data() + 2), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            mValue                                       = rM;
            return row0 || row1;
        }

    private:
        F32 mMin;
        F32 mMax;
    };

    class Mat3Parameter : public Parameter<Mat3>
    {
    public:
        Mat3Parameter(const Str& name, const Mat3& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2)
            : Parameter<Mat3>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~Mat3Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<F32, 3, 3, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data() + 3), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data() + 6), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            mValue                                       = rM;
            return row0 || row1 || row2;
        }

    private:
        F32 mMin;
        F32 mMax;
    };

    class Mat4Parameter : public Parameter<Mat4>
    {
    public:
        Mat4Parameter(const Str& name, const Mat4& value, const F32& min = FLT_MIN, const F32& max = FLT_MAX / 2)
            : Parameter<Mat4>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~Mat4Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<F32, 4, 4, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data() + 4), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data() + 8), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row3                                    = ImGui::SliderScalarN((mName + "3").c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(rM.data() + 12), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            mValue                                       = rM;
            return row0 || row1 || row2 || row3;
        }

    private:
        F32 mMin;
        F32 mMax;
    };

    class QuatParameter : public Parameter<Quat>
    {
    public:
        QuatParameter(const Str& name, const Quat& value)
            : Parameter<Quat>(name, value)
        {
        }

        ~QuatParameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Float, reinterpret_cast<void*>(mValue.coeffs().data()), 4, nullptr, nullptr);
        }
    };

    class DVec2Parameter : public Parameter<DVec2>
    {
    public:
        DVec2Parameter(const Str& name, const DVec2& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2)
            : Parameter<DVec2>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~DVec2Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(mValue.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        F64 mMin;
        F64 mMax;
    };

    class DVec3Parameter : public Parameter<DVec3>
    {
    public:
        DVec3Parameter(const Str& name, const DVec3& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2)
            : Parameter<DVec3>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~DVec3Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(mValue.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        F64 mMin;
        F64 mMax;
    };

    class DVec4Parameter : public Parameter<DVec4>
    {
    public:
        DVec4Parameter(const Str& name, const DVec4& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2)
            : Parameter<DVec4>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~DVec4Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(mValue.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        F64 mMin;
        F64 mMax;
    };

    class DMat2Parameter : public Parameter<DMat2>
    {
    public:
        DMat2Parameter(const Str& name, const DMat2& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2)
            : Parameter<DMat2>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~DMat2Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<F64, 2, 2, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data() + 2), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            mValue                                       = rM;
            return row0 || row1;
        }

    private:
        F64 mMin;
        F64 mMax;
    };

    class DMat3Parameter : public Parameter<DMat3>
    {
    public:
        DMat3Parameter(const Str& name, const DMat3& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2)
            : Parameter<DMat3>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~DMat3Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<F64, 3, 3, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data() + 3), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data() + 6), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            mValue                                       = rM;
            return row0 || row1 || row2;
        }

    private:
        F64 mMin;
        F64 mMax;
    };

    class DMat4Parameter : public Parameter<DMat4>
    {
    public:
        DMat4Parameter(const Str& name, const DMat4& value, const F64& min = DBL_MIN, const F64& max = DBL_MAX / 2)
            : Parameter<DMat4>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~DMat4Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<F64, 4, 4, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data() + 4), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data() + 8), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row3                                    = ImGui::SliderScalarN((mName + "3").c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(rM.data() + 12), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            mValue                                       = rM;
            return row0 || row1 || row2 || row3;
        }

    private:
        F64 mMin;
        F64 mMax;
    };

    class DQuatParameter : public Parameter<DQuat>
    {
    public:
        DQuatParameter(const Str& name, const DQuat& value)
            : Parameter<DQuat>(name, value)
        {
        }

        ~DQuatParameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_Double, reinterpret_cast<void*>(mValue.coeffs().data()), 4, nullptr, nullptr);
        }
    };

    class IVec2Parameter : public Parameter<IVec2>
    {
    public:
        IVec2Parameter(const Str& name, const IVec2& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2)
            : Parameter<IVec2>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~IVec2Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(mValue.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        I32 mMin;
        I32 mMax;
    };

    class IVec3Parameter : public Parameter<IVec3>
    {
    public:
        IVec3Parameter(const Str& name, const IVec3& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2)
            : Parameter<IVec3>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~IVec3Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(mValue.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        I32 mMin;
        I32 mMax;
    };

    class IVec4Parameter : public Parameter<IVec4>
    {
    public:
        IVec4Parameter(const Str& name, const IVec4& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2)
            : Parameter<IVec4>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~IVec4Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(mValue.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        I32 mMin;
        I32 mMax;
    };

    class IMat2Parameter : public Parameter<IMat2>
    {
    public:
        IMat2Parameter(const Str& name, const IMat2& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2)
            : Parameter<IMat2>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~IMat2Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<I32, 2, 2, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data() + 2), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            mValue                                       = rM;
            return row0 || row1;
        }

    private:
        I32 mMin;
        I32 mMax;
    };

    class IMat3Parameter : public Parameter<IMat3>
    {
    public:
        IMat3Parameter(const Str& name, const IMat3& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2)
            : Parameter<IMat3>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~IMat3Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<I32, 3, 3, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data() + 3), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data() + 6), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            mValue                                       = rM;
            return row0 || row1 || row2;
        }

    private:
        I32 mMin;
        I32 mMax;
    };

    class IMat4Parameter : public Parameter<IMat4>
    {
    public:
        IMat4Parameter(const Str& name, const IMat4& value, const I32& min = INT32_MIN, const I32& max = INT32_MAX / 2)
            : Parameter<IMat4>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~IMat4Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<I32, 4, 4, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data() + 4), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data() + 8), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row3                                    = ImGui::SliderScalarN((mName + "3").c_str(), ImGuiDataType_S32, reinterpret_cast<void*>(rM.data() + 12), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            mValue                                       = rM;
            return row0 || row1 || row2 || row3;
        }

    private:
        I32 mMin;
        I32 mMax;
    };

    class UVec2Parameter : public Parameter<UVec2>
    {
    public:
        UVec2Parameter(const Str& name, const UVec2& value, const U32& min = 0, const U32& max = UINT32_MAX / 2)
            : Parameter<UVec2>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~UVec2Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(mValue.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        U32 mMin;
        U32 mMax;
    };

    class UVec3Parameter : public Parameter<UVec3>
    {
    public:
        UVec3Parameter(const Str& name, const UVec3& value, const U32& min = 0, const U32& max = UINT32_MAX / 2)
            : Parameter<UVec3>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~UVec3Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(mValue.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        U32 mMin;
        U32 mMax;
    };

    class UVec4Parameter : public Parameter<UVec4>
    {
    public:
        UVec4Parameter(const Str& name, const UVec4& value, const U32& min = 0, const U32& max = UINT32_MAX / 2)
            : Parameter<UVec4>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~UVec4Parameter() = default;

        bool onUIRender() override
        {
            return ImGui::SliderScalarN(mName.c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(mValue.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
        }

    private:
        U32 mMin;
        U32 mMax;
    };

    class UMat2Parameter : public Parameter<UMat2>
    {
    public:
        UMat2Parameter(const Str& name, const UMat2& value, const U32& min = 0, const U32& max = UINT32_MAX / 2)
            : Parameter<UMat2>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~UMat2Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<U32, 2, 2, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data()), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data() + 2), 2, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            mValue                                       = rM;
            return row0 || row1;
        }

    private:
        U32 mMin;
        U32 mMax;
    };

    class UMat3Parameter : public Parameter<UMat3>
    {
    public:
        UMat3Parameter(const Str& name, const UMat3& value, const U32& min = 0, const U32& max = UINT32_MAX / 2)
            : Parameter<UMat3>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~UMat3Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<U32, 3, 3, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data()), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data() + 3), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data() + 6), 3, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            mValue                                       = rM;
            return row0 || row1 || row2;
        }

    private:
        U32 mMin;
        U32 mMax;
    };

    class UMat4Parameter : public Parameter<UMat4>
    {
    public:
        UMat4Parameter(const Str& name, const UMat4& value, const U32& min = 0, const U32& max = UINT32_MAX / 2)
            : Parameter<UMat4>(name, value)
            , mMin(min)
            , mMax(max)
        {
        }

        ~UMat4Parameter() = default;

        bool onUIRender() override
        {
            Eigen::Matrix<U32, 4, 4, Eigen::RowMajor> rM = mValue;
            bool row0                                    = ImGui::SliderScalarN((mName + "0").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data()), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row1                                    = ImGui::SliderScalarN((mName + "1").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data() + 4), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row2                                    = ImGui::SliderScalarN((mName + "2").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data() + 8), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            bool row3                                    = ImGui::SliderScalarN((mName + "3").c_str(), ImGuiDataType_U32, reinterpret_cast<void*>(rM.data() + 12), 4, reinterpret_cast<void*>(&mMin), reinterpret_cast<void*>(&mMax));
            mValue                                       = rM;
            return row0 || row1 || row2 || row3;
        }

    private:
        U32 mMin;
        U32 mMax;
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
            bool changed = ImGui::InputText(mName.c_str(), mBuf.begin().base(), mBuf.size());

            if (changed)
            {
                mValue = Str(mBuf.begin().base(), mBuf.size());
            }

            return changed;
        }

    private:
        std::vector<char> mBuf;
    };

    namespace detail
    {
        inline void setupParametersInLuaState(sol::state& state)
        {
            state.new_usertype<ByteParameter>("ByteParameter",
                                              "get", sol::resolve<const Byte&() const>(&ByteParameter::get),
                                              "set", &ByteParameter::set);

            state.new_usertype<I16Parameter>("I16Parameter",
                                             "get", sol::resolve<const I16&() const>(&I16Parameter::get),
                                             "set", &I16Parameter::set);

            state.new_usertype<I32Parameter>("I32Parameter",
                                             "get", sol::resolve<const I32&() const>(&I32Parameter::get),
                                             "set", &I32Parameter::set);

            state.new_usertype<I64Parameter>("I64Parameter",
                                             "get", sol::resolve<const I64&() const>(&I64Parameter::get),
                                             "set", &I64Parameter::set);

            state.new_usertype<U16Parameter>("U16Parameter",
                                             "get", sol::resolve<const U16&() const>(&U16Parameter::get),
                                             "set", &U16Parameter::set);

            state.new_usertype<U32Parameter>("U32Parameter",
                                             "get", sol::resolve<const U32&() const>(&U32Parameter::get),
                                             "set", &U32Parameter::set);

            state.new_usertype<U64Parameter>("U64Parameter",
                                             "get", sol::resolve<const U64&() const>(&U64Parameter::get),
                                             "set", &U64Parameter::set);

            state.new_usertype<F32Parameter>("F32Parameter",
                                             "get", sol::resolve<const F32&() const>(&F32Parameter::get),
                                             "set", &F32Parameter::set);

            state.new_usertype<F64Parameter>("F64Parameter",
                                             "get", sol::resolve<const F64&() const>(&F64Parameter::get),
                                             "set", &F64Parameter::set);

            // TODO: Implement Lua for Eigen Types
            /*
            state.new_usertype<Vec2Parameter>("Vec2Parameter",
                                              "get", sol::resolve<const Vec2&() const>(&Vec2Parameter::get),
                                              "set", &Vec2Parameter::set);

            state.new_usertype<Vec3Parameter>("Vec3Parameter",
                                              "get", sol::resolve<const Vec3&() const>(&Vec3Parameter::get),
                                              "set", &Vec3Parameter::set);

            state.new_usertype<Vec4Parameter>("Vec4Parameter",
                                              "get", sol::resolve<const Vec4&() const>(&Vec4Parameter::get),
                                              "set", &Vec4Parameter::set);

            state.new_usertype<Mat2Parameter>("Mat2Parameter",
                                              "get", sol::resolve<const Mat2&() const>(&Mat2Parameter::get),
                                              "set", &Mat2Parameter::set);

            state.new_usertype<Mat3Parameter>("Mat3Parameter",
                                              "get", sol::resolve<const Mat3&() const>(&Mat3Parameter::get),
                                              "set", &Mat3Parameter::set);

            state.new_usertype<Mat4Parameter>("Mat4Parameter",
                                              "get", sol::resolve<const Mat4&() const>(&Mat4Parameter::get),
                                              "set", &Mat4Parameter::set);

            state.new_usertype<QuatParameter>("QuatParameter",
                                              "get", sol::resolve<const Quat&() const>(&QuatParameter::get),
                                              "set", &QuatParameter::set);

            state.new_usertype<DVec2Parameter>("DVec2Parameter",
                                               "get", sol::resolve<const DVec2&() const>(&DVec2Parameter::get),
                                               "set", &DVec2Parameter::set);

            state.new_usertype<DVec3Parameter>("DVec3Parameter",
                                               "get", sol::resolve<const DVec3&() const>(&DVec3Parameter::get),
                                               "set", &DVec3Parameter::set);

            state.new_usertype<DVec4Parameter>("DVec4Parameter",
                                               "get", sol::resolve<const DVec4&() const>(&DVec4Parameter::get),
                                               "set", &DVec4Parameter::set);

            state.new_usertype<DMat2Parameter>("DMat2Parameter",
                                               "get", sol::resolve<const DMat2&() const>(&DMat2Parameter::get),
                                               "set", &DMat2Parameter::set);

            state.new_usertype<DMat3Parameter>("DMat3Parameter",
                                               "get", sol::resolve<const DMat3&() const>(&DMat3Parameter::get),
                                               "set", &DMat3Parameter::set);

            state.new_usertype<DMat4Parameter>("DMat4Parameter",
                                               "get", sol::resolve<const DMat4&() const>(&DMat4Parameter::get),
                                               "set", &DMat4Parameter::set);

            state.new_usertype<DQuatParameter>("DQuatParameter",
                                               "get", sol::resolve<const DQuat&() const>(&DQuatParameter::get),
                                               "set", &DQuatParameter::set);

            state.new_usertype<IVec2Parameter>("IVec2Parameter",
                                               "get", sol::resolve<const IVec2&() const>(&IVec2Parameter::get),
                                               "set", &IVec2Parameter::set);

            state.new_usertype<IVec3Parameter>("IVec3Parameter",
                                               "get", sol::resolve<const IVec3&() const>(&IVec3Parameter::get),
                                               "set", &IVec3Parameter::set);

            state.new_usertype<IVec4Parameter>("IVec4Parameter",
                                               "get", sol::resolve<const IVec4&() const>(&IVec4Parameter::get),
                                               "set", &IVec4Parameter::set);

            state.new_usertype<IMat2Parameter>("IMat2Parameter",
                                               "get", sol::resolve<const IMat2&() const>(&IMat2Parameter::get),
                                               "set", &IMat2Parameter::set);

            state.new_usertype<IMat3Parameter>("IMat3Parameter",
                                               "get", sol::resolve<const IMat3&() const>(&IMat3Parameter::get),
                                               "set", &IMat3Parameter::set);

            state.new_usertype<IMat4Parameter>("IMat4Parameter",
                                               "get", sol::resolve<const IMat4&() const>(&IMat4Parameter::get),
                                               "set", &IMat4Parameter::set);

            state.new_usertype<UVec2Parameter>("UVec2Parameter",
                                               "get", sol::resolve<const UVec2&() const>(&UVec2Parameter::get),
                                               "set", &UVec2Parameter::set);

            state.new_usertype<UVec3Parameter>("UVec3Parameter",
                                               "get", sol::resolve<const UVec3&() const>(&UVec3Parameter::get),
                                               "set", &UVec3Parameter::set);

            state.new_usertype<UVec4Parameter>("UVec4Parameter",
                                               "get", sol::resolve<const UVec4&() const>(&UVec4Parameter::get),
                                               "set", &UVec4Parameter::set);

            state.new_usertype<UMat2Parameter>("UMat2Parameter",
                                               "get", sol::resolve<const UMat2&() const>(&UMat2Parameter::get),
                                               "set", &UMat2Parameter::set);

            state.new_usertype<UMat3Parameter>("UMat3Parameter",
                                               "get", sol::resolve<const UMat3&() const>(&UMat3Parameter::get),
                                               "set", &UMat3Parameter::set);

            state.new_usertype<UMat4Parameter>("UMat4Parameter",
                                               "get", sol::resolve<const UMat4&() const>(&UMat4Parameter::get),
                                               "set", &UMat4Parameter::set);
            */

            state.new_usertype<BoolParameter>("BoolParameter",
                                              "get", sol::resolve<const bool&() const>(&BoolParameter::get),
                                              "set", &BoolParameter::set);

            state.new_usertype<StrParameter>("StrParameter",
                                             "get", sol::resolve<const Str&() const>(&StrParameter::get),
                                             "set", &StrParameter::set);
        }
    } // namespace detail
} // namespace saf

#endif // PARAMETER_HPP
