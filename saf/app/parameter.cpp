/**
 * @file      parameter.cpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2025
 * @copyright Apache License 2.0
 */

#include "parameter.hpp"

using namespace saf;

namespace sol
{
#ifdef SAF_SCRIPTING

#define DISABLE_SOL_BREAKING_MATRICES(Type)       \
    template <>                                   \
    struct is_automagical<Type> : std::false_type \
    {                                             \
    };                                            \
    template <>                                   \
    struct is_container<Type> : std::false_type   \
    {                                             \
    }

    DISABLE_SOL_BREAKING_MATRICES(Mat2);
    DISABLE_SOL_BREAKING_MATRICES(Mat3);
    DISABLE_SOL_BREAKING_MATRICES(Mat4);
    DISABLE_SOL_BREAKING_MATRICES(Quat);
    DISABLE_SOL_BREAKING_MATRICES(DMat2);
    DISABLE_SOL_BREAKING_MATRICES(DMat3);
    DISABLE_SOL_BREAKING_MATRICES(DMat4);
    DISABLE_SOL_BREAKING_MATRICES(DQuat);
    DISABLE_SOL_BREAKING_MATRICES(IMat2);
    DISABLE_SOL_BREAKING_MATRICES(IMat3);
    DISABLE_SOL_BREAKING_MATRICES(IMat4);
    DISABLE_SOL_BREAKING_MATRICES(UMat2);
    DISABLE_SOL_BREAKING_MATRICES(UMat3);
    DISABLE_SOL_BREAKING_MATRICES(UMat4);

#undef DISABLE_SOL_BREAKING_MATRICES
}

template <typename P, typename S>
void new_parameter_user_type(const Str& name, sol::state& state)
{
    state.new_usertype<P>(name,
                          "get", sol::resolve<const S&() const>(&P::get),
                          "set", &P::set,
                          "value", sol::property(sol::resolve<const S&() const>(&P::get), &P::set));
}

template <typename V, typename S, size_t N>
void new_vector_user_type(const Str& name, sol::state& state)
{
    if constexpr (N == 2)
    {
        // clang-format off
                state.new_usertype<V>(
                    name, sol::constructors<V(S, S)>(),
                    "x", sol::property(sol::resolve<const S&() const>(&V::x), [](V& self, S x) { self.x() = x; }),
                    "y", sol::property(sol::resolve<const S&() const>(&V::y), [](V& self, S y) { self.y() = y; }),
                    sol::meta_function::addition, [](V& self, const V& other) { return V(self + other); },
                    sol::meta_function::subtraction, [](V& self, const V& other) { return V(self - other); },
                    sol::meta_function::division, [](V& self, S scalar) { return V(self / scalar); },
                    sol::meta_function::multiplication, sol::overload([](V& self, S scalar) { return V(self * scalar); }, [](S scalar, V& self) { return V(self * scalar); }));
        // clang-format on
    }
    else if constexpr (N == 3)
    {
        // clang-format off
                state.new_usertype<V>(
                    name, sol::constructors<V(S, S, S)>(),
                    "x", sol::property(sol::resolve<const S&() const>(&V::x), [](V& self, S x) { self.x() = x; }),
                    "y", sol::property(sol::resolve<const S&() const>(&V::y), [](V& self, S y) { self.y() = y; }),
                    "z", sol::property(sol::resolve<const S&() const>(&V::z), [](V& self, S z) { self.z() = z; }),
                    sol::meta_function::addition, [](V& self, const V& other) { return V(self + other); },
                    sol::meta_function::subtraction, [](V& self, const V& other) { return V(self - other); },
                    sol::meta_function::division, [](V& self, S scalar) { return V(self / scalar); },
                    sol::meta_function::multiplication, sol::overload([](V& self, S scalar) { return V(self * scalar); }, [](S scalar, V& self) { return V(self * scalar); }));
        // clang-format on
    }
    else if constexpr (N == 4)
    {
        // clang-format off
                state.new_usertype<V>(
                    name, sol::constructors<V(S, S, S, S)>(),
                    "x", sol::property(sol::resolve<const S&() const>(&V::x), [](V& self, S x) { self.x() = x; }),
                    "y", sol::property(sol::resolve<const S&() const>(&V::y), [](V& self, S y) { self.y() = y; }),
                    "z", sol::property(sol::resolve<const S&() const>(&V::z), [](V& self, S z) { self.z() = z; }),
                    "w", sol::property(sol::resolve<const S&() const>(&V::w), [](V& self, S w) { self.w() = w; }),
                    sol::meta_function::addition, [](V& self, const V& other) { return V(self + other); },
                    sol::meta_function::subtraction, [](V& self, const V& other) { return V(self - other); },
                    sol::meta_function::division, [](V& self, S scalar) { return V(self / scalar); },
                    sol::meta_function::multiplication, sol::overload([](V& self, S scalar) { return V(self * scalar); }, [](S scalar, V& self) { return V(self * scalar); }));
        // clang-format on
    }
}

template <typename M, typename S>
void new_matrix_user_type(const Str& name, sol::state& state)
{
    // clang-format off
            state.new_usertype<M>(
                name,
                sol::meta_function::construct, sol::factories(
                    [](sol::object)
                    {
                        return std::make_shared<M>(M::Zero());
                    },
                    [](sol::object, S v)
                    {
                        return std::make_shared<M>(M::Constant(v));
                    }),
                sol::meta_function::call, [](M& self, PtrSize r, PtrSize c) { return self(r, c); },
                "get", [](M& self, PtrSize r, PtrSize c) { return self(r, c); },
                "set", [](M& self, PtrSize r, PtrSize c, S v) { self(r, c) = v; });
    // clang-format on
}

void saf::setupParametersInLuaState(sol::state& state)
{
    new_vector_user_type<Vec2, F32, 2>("Vec2", state);
    new_vector_user_type<Vec3, F32, 3>("Vec3", state);
    new_vector_user_type<Vec4, F32, 4>("Vec4", state);

    new_vector_user_type<DVec2, F64, 2>("DVec2", state);
    new_vector_user_type<DVec3, F64, 3>("DVec3", state);
    new_vector_user_type<DVec4, F64, 4>("DVec4", state);

    new_vector_user_type<IVec2, I32, 2>("IVec2", state);
    new_vector_user_type<IVec3, I32, 3>("IVec3", state);
    new_vector_user_type<IVec4, I32, 4>("IVec4", state);

    new_vector_user_type<UVec2, U32, 2>("UVec2", state);
    new_vector_user_type<UVec3, U32, 3>("UVec3", state);
    new_vector_user_type<UVec4, U32, 4>("UVec4", state);

    new_matrix_user_type<Mat2, F32>("Mat2", state);
    new_matrix_user_type<Mat3, F32>("Mat3", state);
    new_matrix_user_type<Mat4, F32>("Mat4", state);
    // new_matrix_user_type<Quat, F32>("Quat", state);

    new_matrix_user_type<DMat2, F64>("DMat2", state);
    new_matrix_user_type<DMat3, F64>("DMat3", state);
    new_matrix_user_type<DMat4, F64>("DMat4", state);
    // new_matrix_user_type<DQuat, F64>("Quat", state);

    new_matrix_user_type<IMat2, I32>("IMat2", state);
    new_matrix_user_type<IMat3, I32>("IMat3", state);
    new_matrix_user_type<IMat4, I32>("IMat4", state);

    new_matrix_user_type<UMat2, U32>("UMat2", state);
    new_matrix_user_type<UMat3, U32>("UMat3", state);
    new_matrix_user_type<UMat4, U32>("UMat4", state);

    new_parameter_user_type<ByteParameter, Byte>("ByteParameter", state);
    new_parameter_user_type<I16Parameter, I16>("I16Parameter", state);
    new_parameter_user_type<I32Parameter, I32>("I32Parameter", state);
    new_parameter_user_type<I64Parameter, I64>("I64Parameter", state);
    new_parameter_user_type<U16Parameter, U16>("U16Parameter", state);
    new_parameter_user_type<U32Parameter, U32>("U32Parameter", state);
    new_parameter_user_type<U64Parameter, U64>("U64Parameter", state);
    new_parameter_user_type<F32Parameter, F32>("F32Parameter", state);
    new_parameter_user_type<F64Parameter, F64>("F64Parameter", state);

    new_parameter_user_type<Vec2Parameter, Vec2>("Vec2Parameter", state);
    new_parameter_user_type<Vec3Parameter, Vec3>("Vec3Parameter", state);
    new_parameter_user_type<Vec4Parameter, Vec4>("Vec4Parameter", state);

    new_parameter_user_type<Mat2Parameter, Mat2>("Mat2Parameter", state);
    new_parameter_user_type<Mat3Parameter, Mat3>("Mat3Parameter", state);
    new_parameter_user_type<Mat4Parameter, Mat4>("Mat4Parameter", state);
    // new_parameter_user_type<QuatParameter, Quat>("QuatParameter", state);

    new_parameter_user_type<DVec2Parameter, DVec2>("DVec2Parameter", state);
    new_parameter_user_type<DVec3Parameter, DVec3>("DVec3Parameter", state);
    new_parameter_user_type<DVec4Parameter, DVec4>("DVec4Parameter", state);

    new_parameter_user_type<DMat2Parameter, DMat2>("DMat2Parameter", state);
    new_parameter_user_type<DMat3Parameter, DMat3>("DMat3Parameter", state);
    new_parameter_user_type<DMat4Parameter, DMat4>("DMat4Parameter", state);
    // new_parameter_user_type<DQuatParameter, DQuat>("DQuatParameter", state);

    new_parameter_user_type<IVec2Parameter, IVec2>("IVec2Parameter", state);
    new_parameter_user_type<IVec3Parameter, IVec3>("IVec3Parameter", state);
    new_parameter_user_type<IVec4Parameter, IVec4>("IVec4Parameter", state);

    new_parameter_user_type<IMat2Parameter, IMat2>("IMat2Parameter", state);
    new_parameter_user_type<IMat3Parameter, IMat3>("IMat3Parameter", state);
    new_parameter_user_type<IMat4Parameter, IMat4>("IMat4Parameter", state);

    new_parameter_user_type<UVec2Parameter, UVec2>("UVec2Parameter", state);
    new_parameter_user_type<UVec3Parameter, UVec3>("UVec3Parameter", state);
    new_parameter_user_type<UVec4Parameter, UVec4>("UVec4Parameter", state);

    new_parameter_user_type<UMat2Parameter, UMat2>("UMat2Parameter", state);
    new_parameter_user_type<UMat3Parameter, UMat3>("UMat3Parameter", state);
    new_parameter_user_type<UMat4Parameter, UMat4>("UMat4Parameter", state);

    new_parameter_user_type<BoolParameter, bool>("BoolParameter", state);
    new_parameter_user_type<StrParameter, Str>("StrParameter", state);
#endif
}
