/**
 * @file      types.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2024
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef TYPES_HPP
#define TYPES_HPP

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <iostream>
#include <memory>
#include <stdint.h>
#include <string.h>

/** @cond NO_DOC */
#pragma warning(push, 0)
#define NOMINMAX
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#pragma warning(pop)
/** @endcond */

namespace saf
{

#ifndef SAF_ASSERT
#include <cassert>
#define SAF_ASSERT(expr) assert(expr)
#endif

    /** @brief Typedef for one byte. */
    using Byte = unsigned char;
    /** @brief Typedef for 16 bit integers. */
    using I16 = short int;
    /** @brief Typedef for 32 bit integers. */
    using I32 = int;
    /** @brief Typedef for 64 bit integers. */
    using I64 = long long;
    /** @brief Typedef for 16 bit unsigned integers. */
    using U16 = unsigned short int;
    /** @brief Typedef for 32 bit unsigned integers. */
    using U32 = unsigned int;
    /** @brief Typedef for 64 bit unsigned integers. */
    using U64 = unsigned long long;
    /** @brief Typedef for size_t. */
    using PtrSize = std::size_t;

    /** @brief Typedef for float. */
    using F32 = float;
    /** @brief Typedef for double. */
    using F64 = double;

    /** @brief Typedef for a 4 component F32 vector. */
    using Vec4 = Eigen::Vector4<F32>;
    /** @brief Typedef for a 4x4 F32 matrix. */
    using Mat4 = Eigen::Matrix4<F32>;
    /** @brief Typedef for a 3 component F32 vector. */
    using Vec3 = Eigen::Vector3<F32>;
    /** @brief Typedef for a 3x3 F32 matrix. */
    using Mat3 = Eigen::Matrix3<F32>;
    /** @brief Typedef for a 2 component F32 vector. */
    using Vec2 = Eigen::Vector2<F32>;
    /** @brief Typedef for a 2x2 F32 matrix. */
    using Mat2 = Eigen::Matrix2<F32>;
    /** @brief Type alias for a Eigen::Quaternionf. */
    using Quat = Eigen::Quaternionf;

    /** @brief Typedef for a 4 component F64 vector. */
    using DVec4 = Eigen::Vector4<F64>;
    /** @brief Typedef for a 4x4 F64 matrix. */
    using DMat4 = Eigen::Matrix4<F64>;
    /** @brief Typedef for a 3 component F64 vector. */
    using DVec3 = Eigen::Vector3<F64>;
    /** @brief Typedef for a 3x3 F64 matrix. */
    using DMat3 = Eigen::Matrix3<F64>;
    /** @brief Typedef for a 2 component F64 vector. */
    using DVec2 = Eigen::Vector2<F64>;
    /** @brief Typedef for a 2x2 F64 matrix. */
    using DMat2 = Eigen::Matrix2<F64>;
    /** @brief Type alias for a Eigen::Quaterniond. */
    using DQuat = Eigen::Quaterniond;

    /** @brief Typedef for a 4 component I32 vector. */
    using IVec4 = Eigen::Vector4<I32>;
    /** @brief Typedef for a 4x4 I32 matrix. */
    using IMat4 = Eigen::Matrix4<I32>;
    /** @brief Typedef for a 3 component I32 vector. */
    using IVec3 = Eigen::Vector3<I32>;
    /** @brief Typedef for a 3x3 I32 matrix. */
    using IMat3 = Eigen::Matrix3<I32>;
    /** @brief Typedef for a 2 component I32 vector. */
    using IVec2 = Eigen::Vector2<I32>;
    /** @brief Typedef for a 2x2 I32 matrix. */
    using IMat2 = Eigen::Matrix2<I32>;

    /** @brief Typedef for a 4 component U32 vector. */
    using UVec4 = Eigen::Vector4<U32>;
    /** @brief Typedef for a 4x4 U32 matrix. */
    using UMat4 = Eigen::Matrix4<U32>;
    /** @brief Typedef for a 3 component U32 vector. */
    using UVec3 = Eigen::Vector3<U32>;
    /** @brief Typedef for a 3x3 U32 matrix. */
    using UMat3 = Eigen::Matrix3<U32>;
    /** @brief Typedef for a 2 component U32 vector. */
    using UVec2 = Eigen::Vector2<U32>;
    /** @brief Typedef for a 2x2 U32 matrix. */
    using UMat2 = Eigen::Matrix2<U32>;

    /** @brief Typedef for a string. */
    using Str = std::string;

    /** @cond NO_DOC */

    template <typename e>
    struct BitMaskOperations
    {
        static const bool enable = false;
    };

/** @brief Macro used to enable safe bitmask operations on enum classes. */
#define ENABLE_BITMASK_OPERATIONS(e)         \
    template <>                              \
    struct BitMaskOperations<e>              \
    {                                        \
        static constexpr bool enable = true; \
    };

    template <typename e>
    typename std::enable_if<BitMaskOperations<e>::enable, e>::type operator|(e Lhs, e Rhs)
    {
        typedef typename std::underlying_type<e>::type Underlying;
        return static_cast<e>(static_cast<Underlying>(Lhs) | static_cast<Underlying>(Rhs));
    }

    template <typename e>
    typename std::enable_if<BitMaskOperations<e>::enable, e>::type operator&(e Lhs, e Rhs)
    {
        typedef typename std::underlying_type<e>::type Underlying;
        return static_cast<e>(static_cast<Underlying>(Lhs) & static_cast<Underlying>(Rhs));
    }

    template <typename e>
    typename std::enable_if<BitMaskOperations<e>::enable, e>::type operator^(e Lhs, e Rhs)
    {
        typedef typename std::underlying_type<e>::type Underlying;
        return static_cast<e>(static_cast<Underlying>(Lhs) ^ static_cast<Underlying>(Rhs));
    }

    template <typename e>
    typename std::enable_if<BitMaskOperations<e>::enable, e>::type operator~(e Lhs)
    {
        typedef typename std::underlying_type<e>::type Underlying;
        return static_cast<e>(~static_cast<Underlying>(Lhs));
    }

    template <typename e>
    typename std::enable_if<BitMaskOperations<e>::enable, e&>::type operator|=(e& Lhs, e Rhs)
    {
        typedef typename std::underlying_type<e>::type Underlying;
        Lhs = static_cast<e>(static_cast<Underlying>(Lhs) | static_cast<Underlying>(Rhs));
        return Lhs;
    }

    template <typename e>
    typename std::enable_if<BitMaskOperations<e>::enable, e&>::type operator&=(e& Lhs, e Rhs)
    {
        typedef typename std::underlying_type<e>::type Underlying;
        Lhs = static_cast<e>(static_cast<Underlying>(Lhs) & static_cast<Underlying>(Rhs));
        return Lhs;
    }

    template <typename e>
    typename std::enable_if<BitMaskOperations<e>::enable, e&>::type operator^=(e& Lhs, e Rhs)
    {
        typedef typename std::underlying_type<e>::type Underlying;
        Lhs = static_cast<e>(static_cast<Underlying>(Lhs) ^ static_cast<Underlying>(Rhs));
        return Lhs;
    }

    /** @endcond

/** @brief Pi. */
#define PI 3.1415926535897932384626433832795
/** @brief Pi times two. */
#define TWO_PI (2.0 * PI)
/** @brief Pi divided by two. */
#define HALF_PI (0.5 * PI)
/** @brief One divided by Pi. */
#define INV_PI (1.0 / PI)

    inline void checkVkResult(VkResult err)
    {
        if (err == 0)
        {
            return;
        }
        std::cerr << "[vulkan] Error: VkResult is " << err << '\n';
        if (err < 0)
        {
            std::abort();
        }
    }

#define ARRAYSIZE(carray) (static_cast<int>(sizeof(carray) / sizeof(*(carray))))

} // namespace saf

#endif // TYPES_HPP
