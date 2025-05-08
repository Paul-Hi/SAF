/**
 * @file      types.hpp
 * @author    Paul Himmler
 * @version   0.01
 * @date      2025
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
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#pragma warning(pop)
/** @endcond */

namespace saf
{

#ifndef SAF_ASSERT
#include <cassert>
#define SAF_ASSERT(expr) assert(expr)
#endif

#ifdef SAF_CUDA_INTEROP

#ifdef __CUDA_ARCH__

/** @brief Wrapper for __CUDA_ARCH__. */
#define CUDA_COMPILE_PHASE

/** @brief Function to use from kernels and host functions. */
#define CUDA_HOST_DEVICE __host__ __device__
/** @brief Host function. */
#define CUDA_HOST __host__
/** @brief Device function. */
#define CUDA_DEVICE __device__
/** @brief Kernel function. */
#define CUDA_GLOBAL_KERNEL __global__

#else

/** @brief Function to use from kernels and host functions. */
#define CUDA_HOST_DEVICE
/** @brief Host function. */
#define CUDA_HOST
/** @brief Device function. */
#define CUDA_DEVICE
/** @brief Kernel function. */
#define CUDA_GLOBAL_KERNEL __global__

#endif

#define CUDA_CHECK(call)                                                                                                                                    \
    {                                                                                                                                                       \
        cudaError_t _result = call;                                                                                                                         \
        if (cudaSuccess != _result)                                                                                                                         \
        {                                                                                                                                                   \
            std::cerr << "[CUDA] Error " << _result << " in " << __FILE__ << " " << __LINE__ << " " << cudaGetErrorString(_result) << " " << #call << "\n"; \
            exit(0);                                                                                                                                        \
        }                                                                                                                                                   \
    }

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

    /**
     * @brief Create a perspective matrix (Vulkan).
     * @param[in] fovy The vertical field of view in radians.
     * @param[in] aspect The aspect ratio.
     * @param[in] zNear Near plane depth.
     * @param[in] zFar Far plane depth.
     * @return An Vulkan perspective matrix.
     */
    template <typename Scalar>
    Eigen::Matrix<Scalar, 4, 4> perspective(Scalar fovy, Scalar aspect, Scalar zNear, Scalar zFar)
    {
        Eigen::Transform<Scalar, 3, Eigen::Projective> tr;
        tr.matrix().setZero();
        assert(aspect > 0);
        assert(zFar > zNear);
        assert(zNear > 0);
        Scalar tan_half_fovy = std::tan(fovy / static_cast<Scalar>(2));
        Scalar focalLength   = static_cast<Scalar>(1) / (tan_half_fovy);
        tr(0, 0)             = focalLength / aspect;
        tr(1, 1)             = -focalLength;
        tr(2, 2)             = zFar / (zNear - zFar);
        tr(3, 2)             = -static_cast<Scalar>(1);
        tr(2, 3)             = -(zFar * zNear) / (zFar - zNear);
        return tr.matrix();
    }

    /**
     * @brief Create a view matrix.
     * @param[in] eye Eye position.
     * @param[in] center The target position to look at.
     * @param[in] up Up vector.
     * @return A view matrix.
     */
    template <typename Derived>
    Eigen::Matrix<typename Derived::Scalar, 4, 4> lookAt(const Derived& eye, const Derived& center, const Derived& up)
    {
        typedef Eigen::Matrix<typename Derived::Scalar, 4, 4> Matrix4;
        typedef Eigen::Matrix<typename Derived::Scalar, 3, 1> Vector3;
        typedef Eigen::Matrix<typename Derived::Scalar, 4, 1> Vector4;

        const Vector3 z = (center - eye).normalized();
        const Vector3 x = (z.cross(up)).normalized();
        const Vector3 y = x.cross(z);

        Matrix4 lA;
        lA << x.x(), x.y(), x.z(), -x.dot(eye), y.x(), y.y(), y.z(), -y.dot(eye), -z.x(), -z.y(), -z.z(), z.dot(eye), 0.0, 0.0, 0.0, 1.0;

        return lA;
    }

    /**
     * @brief Create a scale matrix.
     * @param[in] x Scaling in x direction.
     * @param[in] y Scaling in y direction.
     * @param[in] z Scaling in z direction.
     * @return The scale matrix.
     */
    template <typename Scalar>
    Eigen::Matrix<Scalar, 4, 4> scale(Scalar x, Scalar y, Scalar z)
    {
        Eigen::Transform<Scalar, 3, Eigen::Affine> tr;
        tr.matrix().setZero();
        tr(0, 0) = x;
        tr(1, 1) = y;
        tr(2, 2) = z;
        tr(3, 3) = 1;
        return tr.matrix();
    }

    /**
     * @brief Create a translation matrix.
     * @param[in] x Translation in x direction.
     * @param[in] y Translation in y direction.
     * @param[in] z Translation in z direction.
     * @return The translation matrix.
     */
    template <typename Scalar>
    Eigen::Matrix<Scalar, 4, 4> translate(Scalar x, Scalar y, Scalar z)
    {
        Eigen::Transform<Scalar, 3, Eigen::Affine> tr;
        tr.matrix().setIdentity();
        tr(0, 3) = x;
        tr(1, 3) = y;
        tr(2, 3) = z;
        return tr.matrix();
    }

    /**
     * @brief Create a scale matrix.
     * @param[in] scale Scaling in x,y,z direction.
     * @return The scale matrix.
     */
    template <typename Scalar>
    Eigen::Matrix<Scalar, 4, 4> scale(const Eigen::Vector3<Scalar>& scale)
    {
        Eigen::Transform<Scalar, 3, Eigen::Affine> tr;
        tr.matrix().setZero();
        tr(0, 0) = scale.x();
        tr(1, 1) = scale.y();
        tr(2, 2) = scale.z();
        tr(3, 3) = 1;
        return tr.matrix();
    }

    /**
     * @brief Create a translation matrix.
     * @param[in] translation Translation in x,y,z direction.
     * @return The translation matrix.
     */
    template <typename Scalar>
    Eigen::Matrix<Scalar, 4, 4> translate(const Eigen::Vector3<Scalar>& translation)
    {
        Eigen::Transform<Scalar, 3, Eigen::Affine> tr;
        tr.matrix().setIdentity();
        tr(0, 3) = translation.x();
        tr(1, 3) = translation.y();
        tr(2, 3) = translation.z();
        return tr.matrix();
    }

} // namespace saf

#endif // TYPES_HPP
