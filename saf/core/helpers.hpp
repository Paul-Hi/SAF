/**
 * @file      helpers.hpp
 * @author    Paul Himmler
 * @version   1.00
 * @date      2026
 * @copyright Apache License 2.0
 */

#pragma once

#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <vulkan/vk_enum_string_helper.h>
#include <vulkan/vulkan_core.h>

#ifdef SAF_CUDA_INTEROP
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace saf
{
    inline void reportError(VkResult result, const char* expression, const char* file, int line)
    {
        const char* errMsg = string_VkResult(result);
        std::cerr << "[vulkan] Error:  " << errMsg << " from " << expression << ", " << file << ":" << line << "\n";
    }

#define VK_CHECK(vkFnc)                                           \
    {                                                             \
        const VkResult checkResult = (vkFnc);                     \
        if (checkResult < 0)                                      \
        {                                                         \
            reportError(checkResult, #vkFnc, __FILE__, __LINE__); \
            exit(1);                                              \
        }                                                         \
    }

#define VK_CHECK_RETURN(vkFnc)                                    \
    {                                                             \
        const VkResult checkResult = (vkFnc);                     \
        if (checkResult < 0)                                      \
        {                                                         \
            reportError(checkResult, #vkFnc, __FILE__, __LINE__); \
            return checkResult;                                   \
        }                                                         \
    }

#define VK_CHECK_RETURN_BOOL(vkFnc)                               \
    {                                                             \
        const VkResult checkResult = (vkFnc);                     \
        if (checkResult < 0)                                      \
        {                                                         \
            reportError(checkResult, #vkFnc, __FILE__, __LINE__); \
            return false;                                         \
        }                                                         \
    }

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
            std::cerr << "[cuda] Error " << _result << " in " << __FILE__ << " " << __LINE__ << " " << cudaGetErrorString(_result) << " " << #call << "\n"; \
            exit(0);                                                                                                                                        \
        }                                                                                                                                                   \
    }

#endif
} // namespace saf

#endif // HELPERS_HPP
