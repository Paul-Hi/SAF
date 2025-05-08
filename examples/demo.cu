#ifdef SAF_CUDA_INTEROP
#include "demo.cuh"
#include <cuda.h>

using namespace saf;

CUDA_DEVICE unsigned int rgbaFloatToInt(float4 rgba)
{
    rgba.x = min(max(rgba.x, 0.0f), 1.0f);
    rgba.y = min(max(rgba.y, 0.0f), 1.0f);
    rgba.z = min(max(rgba.z, 0.0f), 1.0f);
    rgba.w = min(max(rgba.w, 0.0f), 1.0f);
    return ((unsigned int)(rgba.w * 255.0f) << 24) |
           ((unsigned int)(rgba.z * 255.0f) << 16) |
           ((unsigned int)(rgba.y * 255.0f) << 8) |
           ((unsigned int)(rgba.x * 255.0f));
}

CUDA_DEVICE float4 rgbaIntToFloat(unsigned int c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;         //  /255.0f;
    rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c >> 16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c >> 24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
}

CUDA_GLOBAL_KERNEL void grayScaleKernel(cudaSurfaceObject_t imageSurface, cudaTextureObject_t imageTexture, I32 w, I32 h)
{
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;

    F32 u = (F32)x / (F32)w;
    F32 v = (F32)y / (F32)w;

    //float4 readTest = tex2D<float4>(imageTexture, u, v);

    float4 readTest = rgbaIntToFloat(surf2Dread<unsigned int>(imageTexture, x * 4, y));

    F32 grayScale = readTest.x * 0.3 + readTest.y * 0.59 + readTest.z * 0.11;
    readTest.x    = grayScale;
    readTest.y    = grayScale;
    readTest.z    = grayScale;

    surf2Dwrite(rgbaFloatToInt(readTest), imageSurface, x * 4, y);
}

void callGrayScaleKernel(cudaSurfaceObject_t imageSurface, cudaTextureObject_t imageTexture, I32 w, I32 h)
{
    grayScaleKernel<<<{ w / 16, h / 16 }, { 16, 16 }>>>(imageSurface, imageTexture, w, h);
    CUDA_CHECK(cudaPeekAtLastError());
}

#endif