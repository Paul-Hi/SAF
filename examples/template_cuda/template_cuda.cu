// similar to https://www.shadertoy.com/view/ldl3W8 ported to CUDA from my Slang version
#include "template_cuda.cuh"
#include <core/helpers.hpp>
#include <math_constants.h>

using namespace saf;

// RGBA colors the kernel picks from.
__constant__ float4 cPalette[] = {
    { 0.06f, 0.10f, 0.23f, 1.0f },
    { 0.09f, 0.19f, 0.39f, 1.0f },
    { 0.12f, 0.29f, 0.55f, 1.0f },
    { 0.17f, 0.47f, 0.68f, 1.0f },
    { 0.23f, 0.64f, 0.79f, 1.0f },
    { 0.47f, 0.78f, 0.86f, 1.0f },
    { 0.72f, 0.89f, 0.91f, 1.0f },
    { 0.98f, 0.88f, 0.62f, 1.0f },
    { 0.95f, 0.72f, 0.35f, 1.0f },
    { 0.85f, 0.35f, 0.18f, 1.0f },
    { 0.60f, 0.16f, 0.12f, 1.0f }
};
constexpr U32 paletteSize = sizeof(cPalette) / sizeof(float4);

__device__ F32 fract(F32 x)
{
    return x - floorf(x);
}

__device__ F32 smoothstep(F32 edge0, F32 edge1, F32 x)
{
    const F32 t = __saturatef((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

__device__ float4 lerp(float4 a, float4 b, F32 t)
{
    return make_float4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
}

// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
__device__ U32 pcg(U32 v)
{
    const U32 state = v * 747796405u + 2891336453u;
    const U32 word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Explicit FMA, rounded like the shader's, because FMA inlining was not deterministic on my GPU
__device__ float2 hash(float2 p, U32 h)
{
    p.x += F32(h & 0x3FFu);
    p.y += F32((h >> 10u) & 0x3FFu);
    return make_float2(fract(sinf(__fmaf_rn(p.x, 127.1f, __fmul_rn(p.y, 311.7f))) * 18.5453f),
                       fract(sinf(__fmaf_rn(p.x, 269.5f, __fmul_rn(p.y, 183.3f))) * 18.5453f));
}

// The animated point of cell base + c, relative to fract
__device__ float2 cellPoint(float2 base, float2 c, float2 fract, F32 t, U32 h)
{
    const float2 r = hash(make_float2(base.x + c.x, base.y + c.y), h);
    return make_float2(c.x + 0.5f + 0.5f * sinf(t + 2.0f * CUDART_PI_F * r.x) - fract.x,
                       c.y + 0.5f + 0.5f * sinf(t + 2.0f * CUDART_PI_F * r.y) - fract.y);
}

// two 'pass' voronoi
__device__ float3 voronoiSDF(float2 uv, F32 t, F32 d, U32 h)
{
    F32 md = d;

    const float2 base  = make_float2(floorf(uv.x * md), floorf(uv.y * md));
    const float2 fract = make_float2(uv.x * md - base.x, uv.y * md - base.y);

    float2 mc{}, mp{};

    for (I32 i = -1; i <= 1; ++i)
    {
        for (I32 j = -1; j <= 1; ++j)
        {
            const float2 c = make_float2(i, j);
            const float2 p = cellPoint(base, c, fract, t, h);

            const F32 pd = p.x * p.x + p.y * p.y;

            if (pd < md)
            {
                mp = p;
                mc = c;
                md = pd;
            }
        }
    }

    md = d;
    for (I32 i = -2; i <= 2; ++i)
    {
        for (I32 j = -2; j <= 2; ++j)
        {
            const float2 p = cellPoint(base, make_float2(mc.x + i, mc.y + j), fract, t, h);

            const float2 n = make_float2(p.x - mp.x, p.y - mp.y);
            const F32 nn   = n.x * n.x + n.y * n.y;
            if (nn > 1e-4f)
            {
                const F32 pd = (0.5f * (p.x + mp.x) * n.x + 0.5f * (p.y + mp.y) * n.y) * rsqrtf(nn);
                md           = fminf(md, pd);
            }
        }
    }

    return make_float3(md, mp.x, mp.y);
}

CUDA_GLOBAL_KERNEL void voronoiKernel(cudaSurfaceObject_t imageSurface, U32 width, U32 height, F32 t, F32 d, U32 seed)
{
    const U32 x = threadIdx.x + blockIdx.x * blockDim.x;
    const U32 y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;
    const F32 px = d / F32(width); // one pixel in cell units

    const float2 uv = make_float2(F32(x) / F32(width), F32(y) / F32(height));

    const U32 h    = pcg(seed);
    const float3 v = voronoiSDF(uv, t, d, h);

    const F32 s  = __saturatef(v.x * 2.0f) * F32(paletteSize - 1) + 0.5f;
    const U32 i  = min(U32(s), paletteSize - 1);
    const F32 w  = 2.0f * F32(paletteSize - 1) * px; // how much s changes over one pixel
    float4 color = lerp(cPalette[i], cPalette[min(i + 1, paletteSize - 1)], smoothstep(1.0f - w, 1.0f, s - F32(i)));

    color = lerp(make_float4(0.0f, 0.0f, 0.0f, 1.0f), color, smoothstep(0.02f, 0.02f + px, hypotf(v.y, v.z))); // points

    // x is a byte offset
    surf2Dwrite(make_uchar4(color.x * 255.0f + 0.5f, color.y * 255.0f + 0.5f, color.z * 255.0f + 0.5f, color.w * 255.0f + 0.5f), imageSurface, x * 4, y);
}

void callVoronoiKernel(cudaSurfaceObject_t imageSurface, U32 width, U32 height, F32 time, F32 maxDistance, U32 seed)
{
    voronoiKernel<<<{ (width + 15) / 16, (height + 15) / 16 }, { 16, 16 }>>>(imageSurface, width, height, time, maxDistance, seed);
    CUDA_CHECK(cudaPeekAtLastError());
}
