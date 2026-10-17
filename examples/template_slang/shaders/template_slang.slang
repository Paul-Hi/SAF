
// similar to https://www.shadertoy.com/view/ldl3W8

struct PushConstants
{
    float t;
    float d;
    uint s;
    uint pad;
}

[[vk::binding(0,0)]] StructuredBuffer<float4> palette;
[[vk::image_format("rgba8")]] RWTexture2D<float4> outImage;

// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
uint pcg(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Explicit fma instead of dot product - since there might be issues with precision issues due to inlining.
float2 hash(in float2 p, in uint h)
{
    p += float2(h & 0x3FFu, (h >> 10u) & 0x3FFu);
    p = float2(fma(p.x, 127.1, p.y * 311.7), fma(p.x, 269.5, p.y * 183.3));
    return fract(sin(p) * 18.5453);
}

// two 'pass' voronoi
float3 voronoiSDF(in float2 uv, in float t, in float d, in uint h)
{
    const float TWO_PI = 2.0 * float.getPi();

    float md = d;

    const float2 base = floor(uv * md);
    const float2 fract = frac(uv * md);

    float2 mc, mp;

    for (int i = -1; i <= 1; ++i)
    {
       for (int j = -1; j <= 1; ++j)
        {
            const float2 c = float2(i, j);
            float2 r = hash(base + c, h);
            r = float2(0.5) + float2(0.5) * sin(t + TWO_PI * r);
            const float2 p = c + r - fract;

            const float d = dot(p, p);

            if(d < md)
            {
                mp = p;
                mc = c;
                md = d;
            }
        }
    }

    md = d;
    for (int i = -2; i <= 2; ++i)
    {
        for (int j = -2; j <= 2; ++j)
        {
            const float2 c = mc + float2(i, j);
            float2 r = hash(base + c, h);
            r = float2(0.5) + float2(0.5) * sin(t + TWO_PI * r);
            const float2 p = c + r - fract;

            const float2 n = p - mp;
            const float nn = dot(n, n);
            if (nn > 1e-4)
            {
                const float d = dot(0.5 * (p + mp), n) * rsqrt(nn);
                md = min(md, d);
            }
        }
    }

    return float3(md, mp);
}


[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id: SV_DispatchThreadID, uniform PushConstants params)
{
    uint2 size;
    outImage.GetDimensions(size.x, size.y);
    if (any(id.xy >= size))
        return;
    const float px = params.d / float(size.x); // one pixel in cell units

    uint paletteSize, stride;
    palette.GetDimensions(paletteSize, stride);

    const float2 uv = float2(id.xy) / float2(size);

    const uint h = pcg(params.s);
    const float3 v = voronoiSDF(uv, params.t, params.d, h);

    const float s = saturate(v.x * 2.0) * float(paletteSize - 1) + 0.5;
    const uint i = min(uint(s), paletteSize - 1);
    const float w = 2.0 * float(paletteSize - 1) * px; // how much s changes over one pixel
    float4 color = lerp(palette[i], palette[min(i + 1, paletteSize - 1)], smoothstep(1.0 - w, 1.0, s - float(i)));

    color = lerp(float4(0, 0, 0, 1), color, smoothstep(0.02, 0.02 + px, length(v.yz))); // points

    outImage[id.xy] = color;
}
