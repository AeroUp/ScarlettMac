// Original test shader for tools/shader-diff. Depth and resource reads: a
// float depth texture, a typed integer stencil view, and a raw buffer, all
// read by index and written to a structured UAV. These are the console's
// image_load and buffer_load forms. No game data.

Texture2D<float> Depth : register(t0);
Texture2D<uint2> Stencil : register(t1);
Buffer<uint4> Raw : register(t2);
RWStructuredBuffer<uint4> Out : register(u0);

[RootSignature("DescriptorTable(SRV(t0), SRV(t1), SRV(t2)), DescriptorTable(UAV(u0))")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint2 p = id.xy;

    float d = Depth.Load(int3(p, 0));
    uint2 s = Stencil.Load(int3(p, 0));
    uint4 r = Raw.Load(id.x);

    Out[id.y * 8 + id.x] = uint4(asuint(d), s.x, s.y, r.x);
}
