// Original test shader for tools/shader-diff. A comparison gather and a
// comparison sample from a compute shader, where the console uses
// image_gather4_c_lz and image_sample_c_lz with a table descriptor. No game
// data.

Texture2D<float> Shadow : register(t0);
SamplerComparisonState ShadowCmp : register(s0);
RWStructuredBuffer<float> Out : register(u0);

[RootSignature("DescriptorTable(SRV(t0)), DescriptorTable(Sampler(s0)), DescriptorTable(UAV(u0))")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    float2 uv = (id.xy + 0.5) / 64.0;
    float z = id.x / 64.0;

    float4 gather = Shadow.GatherCmpRed(ShadowCmp, uv, z, int2(1, -1));
    float sample = Shadow.SampleCmpLevelZero(ShadowCmp, uv, z);

    Out[id.y * 8 + id.x] = gather.x + gather.y + gather.z + gather.w + sample;
}
