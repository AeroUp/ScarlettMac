// Original test shader for tools/shader-diff. Comparison sampling of a depth
// texture: level zero, implicit derivatives, a constant texel offset, a
// comparison gather, and a depth array. These are the console forms the
// translator's image_sample_c_lz / image_sample_c / image_gather4_c_lz paths
// have to carry. No game data.

Texture2D<float> Depth : register(t0);
Texture2DArray<float> DepthArray : register(t1);
SamplerComparisonState Shadow : register(s0);

[RootSignature("DescriptorTable(SRV(t0), SRV(t1)), DescriptorTable(Sampler(s0))")]
float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target
{
    float z = saturate(pos.z);
    float2 p = uv;

    float level_zero = Depth.SampleCmpLevelZero(Shadow, p, z);
    float implicit_lod = Depth.SampleCmp(Shadow, p, z);
    float offset = Depth.SampleCmpLevelZero(Shadow, p, z, int2(1, -1));
    float4 gather = Depth.GatherCmpRed(Shadow, p, z);
    float slice = DepthArray.SampleCmpLevelZero(Shadow, float3(p, 0), z);

    return float4(level_zero + slice, implicit_lod, offset, gather.x + gather.y);
}
