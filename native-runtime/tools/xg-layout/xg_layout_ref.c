// Reference generator for the texture layout/detile differential harness.
//
// Build on a Windows development PC with the Xbox Series X|S GDKX installed:
//
//   cl /nologo /TP /EHsc /I"%GameDK%\xbox\include\devpctools\gen9" xg_layout_ref.c ^
//      /link /LIBPATH:"%GameDK%\xbox\lib\devpctools\gen9" xg_xs.lib
//
// (Or with clang-cl and the same include/lib directories.) Copy xg_xs.dll beside the executable.
//
// On macOS the same source builds with MinGW, which links xg_xs.dll itself (it cannot use the
// MSVC import library), and runs under Wine:
//
//   x86_64-w64-mingw32-g++ -std=c++17 -O2 -Wno-unknown-pragmas -I <devpctools include dir> \
//       xg_layout_ref.c <devpctools bin dir>/xg_xs.dll -o xg_layout_ref.exe
//   WINEPREFIX=<fresh> WINEDLLOVERRIDES='winemac.drv=d;mscoree,mshtml=' WINEDEBUG=-all \
//       wine xg_layout_ref.exe anaconda.txt
//
// Then:
//
//   xg_layout_ref.exe <reference.txt>             Series X (Anaconda)
//   xg_layout_ref.exe --hw=lockhart <reference.txt>  Series S (Lockhart)
//
// The file is read by crates/victory-formats/examples/xg_layout_diff.rs, which compares it with
// victory_formats::texture. This source is compiled as C++ (/TP) because xg_xs.h declares the
// computer as a class; it uses the SDK's own header and library and copies nothing from them.
//
// Everything here is synthetic: descriptors, byte patterns and hashes. No game content is read or
// written. The output is generated data: keep it outside the repository.
//
// The text format is documented in tools/xg-layout/README.md. Coordinate units: texel queries use
// element coordinates (one element is a 4x4 block for a block-compressed format).

#if defined(_MSC_VER)
#include <xg_xs.h>
#else
// The SDK header declares its C entry points without extern "C", so a GNU-ABI compiler mangles
// them differently from the names xg_xs.dll exports. Wrapping the header gives those declarations
// C linkage (the class and its virtual calls are unaffected). MSVC builds use the SDK's own
// import library instead; MinGW builds link xg_xs.dll itself.
#include <windows.h>
extern "C" {
#include <xg_xs.h>
}
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- shared bytes

// The same pattern as `pattern` in crates/victory-formats/examples/xg_layout_diff.rs.
static unsigned char pattern_at(uint64_t i, uint64_t seed)
{
    uint64_t v = i * 0x9E3779B97F4A7C15ull + seed * 0x100000001B3ull;
    v ^= v >> 29;
    return (unsigned char)(v >> 32);
}

static void fill_pattern(unsigned char* data, size_t len, uint64_t seed)
{
    for (size_t i = 0; i < len; ++i)
        data[i] = pattern_at((uint64_t)i, seed);
}

static uint64_t fnv(const unsigned char* data, size_t len)
{
    uint64_t hash = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < len; ++i)
    {
        hash ^= (uint64_t)data[i];
        hash *= 0x100000001B3ull;
    }
    return hash;
}

// ---------------------------------------------------------------- cases

struct Case
{
    const char* Name;
    UINT Width;
    UINT Height;
    UINT ArraySize;
    UINT MipLevels; // 0: the full chain
    UINT Format;    // XG_FORMAT (= DXGI format number)
    UINT Swizzle;   // XG_SWIZZLE_MODE
};

// Original synthetic cases: NPOT block-compressed textures, packed mip tails, arrays and every
// GFX10 swizzle mode.
static const Case CASES[] =
{
    { "bc7_npot_130x68_64kb_d",            130,  68, 1, 0,  98, 10 },
    { "bc7_npot_130x68_64kb_s",            130,  68, 1, 0,  98,  9 },
    { "bc1_npot_100x60_4kb_s",             100,  60, 1, 0,  71,  5 },
    { "bc1_npot_100x60_4kb_d",             100,  60, 1, 0,  71,  6 },
    { "bc4_npot_67x43_64kb_d",              67,  43, 1, 0,  80, 10 },
    { "bc5_npot_250x175_4kb_s",            250, 175, 1, 0,  83,  5 },
    { "bc7_256x256_64kb_d_full_tail",      256, 256, 1, 0,  98, 10 },
    { "bc7_256x256_64kb_d_mips3",          256, 256, 1, 3,  98, 10 },
    { "bc7_192x192_4kb_s_full",            192, 192, 1, 0,  98,  5 },
    { "bc1_512x512_64kb_s_full",           512, 512, 1, 0,  71,  9 },
    { "rgba8_npot_257x129_64kb_d",         257, 129, 1, 0,  28, 10 },
    { "rgba8_300x200_4kb_d",               300, 200, 1, 0,  28,  6 },
    { "r8_128x4_4kb_s",                    128,   4, 1, 0,  61,  5 },
    { "r16_520x300_64kb_s",                520, 300, 1, 0,  56,  9 },
    { "rgba16f_33x17_4kb_d_mips1",          33,  17, 1, 1,  10,  6 },
    { "bc7_array3_128x128_64kb_d",         128, 128, 3, 0,  98, 10 },
    { "bc1_array4_130x68_4kb_s",           130,  68, 4, 0,  71,  5 },
    { "rgba8_array2_256x256_64kb_s",       256, 256, 2, 0,  28,  9 },
    { "bc7_130x68_64kb_s_t",               130,  68, 1, 0,  98, 17 },
    { "bc7_130x68_64kb_d_t",               130,  68, 1, 0,  98, 18 },
    { "bc7_130x68_4kb_s_x",                130,  68, 1, 0,  98, 21 },
    { "bc7_130x68_4kb_d_x",                130,  68, 1, 0,  98, 22 },
    { "d32_130x68_64kb_z_x",               130,  68, 1, 0,  40, 24 },
    { "bc7_130x68_64kb_s_x",               130,  68, 1, 0,  98, 25 },
    { "bc7_130x68_64kb_d_x",               130,  68, 1, 0,  98, 26 },
    { "bc7_130x68_64kb_r_x",               130,  68, 1, 0,  98, 27 },
    { "rgba8_128x128_linear",              128, 128, 1, 0,  28,  0 },
    { "rgba8_300x200_64kb_s_x_array2",     300, 200, 2, 0,  28, 25 },
    { "bc7_64x64_256b_s_full",              64,  64, 1, 0,  98,  1 },
};

static UINT full_mips(UINT width, UINT height)
{
    UINT largest = width > height ? width : height;
    UINT mips = 1;
    while (largest > 1)
    {
        largest >>= 1;
        ++mips;
    }
    return mips;
}

// ---------------------------------------------------------------- output

static void emit_case(FILE* out, const Case& c, UINT mips)
{
    XG_TEXTURE2D_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.Width = c.Width;
    desc.Height = c.Height;
    desc.MipLevels = mips;
    desc.ArraySize = c.ArraySize;
    desc.Format = (XG_FORMAT)c.Format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = XG_USAGE_DEFAULT;
    desc.BindFlags = XG_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;
    desc.SwizzleMode = (XG_SWIZZLE_MODE)c.Swizzle;
    desc.Pitch = 0;

    fprintf(out, "case %s\n", c.Name);
    fprintf(out, "desc dim=2d w=%u h=%u array=%u mips=%u fmt=%u samples=1 swz=%u\n",
            c.Width, c.Height, c.ArraySize, mips, c.Format, c.Swizzle);

    XGTextureAddressComputer* computer = NULL;
    HRESULT hr = XGCreateTexture2DComputer(&desc, &computer);
    if (FAILED(hr) || computer == NULL)
    {
        fprintf(out, "status fail hr=0x%08x\nend\n", (unsigned)hr);
        if (computer != NULL)
            computer->Release();
        return;
    }

    XG_RESOURCE_LAYOUT layout;
    memset(&layout, 0, sizeof(layout));
    hr = computer->GetResourceLayout(&layout);
    if (FAILED(hr))
    {
        fprintf(out, "status fail hr=0x%08x\nend\n", (unsigned)hr);
        computer->Release();
        return;
    }
    fprintf(out, "status ok\n");
    fprintf(out, "res size=%llu align=%llu mips=%u planes=%u\n",
            (unsigned long long)layout.SizeBytes,
            (unsigned long long)layout.BaseAlignmentBytes,
            layout.MipLevels, layout.Planes);

    for (UINT p = 0; p < layout.Planes && p < 4; ++p)
    {
        const XG_PLANE_LAYOUT& plane = layout.Plane[p];
        fprintf(out,
                "plane i=%u usage=%u size=%llu slice=%llu offset=%llu align=%llu bpe=%u std=%u tail=%u\n",
                p, (unsigned)plane.Usage,
                (unsigned long long)plane.SizeBytes,
                (unsigned long long)plane.SliceSizeBytes,
                (unsigned long long)plane.BaseOffsetBytes,
                (unsigned long long)plane.BaseAlignmentBytes,
                plane.BytesPerElement, plane.StandardMipCount, plane.MipTailMipCount);
        for (UINT m = 0; m < layout.MipLevels && m < 15; ++m)
        {
            const XG_MIPLEVEL_LAYOUT& mip = plane.MipLayout[m];
            fprintf(out,
                    "mip plane=%u i=%u size=%llu offset=%llu slice=%llu pitch_px=%u pitch_b=%u "
                    "align=%u pw=%u ph=%u pd=%u ew=%u eh=%u wp=%u hp=%u doa=%u samples=%u swz=%u "
                    "bank0=%llu bank1=%llu slice_depth=%u tail=%u tx=%u ty=%u\n",
                    p, m,
                    (unsigned long long)mip.SizeBytes,
                    (unsigned long long)mip.OffsetBytes,
                    (unsigned long long)mip.Slice2DSizeBytes,
                    mip.PitchPixels, mip.PitchBytes, mip.AlignmentBytes,
                    mip.PaddedWidthElements, mip.PaddedHeightElements, mip.PaddedDepthOrArraySize,
                    mip.WidthElements, mip.HeightElements, mip.WidthPixels, mip.HeightPixels,
                    mip.DepthOrArraySize, mip.SampleCount, (unsigned)mip.SwizzleMode,
                    (unsigned long long)mip.BankRotationAddressBitMask,
                    (unsigned long long)mip.BankRotationBytesPerSlice,
                    mip.SliceDepthElements, mip.IsMipTail,
                    mip.MipTailOffsetElementsX, mip.MipTailOffsetElementsY);
        }
    }

    // Texel queries: the corners and middle of every mip, on the first and last array slice.
    for (UINT m = 0; m < layout.MipLevels && m < 15; ++m)
    {
        const XG_MIPLEVEL_LAYOUT& mip = layout.Plane[0].MipLayout[m];
        UINT w = mip.WidthElements;
        UINT h = mip.HeightElements;
        if (w == 0 || h == 0)
            continue;
        const UINT xs[3] = { 0, w - 1, w / 2 };
        const UINT ys[3] = { 0, h - 1, h / 2 };
        for (UINT zi = 0; zi < 2; ++zi)
        {
            UINT z = zi == 0 ? 0 : c.ArraySize - 1;
            if (zi == 1 && c.ArraySize <= 1)
                continue;
            for (UINT yi = 0; yi < 3; ++yi)
            {
                for (UINT xi = 0; xi < 3; ++xi)
                {
                    UINT32 bit = 0;
                    UINT64 offset = computer->GetTexelElementOffsetBytes(
                        0, m, xs[xi], ys[yi], z, 0, &bit);
                    fprintf(out, "texel plane=0 mip=%u x=%u y=%u z=%u off=%llu bit=%u\n",
                            m, xs[xi], ys[yi], z, (unsigned long long)offset, bit);
                }
            }
        }
    }

    // Copies: every mip of the first and last array slice, out of and into tiled memory.
    size_t size = (size_t)layout.SizeBytes;
    unsigned char* tiled = (unsigned char*)malloc(size);
    if (tiled == NULL)
    {
        fprintf(out, "end\n");
        computer->Release();
        return;
    }
    for (UINT zi = 0; zi < 2; ++zi)
    {
        UINT slice = zi == 0 ? 0 : c.ArraySize - 1;
        if (zi == 1 && c.ArraySize <= 1)
            continue;
        for (UINT m = 0; m < layout.MipLevels && m < 15; ++m)
        {
            const XG_MIPLEVEL_LAYOUT& mip = layout.Plane[0].MipLayout[m];
            UINT bytes = layout.Plane[0].BytesPerElement;
            UINT row = mip.WidthElements * bytes + 16;
            UINT depth = row * mip.HeightElements + 32;
            UINT len = (mip.HeightElements - 1) * row + mip.WidthElements * bytes;
            UINT subresource = m + slice * layout.MipLevels;

            fill_pattern(tiled, size, 1);
            unsigned char* linear = (unsigned char*)calloc(len, 1);
            if (linear == NULL)
                break;
            HRESULT from_hr = computer->CopyFromSubresource(
                linear, 0, subresource, 0, tiled, row, depth);
            uint64_t from = SUCCEEDED(from_hr) ? fnv(linear, len) : 0;
            free(linear);

            fill_pattern(tiled, size, 2);
            linear = (unsigned char*)malloc(len);
            if (linear == NULL)
                break;
            fill_pattern(linear, len, 3);
            HRESULT into_hr = computer->CopyIntoSubresource(
                tiled, 0, subresource, linear, row, depth);
            uint64_t into = SUCCEEDED(into_hr) ? fnv(tiled, size) : 0;
            free(linear);

            fprintf(out,
                    "copy plane=0 sub=%u row=%u depth=%u from=%llu into=%llu len=%u "
                    "from_hr=0x%08x into_hr=0x%08x\n",
                    subresource, row, depth,
                    (unsigned long long)from, (unsigned long long)into, len,
                    (unsigned)from_hr, (unsigned)into_hr);
        }
    }
    free(tiled);

    // Packed mip tails and the standard tile shape, for the 64 KiB modes.
    if (layout.Plane[0].BaseAlignmentBytes == 65536)
    {
        UINT32 tiles = 0;
        XG_PACKED_MIP_DESC packed;
        XG_TILE_SHAPE shape;
        memset(&packed, 0, sizeof(packed));
        memset(&shape, 0, sizeof(shape));
        UINT32 count = 1;
        XG_SUBRESOURCE_TILING tiling;
        memset(&tiling, 0, sizeof(tiling));
        hr = computer->GetResourceTiling(&tiles, &packed, &shape, &count, 0, &tiling);
        if (SUCCEEDED(hr))
        {
            char starts[16] = "";
            if (count > 0)
                snprintf(starts, sizeof(starts), "%u", tiling.StartTileIndexInOverallResource);
            fprintf(out,
                    "tiling tiles=%u std=%u packed=%u ptiles=%u start=%u tw=%u th=%u td=%u n=%u starts=%s\n",
                    tiles, (unsigned)packed.NumStandardMips, (unsigned)packed.NumPackedMips,
                    packed.NumTilesForPackedMips, packed.StartTileIndexInOverallResource,
                    shape.WidthInTexels, shape.HeightInTexels, shape.DepthInTexels,
                    count, starts);
        }
    }

    fprintf(out, "end\n");
    computer->Release();
}

int main(int argc, char** argv)
{
    const char* path = NULL;
    int lockhart = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (strncmp(argv[i], "--hw=", 5) == 0)
            lockhart = strcmp(argv[i] + 5, "lockhart") == 0;
        else if (strncmp(argv[i], "--", 2) == 0)
        {
            fprintf(stderr, "unknown switch %s\n", argv[i]);
            return 2;
        }
        else
            path = argv[i];
    }
    if (path == NULL)
    {
        fprintf(stderr, "usage: xg_layout_ref [--hw=anaconda|lockhart] <reference.txt>\n");
        return 2;
    }

    XGSetHardwareVersion(lockhart ? XG_HARDWARE_VERSION_XBOX_SCARLETT_LOCKHART
                                  : XG_HARDWARE_VERSION_XBOX_SCARLETT_ANACONDA);

    FILE* out = fopen(path, "wb");
    if (out == NULL)
    {
        fprintf(stderr, "cannot write %s\n", path);
        return 2;
    }
    fprintf(out, "xg-layout-ref 1\nhw %s\n", lockhart ? "lockhart" : "anaconda");
    for (const Case& c : CASES)
        emit_case(out, c, c.MipLevels == 0 ? full_mips(c.Width, c.Height) : c.MipLevels);
    fclose(out);
    printf("wrote %s (%zu cases)\n", path, sizeof(CASES) / sizeof(CASES[0]));
    return 0;
}
