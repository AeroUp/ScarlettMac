// Standalone synthetic-case generator for the GDK's BCPack reference codec.
//
// It builds deterministic BCn block streams, packs each with BCPackCompress
// from BCPack.dll, unpacks it with BCPackDecompress, checks the round trip,
// and writes <name>.bc<n> beside <name>.out for the native reference test.
// Nothing here comes from a title.
//
// The compressor refuses a stream whose packed form does not shrink (E_FAIL),
// so the cases are texture-like palette blocks over enough blocks for the
// history and dictionaries to pay off.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef long HRESULT;
typedef HRESULT(__cdecl *bcpack_compress)(
    size_t bc, size_t *packed_size, void *out, size_t out_size,
    const void *in, size_t in_size);
typedef HRESULT(__cdecl *bcpack_decompress)(
    size_t bc, void *out, size_t out_size, const void *packed, size_t packed_size);

#define OK 0

static const unsigned char palette[16] = {
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
};

static size_t block_bytes(int bc)
{
    return (bc == 1 || bc == 4) ? 8 : 16;
}

static void fill_blocks(int bc, size_t blocks, unsigned phase, unsigned char *out)
{
    size_t n = block_bytes(bc);
    for (size_t i = 0; i < blocks; i++) {
        unsigned char *block = out + i * n;
        if (i > 0 && i % 7 == 0) {
            // Every seventh block repeats the previous one, so the codec has
            // back references to encode as well as fresh residuals.
            memcpy(block, out + (i - 1) * n, n);
            continue;
        }
        for (size_t k = 0; k < n; k++)
            block[k] = palette[(i * 3 + k * 5 + phase) & 15];
        if (bc == 6) {
            // Mode 1 (2-bit code 0) on even blocks, mode 11 (5-bit code 3) on odd.
            block[0] = (i & 1) ? (unsigned char)((block[0] & 0xe0) | 3)
                               : (unsigned char)(block[0] & 0xfc);
        } else if (bc == 7) {
            // Mode 6: seven mode bits, value 1 << 6.
            block[0] = (unsigned char)((block[0] & 0x80) | 0x40);
        }
    }
}

static int write_file(const char *dir, const char *name, const char *ext,
                      const void *data, size_t size)
{
    char path[4096];
    if (snprintf(path, sizeof path, "%s/%s.%s", dir, name, ext) >= (int)sizeof path)
        return 0;
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot create %s\n", path);
        return 0;
    }
    int ok = fwrite(data, 1, size, f) == size;
    ok &= fclose(f) == 0;
    if (!ok)
        fprintf(stderr, "cannot write %s\n", path);
    return ok;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <BCPack.dll> <output folder>\n", argv[0]);
        return 2;
    }
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) {
        fprintf(stderr, "LoadLibrary(%s) failed: %lu\n", argv[1], GetLastError());
        return 1;
    }
    bcpack_compress compress = (bcpack_compress)(void *)GetProcAddress(dll, "BCPackCompress");
    bcpack_decompress decompress =
        (bcpack_decompress)(void *)GetProcAddress(dll, "BCPackDecompress");
    if (!compress || !decompress) {
        fprintf(stderr, "BCPackCompress/BCPackDecompress not found\n");
        return 1;
    }
    struct {
        const char *name;
        int bc;
        size_t blocks;
        unsigned phase;
    } cases[] = {
        {"bc1-a", 1, 128, 0},   {"bc1-b", 1, 1000, 3},
        {"bc3-a", 3, 128, 1},   {"bc3-b", 3, 1000, 4},
        {"bc4-a", 4, 128, 2},   {"bc4-b", 4, 1000, 5},
        {"bc5-a", 5, 128, 3},   {"bc5-b", 5, 1000, 6},
        {"bc6-a", 6, 128, 4},   {"bc6-b", 6, 1000, 7},
        {"bc7-a", 7, 128, 5},   {"bc7-b", 7, 1000, 8},
    };
    int failures = 0;
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        int bc = cases[c].bc;
        size_t raw_size = cases[c].blocks * block_bytes(bc);
        size_t packed_cap = raw_size * 2 + (1u << 20);
        unsigned char *raw = malloc(raw_size);
        unsigned char *packed = malloc(packed_cap);
        unsigned char *out = malloc(raw_size);
        if (!raw || !packed || !out) {
            fprintf(stderr, "%s: out of memory\n", cases[c].name);
            return 1;
        }
        fill_blocks(bc, cases[c].blocks, cases[c].phase, raw);
        size_t packed_size = 0;
        HRESULT hr = compress(bc, &packed_size, packed, packed_cap, raw, raw_size);
        if (hr != OK || packed_size == (size_t)-1) {
            fprintf(stderr, "%s: BCPackCompress hr=0x%08lx\n", cases[c].name, (unsigned long)hr);
            failures++;
            free(raw);
            free(packed);
            free(out);
            continue;
        }
        hr = decompress(bc, out, raw_size, packed, packed_size);
        int same = hr == OK && memcmp(out, raw, raw_size) == 0;
        if (!same) {
            fprintf(stderr, "%s: BCPackDecompress hr=0x%08lx, round trip %s\n",
                    cases[c].name, (unsigned long)hr, hr == OK ? "mismatch" : "failed");
            failures++;
        } else {
            char bc_ext[8];
            snprintf(bc_ext, sizeof bc_ext, "bc%d", bc);
            int wrote = write_file(argv[2], cases[c].name, bc_ext, packed, packed_size);
            wrote &= write_file(argv[2], cases[c].name, "out", out, raw_size);
            if (!wrote)
                failures++;
            else
                printf("%-7s bc%d %4zu blocks %5zu -> %5zu bytes\n",
                       cases[c].name, bc, cases[c].blocks, raw_size, packed_size);
        }
        free(raw);
        free(packed);
        free(out);
    }
    if (failures) {
        fprintf(stderr, "%d case(s) failed\n", failures);
        return 1;
    }
    return 0;
}
