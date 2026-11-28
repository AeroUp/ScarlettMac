# BCPack reference fixture generator

A standalone synthetic-case generator for the GDK's `BCPack.dll`. It builds
deterministic BC1-BC7 block streams, calls `BCPackCompress` and
`BCPackDecompress`, checks that the round trip is exact, and writes
`<name>.bc<n>` beside `<name>.out`. Nothing in a case comes from a title.

`BCPack.dll` is not part of this repository. The generator loads it by path,
so use the copy from your installed GDK:

```bash
x86_64-w64-mingw32-gcc -O2 -Wall -Wextra \
  -o bcpack_fixtures.exe tools/bcpack-fixtures/bcpack_fixtures.c

WINEPREFIX=<scratch>/wineprefix \
WINEDLLOVERRIDES='winemac.drv=d;mscoree,mshtml=' \
WINEDEBUG=-all \
  wine bcpack_fixtures.exe <GDK>/BCPack.dll <scratch>/cases
```

The compressor in the current GDK build returns `E_FAIL` for streams whose
packed form does not shrink (small or high-entropy inputs). The generated
cases use texture-like palette blocks over 128 or 1000 blocks so every format
packs successfully.

Run the native reference check against the cases:

```bash
VICTORY_BCPACK_TABLES=crates/victory-formats/src/bcpack-tables.bin \
VICTORY_BCPACK_CASES=<scratch>/cases \
  cargo test -p victory-formats reference_chunks -- --ignored --nocapture
```

Cases, tables, the DLL and the Wine prefix all stay outside the repository.
