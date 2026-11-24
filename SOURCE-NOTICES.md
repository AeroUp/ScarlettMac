# Source provenance

The patch applies to [XCL Mac at d0d9889](https://github.com/utkarshxdev/xcl-mac/commit/d0d9889514e94195ebcdbf86e23a60f4f5756189), which builds on [XCL](https://github.com/ivall/xcl). XCL copyrights and notices remain in the source produced by `scripts/build.sh --prepare`. The patch records changes made on 24 November 2026.

The BCPack decoder and generated tables come from [XCL PR #905](https://github.com/ivall/xcl/pull/905). The VIctory author granted permission for this material to be used under GPL-2.0-only. The input for the generated tables is the [pinned VIctory data file](https://github.com/Marvisak/VIctory/blob/f5660fdcfdd6eb5cb6495be47a23af9aa30f5617/crates/victory-formats/src/bcpack-tables.bin). The rANS model data originates from Microsoft's [silicon-rans sample](https://github.com/microsoft/silicon-rans). See [its license](LICENSE-silicon-rans).

The BCPack test vectors are synthetic and can be reproduced with the generator in the patch. The upstream `THIRD_PARTY.md` lists the original dependency notices.
