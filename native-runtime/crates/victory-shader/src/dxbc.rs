//! The DXBC container: a header, then a table of FourCC-tagged parts.
//!
//! ```text
//! 0x00  "DXBC"
//! 0x04  digest[16]
//! 0x14  version major u16, minor u16
//! 0x18  total size u32
//! 0x1C  part count u32
//! 0x20  part offsets u32[count]
//! part: fourcc[4], size u32, data[size]
//! ```

use crate::ShaderError;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC(pub [u8; 4]);

impl FourCC {
    pub const DXIL: FourCC = FourCC(*b"DXIL");
    /// Native code from another compiler, in place of `DXIL`
    /// ([`crate::native::fspx_program`]).
    pub const ACCC: FourCC = FourCC(*b"ACCC");
    pub const STAT: FourCC = FourCC(*b"STAT");
    pub const RTS0: FourCC = FourCC(*b"RTS0");
    pub const PSV0: FourCC = FourCC(*b"PSV0");
    pub const ISG1: FourCC = FourCC(*b"ISG1");
    pub const OSG1: FourCC = FourCC(*b"OSG1");
    /// Per-primitive outputs of a mesh shader.
    pub const PSG1: FourCC = FourCC(*b"PSG1");
    pub const SFI0: FourCC = FourCC(*b"SFI0");
    pub const HASH: FourCC = FourCC(*b"HASH");
    pub const ILDB: FourCC = FourCC(*b"ILDB");
    pub const ILDN: FourCC = FourCC(*b"ILDN");
}

impl std::fmt::Debug for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::fmt::Display for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for &b in &self.0 {
            let c = if b.is_ascii_graphic() { b as char } else { '.' };
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Part<'a> {
    pub fourcc: FourCC,
    /// Offset of the part header within the container.
    pub offset: usize,
    pub data: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct Container<'a> {
    pub digest: [u8; 16],
    pub version: (u16, u16),
    pub parts: Vec<Part<'a>>,
}

const HEADER_SIZE: usize = 0x20;
const MAX_PARTS: usize = 256;

impl<'a> Container<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, ShaderError> {
        if data.len() < HEADER_SIZE || &data[0..4] != b"DXBC" {
            return Err(ShaderError::NotDxbc);
        }
        let digest = data[4..20].try_into().expect("16 bytes");
        let version = (u16_at(data, 0x14)?, u16_at(data, 0x16)?);
        let total = u32_at(data, 0x18)? as usize;
        if total > data.len() || total < HEADER_SIZE {
            return Err(ShaderError::Malformed("container size disagrees with data"));
        }
        let data = &data[..total];
        let count = u32_at(data, 0x1C)? as usize;
        if count > MAX_PARTS {
            return Err(ShaderError::Malformed("implausible part count"));
        }
        let mut parts = Vec::with_capacity(count);
        for i in 0..count {
            let offset = u32_at(data, HEADER_SIZE + 4 * i)? as usize;
            let fourcc = FourCC(
                data.get(offset..offset + 4)
                    .ok_or(ShaderError::Malformed("part header out of bounds"))?
                    .try_into()
                    .expect("4 bytes"),
            );
            let size = u32_at(data, offset + 4)? as usize;
            let body = data
                .get(offset + 8..offset + 8 + size)
                .ok_or(ShaderError::Malformed("part data out of bounds"))?;
            parts.push(Part {
                fourcc,
                offset,
                data: body,
            });
        }
        Ok(Container {
            digest,
            version,
            parts,
        })
    }

    pub fn part(&self, fourcc: FourCC) -> Option<&Part<'a>> {
        self.parts.iter().find(|p| p.fourcc == fourcc)
    }

    /// The name the compiler left in the container, if any: its `ILDN` part holds the file
    /// the program was compiled from (a path, whose last part names the program). Observed on
    /// this title's containers: `X:\americas\...\XboxBvhBuilder\RadixAlloc.pdb`, which is how
    /// the BVH builder's programs are told apart (`victory-gpu`'s `native::builder`).
    ///
    /// The part's data is a four-byte header (a zero dword, then the name's length as a
    /// `u16`), then the name, NUL-terminated and padded to a multiple of four.
    pub fn debug_name(&self) -> Option<&'a str> {
        let data = self.part(FourCC::ILDN)?.data;
        let len = usize::from(u16_at(data, 2).ok()?);
        let bytes = data.get(4..4 + len)?;
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        std::str::from_utf8(&bytes[..end]).ok().filter(|name| !name.is_empty())
    }
}

pub(crate) fn u16_at(data: &[u8], at: usize) -> Result<u16, ShaderError> {
    data.get(at..at + 2)
        .map(|b| u16::from_le_bytes(b.try_into().expect("2 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u32_at(data: &[u8], at: usize) -> Result<u32, ShaderError> {
    data.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u64_at(data: &[u8], at: usize) -> Result<u64, ShaderError> {
    data.get(at..at + 8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

/// Builds a container from parts (a test helper, and the host's for programs the console runtime
/// hands over as pieces).
pub fn build(parts: &[(FourCC, &[u8])]) -> Vec<u8> {
    let table = HEADER_SIZE + 4 * parts.len();
    let total = table + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    let mut offset = table;
    for (_, d) in parts {
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += 8 + d.len();
    }
    for (fourcc, d) in parts {
        out.extend_from_slice(&fourcc.0);
        out.extend_from_slice(&(d.len() as u32).to_le_bytes());
        out.extend_from_slice(d);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_parts() {
        let bytes = build(&[(FourCC::SFI0, &[1, 2, 3, 4]), (FourCC::HASH, &[9; 20])]);
        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.parts.len(), 2);
        assert_eq!(c.part(FourCC::SFI0).unwrap().data, &[1, 2, 3, 4]);
        assert_eq!(c.part(FourCC::HASH).unwrap().data.len(), 20);
        assert!(c.part(FourCC::DXIL).is_none());
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(Container::parse(b"nope").unwrap_err(), ShaderError::NotDxbc);
        let mut bytes = build(&[(FourCC::SFI0, &[0; 8])]);
        let last = bytes.len() - 1;
        bytes[0x20] = 0xFF; // corrupt the first part offset
        assert!(Container::parse(&bytes).is_err());
        bytes.truncate(last);
        assert!(Container::parse(&bytes).is_err());
    }

    /// The header's digest and version fields are parsed, not just skipped.
    #[test]
    fn parses_the_header_fields() {
        let mut bytes = build(&[(FourCC::SFI0, &[1, 2, 3, 4])]);
        bytes[4..20].copy_from_slice(&[7u8; 16]);
        bytes[0x14..0x16].copy_from_slice(&9u16.to_le_bytes());
        bytes[0x16..0x18].copy_from_slice(&3u16.to_le_bytes());
        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.digest, [7u8; 16]);
        assert_eq!(c.version, (9, 3));
        assert_eq!(c.parts.len(), 1);
    }

    /// Each header check refuses with its own message.
    #[test]
    fn rejects_implausible_headers() {
        // A declared total below the header itself.
        let mut bytes = build(&[(FourCC::SFI0, &[0; 8])]);
        bytes[0x18..0x1C].copy_from_slice(&0x10u32.to_le_bytes());
        assert!(matches!(
            Container::parse(&bytes),
            Err(ShaderError::Malformed("container size disagrees with data"))
        ));
        // More parts than the format allows.
        let mut bytes = build(&[]);
        bytes[0x1C..0x20].copy_from_slice(&257u32.to_le_bytes());
        assert!(matches!(
            Container::parse(&bytes),
            Err(ShaderError::Malformed("implausible part count"))
        ));
        // A part whose data runs past the container.
        let mut bytes = build(&[(FourCC::SFI0, &[0; 8])]);
        bytes[0x28..0x2C].copy_from_slice(&0xFFFFu32.to_le_bytes());
        assert!(matches!(
            Container::parse(&bytes),
            Err(ShaderError::Malformed("part data out of bounds"))
        ));
    }

    /// Bytes past the declared total are not part of the container, and a
    /// total larger than the data is refused.
    #[test]
    fn ignores_bytes_past_the_declared_total() {
        let mut bytes = build(&[(FourCC::SFI0, &[1, 2, 3, 4])]);
        let total = bytes.len();
        bytes.extend_from_slice(&[0xAA; 64]);
        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.parts.len(), 1);
        assert_eq!(c.part(FourCC::SFI0).unwrap().data, &[1, 2, 3, 4]);
        bytes[0x18..0x1C].copy_from_slice(&((total + 128) as u32).to_le_bytes());
        assert!(matches!(
            Container::parse(&bytes),
            Err(ShaderError::Malformed("container size disagrees with data"))
        ));
    }

    /// The `ILDN` part names the file the program was compiled from: a four-byte header
    /// (a zero dword, then the length as a `u16`), the NUL-terminated name, and padding.
    #[test]
    fn the_debug_name_is_read_from_the_ildn_part() {
        let name = b"X:\\americas\\Output\\XboxBvhBuilder\\RadixAlloc.pdb";
        let mut part = vec![0, 0];
        part.extend_from_slice(&(name.len() as u16 + 1).to_le_bytes());
        part.extend_from_slice(name);
        part.push(0);
        part.resize(part.len().next_multiple_of(4), 0);
        let bytes = build(&[(FourCC::ILDN, &part), (FourCC::SFI0, &[0; 8])]);
        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.debug_name(), Some("X:\\americas\\Output\\XboxBvhBuilder\\RadixAlloc.pdb"));
        // A container without one, and a name that runs past the part, have none.
        let plain = build(&[(FourCC::SFI0, &[0; 8])]);
        assert_eq!(Container::parse(&plain).unwrap().debug_name(), None);
        let short = build(&[(FourCC::ILDN, &[0, 0, 0xFF, 0x7F, b'a'])]);
        assert_eq!(Container::parse(&short).unwrap().debug_name(), None);
    }

    /// Two parts with the same fourcc both parse; `part` gives the first.
    #[test]
    fn the_first_part_with_a_fourcc_wins() {
        let bytes = build(&[(FourCC::RTS0, &[1]), (FourCC::RTS0, &[2, 2])]);
        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.parts.len(), 2);
        assert_eq!(c.part(FourCC::RTS0).unwrap().data, &[1]);
    }
}
