//! The `.wandb` container: a LevelDB log (32 KiB blocks; records of
//! `crc32: u32 LE | len: u16 LE | type: u8 | data`, split FIRST/MIDDLE/LAST
//! across blocks, a block tail shorter than 7 bytes zero-padded) behind a
//! 7-byte file header `":W&B" | 0xBEE1 u16 LE | version 0`. As in W&B's
//! `datastore.py`, the checksum is zlib's CRC-32 (IEEE) of `type || data`,
//! not LevelDB's masked CRC-32C.

use std::io::{self, Read, Write};

pub const HEADER_LEN: usize = 7;
pub const BLOCK_LEN: usize = 32768;
pub const DATA_LEN: usize = BLOCK_LEN - HEADER_LEN;
pub const IDENT: &[u8; 4] = b":W&B";
pub const MAGIC: u16 = 0xBEE1;
pub const VERSION: u8 = 0;

const FULL: u8 = 1;
const FIRST: u8 = 2;
const MIDDLE: u8 = 3;
const LAST: u8 = 4;

fn crc_table() -> &'static [u32; 256] {
    static T: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *e = c;
        }
        t
    })
}

/// zlib-compatible CRC-32 continuing from `crc`.
pub fn crc32(crc: u32, data: &[u8]) -> u32 {
    let t = crc_table();
    let mut c = !crc;
    for &b in data {
        c = t[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

fn record_crc(kind: u8, data: &[u8]) -> u32 {
    crc32(crc32(0, &[kind]), data)
}

pub struct LogWriter<W: Write> {
    w: W,
    index: usize,
}

impl<W: Write> LogWriter<W> {
    pub fn new(mut w: W) -> io::Result<Self> {
        w.write_all(IDENT)?;
        w.write_all(&MAGIC.to_le_bytes())?;
        w.write_all(&[VERSION])?;
        Ok(LogWriter { w, index: HEADER_LEN })
    }

    fn record(&mut self, kind: u8, data: &[u8]) -> io::Result<()> {
        debug_assert!(data.len() + HEADER_LEN <= BLOCK_LEN - self.index % BLOCK_LEN);
        self.w.write_all(&record_crc(kind, data).to_le_bytes())?;
        self.w.write_all(&(data.len() as u16).to_le_bytes())?;
        self.w.write_all(&[kind])?;
        self.w.write_all(data)?;
        self.index += HEADER_LEN + data.len();
        Ok(())
    }

    /// Append one payload, splitting it across blocks exactly as
    /// `DataStore._write_data` does.
    pub fn write(&mut self, data: &[u8]) -> io::Result<()> {
        let mut space = BLOCK_LEN - self.index % BLOCK_LEN;
        if space < HEADER_LEN {
            self.w.write_all(&vec![0u8; space])?;
            self.index += space;
            space = BLOCK_LEN;
        }
        if data.len() + HEADER_LEN <= space {
            return self.record(FULL, data);
        }
        let room = space - HEADER_LEN;
        self.record(FIRST, &data[..room])?;
        let mut used = room;
        while data.len() - used > DATA_LEN {
            self.record(MIDDLE, &data[used..used + DATA_LEN])?;
            used += DATA_LEN;
        }
        self.record(LAST, &data[used..])
    }

    pub fn into_inner(mut self) -> io::Result<W> {
        self.w.flush()?;
        Ok(self.w)
    }
}

#[derive(Debug)]
pub enum LogError {
    Io(io::Error),
    BadHeader,
    Checksum { offset: usize },
    Framing { offset: usize, kind: u8 },
    BadPadding { offset: usize },
}

impl std::fmt::Display for LogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogError::Io(e) => write!(f, "io: {e}"),
            LogError::BadHeader => write!(f, "not a .wandb file (bad header)"),
            LogError::Checksum { offset } => write!(f, "checksum mismatch at byte {offset}"),
            LogError::Framing { offset, kind } => write!(f, "unexpected record type {kind} at byte {offset}"),
            LogError::BadPadding { offset } => write!(f, "non-zero block padding at byte {offset}"),
        }
    }
}
impl std::error::Error for LogError {}

/// Read every payload. A truncated final record (a writer that died
/// mid-append) ends the scan without error, like W&B's reader at EOF.
pub fn read_all(mut r: impl Read) -> Result<Vec<Vec<u8>>, LogError> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf).map_err(LogError::Io)?;
    if buf.len() < HEADER_LEN || &buf[..4] != IDENT || u16::from_le_bytes([buf[4], buf[5]]) != MAGIC || buf[6] != VERSION {
        return Err(LogError::BadHeader);
    }
    let mut i = HEADER_LEN;
    let mut out = Vec::new();
    let mut partial: Option<Vec<u8>> = None;
    loop {
        let space = BLOCK_LEN - i % BLOCK_LEN;
        if space < HEADER_LEN {
            let end = (i + space).min(buf.len());
            if buf[i..end].iter().any(|&b| b != 0) {
                return Err(LogError::BadPadding { offset: i });
            }
            i += space;
        }
        if i + HEADER_LEN > buf.len() {
            break;
        }
        let crc = u32::from_le_bytes(buf[i..i + 4].try_into().unwrap());
        let len = u16::from_le_bytes([buf[i + 4], buf[i + 5]]) as usize;
        let kind = buf[i + 6];
        if i + HEADER_LEN + len > buf.len() {
            break; // torn tail
        }
        let data = &buf[i + HEADER_LEN..i + HEADER_LEN + len];
        if record_crc(kind, data) != crc {
            return Err(LogError::Checksum { offset: i });
        }
        match (kind, partial.as_mut()) {
            (FULL, None) => out.push(data.to_vec()),
            (FIRST, None) => partial = Some(data.to_vec()),
            (MIDDLE, Some(p)) => p.extend_from_slice(data),
            (LAST, Some(p)) => {
                p.extend_from_slice(data);
                out.push(partial.take().unwrap());
            }
            _ => return Err(LogError::Framing { offset: i, kind }),
        }
        i += HEADER_LEN + len;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_matches_zlib() {
        // zlib.crc32(b"123456789") == 0xCBF43926
        assert_eq!(crc32(0, b"123456789"), 0xCBF4_3926);
        // the header magic: zlib.crc32(b"Weights & Biases") & 0xffff
        assert_eq!(crc32(0, b"Weights & Biases") & 0xffff, MAGIC as u32);
    }

    #[test]
    fn roundtrip_across_blocks() {
        let mut w = LogWriter::new(Vec::new()).unwrap();
        let payloads: Vec<Vec<u8>> = (0..300).map(|i| vec![(i % 251) as u8; (i * 397) % 9000]).chain([vec![7u8; 100_000], vec![], vec![1u8; DATA_LEN - 3]]).collect();
        for p in &payloads {
            w.write(p).unwrap();
        }
        let bytes = w.into_inner().unwrap();
        assert!(bytes.len() > 3 * BLOCK_LEN);
        assert_eq!(read_all(&bytes[..]).unwrap(), payloads);
        // torn tail: the last payload is dropped, the rest survive
        let cut = &bytes[..bytes.len() - 5];
        let got = read_all(cut).unwrap();
        assert_eq!(&got[..], &payloads[..payloads.len() - 1]);
        // corruption is detected
        let mut bad = bytes.clone();
        bad[HEADER_LEN + 9] ^= 0xff;
        assert!(matches!(read_all(&bad[..]), Err(LogError::Checksum { .. })));
    }
}
