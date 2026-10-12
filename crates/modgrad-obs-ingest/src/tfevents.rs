//! TensorBoard event files (`events.out.tfevents.*`): TFRecord framing
//!
//! ```text
//! u64 LE length | u32 LE masked CRC32C(length) | data | u32 LE masked CRC32C(data)
//! ```
//!
//! around `tensorflow.Event` protobufs. Scalars come from
//! `Summary.Value.simple_value` (torch `SummaryWriter`, TF1) or from a
//! one-element `tensor` (TF2 `tf.summary.scalar`, torch `new_style=True`):
//! `tensor_content` bytes, or `float_val` / `double_val` / `int_val` /
//! `int64_val` / `half_val`. Histograms, images, audio and text are counted
//! and skipped. A bad CRC or a truncated tail (a file still being
//! written) stops reading there and is reported; what came before is kept.
//! Field numbers follow tensorflow/core/util/event.proto and
//! framework/{summary,tensor,tensor_shape}.proto (Apache-2.0).

use crate::{FileReport, LocalRun};
use prost::Message;

#[derive(Clone, PartialEq, prost::Message)]
pub struct Event {
    #[prost(double, tag = "1")]
    pub wall_time: f64,
    #[prost(int64, tag = "2")]
    pub step: i64,
    #[prost(string, optional, tag = "3")]
    pub file_version: Option<String>,
    #[prost(message, optional, tag = "5")]
    pub summary: Option<Summary>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Summary {
    #[prost(message, repeated, tag = "1")]
    pub value: Vec<SummaryValue>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct SummaryValue {
    #[prost(string, tag = "1")]
    pub tag: String,
    #[prost(float, optional, tag = "2")]
    pub simple_value: Option<f32>,
    #[prost(bytes = "vec", optional, tag = "3")]
    pub obsolete_old_style_histogram: Option<Vec<u8>>,
    #[prost(bytes = "vec", optional, tag = "4")]
    pub image: Option<Vec<u8>>,
    #[prost(bytes = "vec", optional, tag = "5")]
    pub histo: Option<Vec<u8>>,
    #[prost(bytes = "vec", optional, tag = "6")]
    pub audio: Option<Vec<u8>>,
    #[prost(message, optional, tag = "8")]
    pub tensor: Option<TensorProto>,
    #[prost(message, optional, tag = "9")]
    pub metadata: Option<SummaryMetadata>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct SummaryMetadata {
    #[prost(message, optional, tag = "1")]
    pub plugin_data: Option<PluginData>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct PluginData {
    #[prost(string, tag = "1")]
    pub plugin_name: String,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct TensorProto {
    #[prost(int32, tag = "1")]
    pub dtype: i32,
    #[prost(message, optional, tag = "2")]
    pub tensor_shape: Option<TensorShape>,
    #[prost(bytes = "vec", tag = "4")]
    pub tensor_content: Vec<u8>,
    #[prost(float, repeated, tag = "5")]
    pub float_val: Vec<f32>,
    #[prost(double, repeated, tag = "6")]
    pub double_val: Vec<f64>,
    #[prost(int32, repeated, tag = "7")]
    pub int_val: Vec<i32>,
    #[prost(int64, repeated, tag = "10")]
    pub int64_val: Vec<i64>,
    #[prost(int32, repeated, tag = "13")]
    pub half_val: Vec<i32>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct TensorShape {
    #[prost(message, repeated, tag = "2")]
    pub dim: Vec<Dim>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct Dim {
    #[prost(int64, tag = "1")]
    pub size: i64,
}

const DT_FLOAT: i32 = 1;
const DT_DOUBLE: i32 = 2;
const DT_INT32: i32 = 3;
const DT_UINT8: i32 = 4;
const DT_INT16: i32 = 5;
const DT_INT8: i32 = 6;
const DT_INT64: i32 = 9;
const DT_BOOL: i32 = 10;
const DT_BFLOAT16: i32 = 14;
const DT_HALF: i32 = 19;

fn crc_table() -> &'static [u32; 256] {
    static T: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0x82F6_3B78 ^ (c >> 1) } else { c >> 1 };
            }
            *e = c;
        }
        t
    })
}

/// CRC-32C (Castagnoli).
pub fn crc32c(data: &[u8]) -> u32 {
    let t = crc_table();
    let mut c = !0u32;
    for &b in data {
        c = t[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

/// TFRecord's masked CRC.
pub fn masked_crc(data: &[u8]) -> u32 {
    let c = crc32c(data);
    (c.rotate_right(15)).wrapping_add(0xa282_ead8)
}

/// Does `head` start with a valid TFRecord length header?
pub fn looks_like(head: &[u8]) -> bool {
    head.len() >= 12 && u32::from_le_bytes(head[8..12].try_into().unwrap()) == masked_crc(&head[..8])
}

/// One framed record (writer side; tests and fixture generation).
pub fn frame(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 16);
    let len = (data.len() as u64).to_le_bytes();
    out.extend_from_slice(&len);
    out.extend_from_slice(&masked_crc(&len).to_le_bytes());
    out.extend_from_slice(data);
    out.extend_from_slice(&masked_crc(data).to_le_bytes());
    out
}

/// Records of a TFRecord file, plus why reading stopped early (if it did).
pub fn records(buf: &[u8]) -> (Vec<&[u8]>, Option<String>) {
    let mut out = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        if buf.len() - i < 12 {
            return (out, Some(format!("truncated record header at byte {i}")));
        }
        let len_b = &buf[i..i + 8];
        if u32::from_le_bytes(buf[i + 8..i + 12].try_into().unwrap()) != masked_crc(len_b) {
            return (out, Some(format!("bad length CRC at byte {i}")));
        }
        let len = u64::from_le_bytes(len_b.try_into().unwrap()) as usize;
        let s = i + 12;
        if buf.len() < s + len + 4 || len > buf.len() {
            return (out, Some(format!("truncated record at byte {i} (file still being written?)")));
        }
        let data = &buf[s..s + len];
        if u32::from_le_bytes(buf[s + len..s + len + 4].try_into().unwrap()) != masked_crc(data) {
            return (out, Some(format!("bad data CRC at byte {i}")));
        }
        out.push(data);
        i = s + len + 4;
    }
    (out, None)
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal
            let mut e = -14i32;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (((e + 127) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | (((e - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// The value of a one-element tensor, if it is numeric.
pub fn tensor_scalar(t: &TensorProto) -> Option<f64> {
    let n: i64 = t.tensor_shape.as_ref().map(|s| s.dim.iter().map(|d| d.size).product()).unwrap_or(1);
    if n != 1 {
        return None;
    }
    let c = &t.tensor_content;
    if !c.is_empty() {
        let at = |k: usize| c.get(..k);
        return match t.dtype {
            DT_FLOAT => at(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64),
            DT_DOUBLE => at(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())),
            DT_INT32 => at(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f64),
            DT_INT64 => at(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f64),
            DT_INT16 => at(2).map(|b| i16::from_le_bytes(b.try_into().unwrap()) as f64),
            DT_INT8 => at(1).map(|b| b[0] as i8 as f64),
            DT_UINT8 | DT_BOOL => at(1).map(|b| b[0] as f64),
            DT_HALF => at(2).map(|b| f16_to_f32(u16::from_le_bytes(b.try_into().unwrap())) as f64),
            DT_BFLOAT16 => at(2).map(|b| f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16) as f64),
            _ => None,
        };
    }
    match t.dtype {
        DT_FLOAT => t.float_val.first().map(|&v| v as f64),
        DT_DOUBLE => t.double_val.first().copied(),
        DT_INT32 | DT_INT16 | DT_INT8 | DT_UINT8 | DT_BOOL => t.int_val.first().map(|&v| v as f64),
        DT_INT64 => t.int64_val.first().map(|&v| v as f64),
        DT_HALF => t.half_val.first().map(|&v| f16_to_f32(v as u16) as f64),
        DT_BFLOAT16 => t.half_val.first().map(|&v| f32::from_bits((v as u32 & 0xffff) << 16) as f64),
        _ => None,
    }
}

pub fn ingest(buf: &[u8], run: &mut LocalRun, rep: &mut FileReport) {
    let (recs, stop) = records(buf);
    rep.lines = recs.len();
    if let Some(s) = &stop {
        rep.notes.push(format!("stopped early: {s}"));
        if recs.is_empty() {
            rep.error = Some(format!("not a readable TFRecord file: {s}"));
            return;
        }
    }
    let mut t0: Option<f64> = None;
    let (mut skipped, mut bad) = (0usize, 0usize);
    let mut kinds = std::collections::BTreeMap::<&'static str, usize>::new();
    let mut steps = std::collections::HashSet::new();
    let mut tags = std::collections::HashSet::new();
    for r in recs {
        let ev = match Event::decode(r) {
            Ok(e) => e,
            Err(_) => {
                bad += 1;
                continue;
            }
        };
        if ev.file_version.is_some() {
            continue;
        }
        let Some(sm) = ev.summary else { continue };
        let t = ev.wall_time - *t0.get_or_insert(ev.wall_time);
        let step = ev.step as f64;
        let mut any = false;
        for v in &sm.value {
            let x = if let Some(x) = v.simple_value {
                Some(x as f64)
            } else if let Some(tp) = &v.tensor {
                let x = tensor_scalar(tp);
                if x.is_none() {
                    let plugin = v.metadata.as_ref().and_then(|m| m.plugin_data.as_ref()).map(|p| p.plugin_name.as_str()).unwrap_or("");
                    *kinds.entry(match plugin {
                        "histograms" => "histogram",
                        "images" => "image",
                        "audio" => "audio",
                        "text" => "text",
                        _ => "non-scalar tensor",
                    }).or_default() += 1;
                }
                x
            } else {
                *kinds.entry(if v.histo.is_some() || v.obsolete_old_style_histogram.is_some() {
                    "histogram"
                } else if v.image.is_some() {
                    "image"
                } else if v.audio.is_some() {
                    "audio"
                } else {
                    "other"
                }).or_default() += 1;
                None
            };
            match x {
                Some(x) => {
                    run.push(&v.tag, step, x);
                    rep.points += 1;
                    tags.insert(v.tag.clone());
                    any = true;
                }
                None => skipped += 1,
            }
        }
        if any {
            run.push_time(step, t);
            if steps.insert(ev.step) {
                run.rows += 1;
            }
        }
    }
    rep.skipped = skipped + bad;
    if !kinds.is_empty() {
        rep.notes.push(format!("skipped non-scalar summaries: {}", kinds.iter().map(|(k, n)| format!("{n} {k}")).collect::<Vec<_>>().join(", ")));
    }
    if bad > 0 {
        rep.notes.push(format!("{bad} record(s) did not decode as Event"));
    }
    rep.metrics = tags.len();
    rep.steps = steps.len();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_known() {
        // RFC 3720 test vector
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(&[0u8; 32]), 0x8A91_36AA);
    }

    #[test]
    fn half() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}
