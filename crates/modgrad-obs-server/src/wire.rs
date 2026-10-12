//! Compact binary responses: `"OBS1" | u32 LE header_len | header JSON |
//! payload`, where the header lists frames `{name, dtype: "f32"|"f64",
//! shape}` and the payload holds their little-endian values back to back.

use serde_json::{Value, json};

pub const MAGIC: &[u8; 4] = b"OBS1";
pub const CONTENT_TYPE: &str = "application/x-obs-frames";

#[derive(Debug, Clone, PartialEq)]
pub enum FrameData {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: FrameData,
}

/// Encode frames plus free-form `meta` (merged into the header).
pub fn encode(frames: &[Frame], meta: Value) -> Vec<u8> {
    let hdr_frames: Vec<Value> = frames
        .iter()
        .map(|f| json!({ "name": f.name, "dtype": match f.data { FrameData::F32(_) => "f32", FrameData::F64(_) => "f64" }, "shape": f.shape }))
        .collect();
    let mut header = json!({ "frames": hdr_frames });
    if let Value::Object(m) = meta {
        for (k, v) in m {
            header[k] = v;
        }
    }
    let h = header.to_string().into_bytes();
    let mut out = Vec::with_capacity(8 + h.len() + frames.iter().map(|f| f.shape.iter().product::<usize>() * 8).sum::<usize>());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(h.len() as u32).to_le_bytes());
    out.extend_from_slice(&h);
    for f in frames {
        match &f.data {
            FrameData::F32(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
            FrameData::F64(v) => v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes())),
        }
    }
    out
}

/// Decode (used by tests and Rust clients).
pub fn decode(buf: &[u8]) -> Option<(Value, Vec<Frame>)> {
    if buf.len() < 8 || &buf[..4] != MAGIC {
        return None;
    }
    let hl = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;
    let header: Value = serde_json::from_slice(buf.get(8..8 + hl)?).ok()?;
    let mut pos = 8 + hl;
    let mut frames = Vec::new();
    for f in header["frames"].as_array()? {
        let shape: Vec<usize> = serde_json::from_value(f["shape"].clone()).ok()?;
        let n: usize = shape.iter().product();
        let data = match f["dtype"].as_str()? {
            "f32" => {
                let b = buf.get(pos..pos + 4 * n)?;
                pos += 4 * n;
                FrameData::F32(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
            }
            "f64" => {
                let b = buf.get(pos..pos + 8 * n)?;
                pos += 8 * n;
                FrameData::F64(b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
            }
            _ => return None,
        };
        frames.push(Frame { name: f["name"].as_str()?.to_string(), shape, data });
    }
    (pos == buf.len()).then_some((header, frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let fr = vec![
            Frame { name: "step".into(), shape: vec![3], data: FrameData::F64(vec![1.0, 2.0, 1e12]) },
            Frame { name: "loss".into(), shape: vec![3], data: FrameData::F32(vec![0.5, f32::NAN, -1.0]) },
        ];
        let b = encode(&fr, json!({"run": "r"}));
        let (h, back) = decode(&b).unwrap();
        assert_eq!(h["run"], "r");
        assert_eq!(back[0], fr[0]);
        if let FrameData::F32(v) = &back[1].data {
            assert!(v[1].is_nan() && v[2] == -1.0);
        } else {
            panic!()
        }
        assert!(decode(&b[..b.len() - 1]).is_none());
    }
}
