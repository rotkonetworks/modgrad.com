//! wasm-bindgen exports. Slices cross as typed arrays (`Float64Array`,
//! `Float32Array`, `Uint32Array`); multi-output results are small structs
//! with getter methods.

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn ema(vals: &[f64], weight: f64) -> Vec<f64> {
    crate::ema(vals, weight)
}

/// `[max, avg, min, current]`.
#[wasm_bindgen]
pub fn legend_stats(vals: &[f64]) -> Vec<f64> {
    crate::legend_stats(vals).to_vec()
}

#[wasm_bindgen]
pub struct Series {
    steps: Vec<f64>,
    mean: Vec<f64>,
    min: Vec<f64>,
    max: Vec<f64>,
}

#[wasm_bindgen]
impl Series {
    pub fn steps(&self) -> Vec<f64> {
        self.steps.clone()
    }
    pub fn mean(&self) -> Vec<f64> {
        self.mean.clone()
    }
    pub fn min(&self) -> Vec<f64> {
        self.min.clone()
    }
    pub fn max(&self) -> Vec<f64> {
        self.max.clone()
    }
}

#[wasm_bindgen]
pub fn downsample_series(steps: &[f64], vals: &[f64], max_points: usize) -> Series {
    let (steps, mean, min, max) = crate::downsample_series(steps, vals, max_points);
    Series { steps, mean, min, max }
}

#[wasm_bindgen]
pub struct Pca {
    coords: Vec<f32>,
    explained: Vec<f64>,
}

#[wasm_bindgen]
impl Pca {
    /// `n × 3` row-major.
    pub fn coords(&self) -> Vec<f32> {
        self.coords.clone()
    }
    /// Explained-variance ratio of PC1..PC3.
    pub fn explained(&self) -> Vec<f64> {
        self.explained.clone()
    }
}

#[wasm_bindgen]
pub fn pca3(data: &[f32], n: usize, d: usize) -> Pca {
    let (coords, r) = crate::pca3(data, n, d);
    Pca { coords, explained: r.to_vec() }
}

#[wasm_bindgen]
pub struct SurfaceOut {
    inner: crate::Surface,
}

#[wasm_bindgen]
impl SurfaceOut {
    pub fn xs(&self) -> Vec<f64> {
        self.inner.xs.clone()
    }
    pub fn ys(&self) -> Vec<f64> {
        self.inner.ys.clone()
    }
    pub fn z(&self) -> Vec<f64> {
        self.inner.z.clone()
    }
    pub fn count(&self) -> Vec<u32> {
        self.inner.count.clone()
    }
}

/// `off` has `runs + 1` entries (as `Uint32Array`).
#[wasm_bindgen]
pub fn assemble_surface(run_x: &[f64], off: &[u32], steps: &[f64], vals: &[f64], step_bin: f64) -> SurfaceOut {
    let off: Vec<usize> = off.iter().map(|&o| o as usize).collect();
    SurfaceOut { inner: crate::assemble_surface(run_x, &off, steps, vals, step_bin) }
}

/// `[sx, sy, depth]` per point.
#[wasm_bindgen]
pub fn project(xyz: &[f64], az_deg: f64, el_deg: f64) -> Vec<f64> {
    crate::project(xyz, az_deg, el_deg)
}

/// Scaled points followed by 6 bounds: `[…3n…, minx, maxx, miny, maxy, minz, maxz]`.
#[wasm_bindgen]
pub fn normalize_box(xyz: &[f64]) -> Vec<f64> {
    let (mut v, b) = crate::normalize_box(xyz);
    v.extend_from_slice(&b);
    v
}

#[wasm_bindgen]
pub fn painter_order(nx: usize, ny: usize, depth: &[f64]) -> Vec<u32> {
    crate::painter_order(nx, ny, depth)
}

#[wasm_bindgen]
pub fn depth_order(depth: &[f64]) -> Vec<u32> {
    crate::depth_order(depth)
}

#[wasm_bindgen]
pub fn nice_ticks(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    crate::nice_ticks(lo, hi, n)
}

/// Mean-pool an N-d array to at most `max_elems`; returns data, and the new
/// shape via `downsample_shape` (same arguments).
#[wasm_bindgen]
pub fn downsample(shape: &[u32], data: &[f32], max_elems: usize) -> Vec<f32> {
    let s: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    crate::downsample(&s, data, max_elems).1
}

#[wasm_bindgen]
pub fn downsample_shape(shape: &[u32], max_elems: usize) -> Vec<u32> {
    let mut s: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    let mut n: usize = s.iter().product();
    while n > max_elems.max(1) {
        let (ax, &len) = s.iter().enumerate().max_by_key(|&(_, l)| *l).unwrap();
        if len <= 1 {
            break;
        }
        s[ax] = len.div_ceil(2);
        n = s.iter().product();
    }
    s.into_iter().map(|x| x as u32).collect()
}
