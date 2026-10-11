// Parses dropped log files off the main thread with the Rust ingest
// (crates/modgrad-obs-ingest, built into ./pkg by scripts/build_obs_wasm.sh).
// One message = one run: {id, name, files: [{rel, file}]}. Files are read in
// 4 MiB slices, so a 100 MB log is never one string and progress is posted
// per slice. Replies: {id, type: 'progress' | 'done' | 'error', ...}; the
// metric arrays come back as transferred Float64Arrays.
import init, { RunBuilder } from './pkg/modgrad_obs_ingest.js';

const ready = init();
const CHUNK = 4 << 20;

self.onmessage = async (e) => {
  const { id, name, files } = e.data;
  let b = null;
  try {
    await ready;
    b = new RunBuilder(name);
    const total = files.reduce((a, f) => a + f.file.size, 0);
    let done = 0, last = 0;
    for (const { rel, file } of files) {
      b.begin(rel, file.lastModified / 1000);
      for (let off = 0; off < file.size; off += CHUNK) {
        const buf = new Uint8Array(await file.slice(off, off + CHUNK).arrayBuffer());
        b.push(buf);
        done += buf.length;
        const now = performance.now();
        if (now - last > 100 || done === total) { last = now; self.postMessage({ id, type: 'progress', done, total, file: rel }); }
      }
      b.end();
    }
    const meta = JSON.parse(b.finish());
    const steps = [], values = [], transfer = [];
    for (let i = 0; i < meta.keys.length; i++) {
      const s = b.steps(i), v = b.values(i);
      steps.push(s); values.push(v); transfer.push(s.buffer, v.buffer);
    }
    const ts = b.time_steps(), tv = b.time_values();
    transfer.push(ts.buffer, tv.buffer);
    self.postMessage({ id, type: 'done', meta, steps, values, ts, tv }, transfer);
  } catch (err) {
    self.postMessage({ id, type: 'error', error: String(err && err.message || err) });
  } finally {
    if (b) b.free();
  }
};
