// Forecast fits off the main thread: the same forecastTrace (ridge search + power law, wasm) as
// the page, in a worker with its own instance of the analysis module. One message = one fit:
// {id, xs, raw, to, opts}; reply {id, f} or {id, error}.
import { initWasm } from '@/lib/wasm';
import { forecastTrace, type ForecastOpts } from '@/logic/data';

const ready = initWasm();
const post = (m: unknown) => (self as unknown as Worker).postMessage(m);

self.onmessage = async (e: MessageEvent<{ id: number; xs: Float64Array; raw: Float64Array; to: number; opts: ForecastOpts }>) => {
  const { id, xs, raw, to, opts } = e.data;
  try {
    await ready;
    post({ id, f: forecastTrace({ xs, raw }, to, opts) });
  } catch (err) {
    post({ id, error: String((err as Error)?.message || err) });
  }
};
