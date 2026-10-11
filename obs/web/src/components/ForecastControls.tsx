import { Show } from 'solid-js';
import * as Bd from '@/logic/board';
import { board, findPanel, setStage, setForecastView, toggleLiveRefit, placeTarget, commit } from '@/stores/board';

/** forecast settings for one panel: in the badge popover and in the inspector */
export default function ForecastControls(props: { id: string }) {
  const pv = () => board.panels.find((p) => p.id === props.id);
  const model = () => findPanel(props.id);
  const stage = () => { const p = pv(); return p ? Bd.getStage(p.q, 'forecast') : null; };
  const parts = () => { const st = stage(); const m = st ? st.match(/^forecast(?: to (\S+))?(?: (\S+))?$/) : null; return { to: m && m[1] ? m[1] : '', method: m && m[2] ? m[2] : 'auto' }; };
  const write = (t: string, me: string) => { const p = model(); if (p) setStage(p, 'forecast', `forecast${t ? ' to ' + t : ''}${me && me !== 'auto' ? ' ' + me : ''}`, 'forecast settings'); };
  const fc = () => pv()!.view.fc;
  return (
    <Show when={pv()}>
      <div class="row">
        <button type="button" classList={{ on: !!stage() }} aria-pressed={!!stage()} title={stage() ? 'turn the forecast off' : 'turn the forecast on'} aria-label={stage() ? 'turn the forecast off' : 'turn the forecast on'}
          onClick={() => { const p = model(); if (!p) return; if (stage()) setStage(p, 'forecast', null); else write('', 'auto'); }}>{stage() ? 'on' : 'off'}</button>
        <Show when={stage()}>
          <label>to <input value={parts().to} size="6" placeholder="auto" aria-label="forecast horizon step" title="forecast horizon (step, e.g. 10k); empty = planned steps or +50%" onChange={(e) => write(e.currentTarget.value.trim(), parts().method)} /></label>
        </Show>
      </div>
      <Show when={stage()}>
        <div class="row">
          <label>method <select aria-label="forecast method" title="auto picks the method with skill and the lowest backtest error" value={parts().method} onChange={(e) => write(parts().to, e.currentTarget.value)}>
            <option value="auto">auto</option><option value="ridge">ridge</option><option value="powerlaw">power law</option><option value="last">last value</option>
          </select></label>
          <label>band <select aria-label="band level" value={String(fc().band)} onChange={(e) => { const p = model(); if (p) setForecastView(p, { band: +e.currentTarget.value }); }}>
            <option value="50">50%</option><option value="80">80%</option><option value="95">95%</option>
          </select></label>
        </div>
        <div class="row">
          <label>pool <select aria-label="pool the ridge fit" title="pool the ridge fit over other series in the panel" value={fc().pool} onChange={(e) => { const p = model(); if (p) setForecastView(p, { pool: e.currentTarget.value as 'none' }); }}>
            <option value="none">none</option><option value="metric">same metric</option><option value="group">same letter</option>
          </select></label>
          <button type="button" classList={{ on: fc().live }} aria-pressed={fc().live} title="refit when new rows arrive; off freezes the current forecast" aria-label="live refit" onClick={() => { const p = model(); if (p) toggleLiveRefit(p); }}>{fc().live ? 'live refit: on' : 'live refit: off'}</button>
        </div>
        <div class="row">
          <label>target <input value={fc().target ?? ''} size="7" aria-label="target value" title="target value for the ETA readout" onChange={(e) => { const v = e.currentTarget.value.trim() === '' ? null : +e.currentTarget.value; const p = model(); if (p && (v === null || Number.isFinite(v))) setForecastView(p, { target: v }, 'forecast target'); }} /></label>
          <button type="button" title="place a target near the current value" aria-label="place a target near the current value" onClick={() => { const p = model(); if (p) placeTarget(p); }}>place</button>
        </div>
        <div class="row">
          <span>fit window</span>
          <input value={fc().window && Math.abs(fc().window![0]) < 1e300 ? Math.round(fc().window![0]) : ''} size="6" placeholder="start" aria-label="fit window start step" onChange={(e) => setWin(0, e.currentTarget.value)} />
          <input value={fc().window && Math.abs(fc().window![1]) < 1e300 ? Math.round(fc().window![1]) : ''} size="6" placeholder="end" aria-label="fit window end step" onChange={(e) => setWin(1, e.currentTarget.value)} />
          <button type="button" title="fit on all points" aria-label="fit on all points" onClick={() => { const p = model(); if (p) setForecastView(p, { window: null }, 'forecast fit window'); }}>all</button>
        </div>
      </Show>
    </Show>
  );
  function setWin(i: 0 | 1, v: string): void {
    const p = model(); if (!p) return;
    const w = p.view.fc.window ? [...p.view.fc.window] : [-1e300, 1e300];
    w[i] = v.trim() === '' ? (i === 0 ? -1e300 : 1e300) : +v;
    commit('forecast fit window', () => { p.view.fc.window = w[0] <= -1e300 && w[1] >= 1e300 ? null : [w[0], w[1]]; });
  }
}
