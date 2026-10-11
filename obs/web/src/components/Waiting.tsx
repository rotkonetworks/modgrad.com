import { For, Show } from 'solid-js';
import { S } from '@/stores/runs';
import { waitingFor, recallToken } from '@/stores/live';
import { hash } from '@/lib/hash';
import { serverURL } from '@/lib/base';

/** a run/project board with no data yet: how to send some (server mode) */
export default function Waiting() {
  return (
    <Show when={waitingFor()}>{(what) => {
      const t = hash();
      const origin = serverURL();
      const token = t.project ? recallToken(t.project) : null;
      const dev = !!S.index?.dev && !token; // a dev server takes tokenless writes
      const tok = token || '<token>';
      const run = t.run || 'my-run';
      const blocks: [string, string][] = [
        ['Ask your coding agent', `Add modgrad-obs logging to this repo. Server: ${origin}${dev ? '' : '  Token: ' + tok}  API reference: ${origin}/api/v1`],
        ['Or wrap your training command', `modgrad run --server ${origin}${dev ? '' : ' --token ' + tok} --name ${run} -- python train.py`],
        ['Or send one row with curl', `curl -X POST ${origin}/api/v1/runs/${run}/log${dev ? '' : ` -H "Authorization: Bearer ${tok}"`} -d '{"loss": 0.5}'`],
      ];
      return (
        <div class="live-wait" role="status">
          <div><b>Listening</b> for {what()}. Graphs appear here when the first row arrives.</div>
          <For each={blocks}>{([label, text]) => (<><div><b>{label}</b></div><pre>{text}</pre></>)}</For>
          <div class="dim">{token ? 'This token writes to this board; keep it to yourself. Anyone with this page URL can read the board.' : dev ? 'Dev server: writes need no token.' : 'Use the token that came with this board (POST /api/v1/projects issues one).'}</div>
        </div>
      );
    }}</Show>
  );
}
