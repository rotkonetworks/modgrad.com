import { For, Show } from 'solid-js';

/** a query error with the failing span underlined; `suggest` replacements for the span are offered
 *  as buttons ("did you mean ema?") that hand the fixed text to `onFix` */
export default function ErrLine(props: { src: string | null; msg: string; a: number; b: number; suggest?: string[]; onFix?: (text: string) => void }) {
  const fixed = (s: string) => props.src!.slice(0, props.a) + s + props.src!.slice(props.b);
  return (
    <div class="berr" role="alert">
      {props.src && props.b > props.a ? <>{props.src.slice(0, props.a)}<u>{props.src.slice(props.a, props.b) || ' '}</u>{props.src.slice(props.b) + '\n'}</> : null}
      {props.msg}
      <Show when={props.src && props.onFix && props.suggest && props.suggest.length}>
        <span class="fixes">{'\ndid you mean '}<For each={props.suggest}>{(s, i) => <>{i() ? ' or ' : ''}<button type="button" class="mini fix" title={`replace "${props.src!.slice(props.a, props.b)}" with "${s}": ${fixed(s)}`} aria-label={`fix: use ${s}`} onClick={() => props.onFix!(fixed(s))}>{s}</button></>}</For>?</span>
      </Show>
    </div>
  );
}
