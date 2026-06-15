/** The modular-gradient mark: a ∇ (del / gradient operator) built from
 *  modular cells under one flowing indigo→violet→fuchsia gradient. */
export default function Logo(props: { size?: number; id?: string; class?: string }) {
  const s = props.size ?? 26;
  const gid = props.id ?? "mg-mark";
  return (
    <svg
      width={s}
      height={s}
      viewBox="0 0 32 32"
      class={props.class}
      role="img"
      aria-label="modgrad"
    >
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#6366f1" />
          <stop offset="0.5" stop-color="#a855f7" />
          <stop offset="1" stop-color="#e879f9" />
        </linearGradient>
      </defs>
      <g fill={`url(#${gid})`}>
        <rect x="1.75" y="2" width="6" height="6" rx="1.3" />
        <rect x="9.25" y="2" width="6" height="6" rx="1.3" />
        <rect x="16.75" y="2" width="6" height="6" rx="1.3" />
        <rect x="24.25" y="2" width="6" height="6" rx="1.3" />
        <rect x="5.5" y="9.5" width="6" height="6" rx="1.3" />
        <rect x="13" y="9.5" width="6" height="6" rx="1.3" />
        <rect x="20.5" y="9.5" width="6" height="6" rx="1.3" />
        <rect x="9.25" y="17" width="6" height="6" rx="1.3" />
        <rect x="16.75" y="17" width="6" height="6" rx="1.3" />
        <rect x="13" y="24.5" width="6" height="6" rx="1.3" />
      </g>
    </svg>
  );
}
