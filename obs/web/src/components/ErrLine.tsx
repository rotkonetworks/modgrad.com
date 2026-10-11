/** a query error with the failing span underlined */
export default function ErrLine(props: { src: string | null; msg: string; a: number; b: number }) {
  return (
    <div class="berr" role="alert">
      {props.src && props.b > props.a ? <>{props.src.slice(0, props.a)}<u>{props.src.slice(props.a, props.b) || ' '}</u>{props.src.slice(props.b) + '\n'}</> : null}
      {props.msg}
    </div>
  );
}
