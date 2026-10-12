// Where data and the API live relative to the page. Default: next to the page. `?data=../../`
// (only ../ sequences are accepted) serves the app from a subfolder, e.g. obs/dashboard.html
// redirecting to web/dist/ while index.json, runs/ and api/ stay at the obs root.
const q = new URLSearchParams(location.search).get('data');
export const DATA: string = q && /^(\.\.\/)+$/.test(q) ? q : '';
/** a data or API path (relative, as the index and the server use them) -> fetchable URL */
export const dataURL = (path: string): string => DATA + path;
/** the server's base URL as users should type it (no trailing slash) */
export const serverURL = (): string => new URL(DATA || './', location.href).href.replace(/\/$/, '');
