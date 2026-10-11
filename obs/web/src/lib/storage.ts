// Browser storage, always behind try/catch (private windows, blocked site data).
export function lsGet(k: string): string | null { try { return localStorage.getItem(k); } catch { return null; } }
export function lsSet(k: string, v: string): void { try { localStorage.setItem(k, v); } catch { /* page-only */ } }
export function ssGet(k: string): string | null { try { return sessionStorage.getItem(k); } catch { return null; } }
export function ssSet(k: string, v: string): void { try { sessionStorage.setItem(k, v); } catch { /* page-only */ } }
