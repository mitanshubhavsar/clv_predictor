const base = import.meta.env.BASE_URL;
export const API_URL: string = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export async function loadJson<T>(name: string): Promise<T> {
  const r = await fetch(`${base}data/${name}.json`);
  if (!r.ok) throw new Error(`Could not load ${name}.json`);
  return r.json() as Promise<T>;
}
export const img = (name: string) => `${base}img/${name}`;
export const gbp = (x: number) => `£${Math.round(x).toLocaleString("en-GB")}`;
export const pct = (x: number, d = 0) => `${(x * 100).toFixed(d)}%`;
export const num = (x: number, d = 3) => x.toFixed(d);
