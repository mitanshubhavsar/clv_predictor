import { useEffect, useState } from "react";
import { API_URL, gbp, pct } from "./lib";
import type { Preset, Score } from "./types";

type Status = "unset" | "checking" | "waking" | "online" | "offline";
type Out = { score: Score; source: "api" | "saved" } | null;

export default function Scorer({ presets }: { presets: Preset[] }) {
  const [sel, setSel] = useState(0);
  const p = presets[sel];
  const [rec, setRec] = useState(String(p.features.Recency));
  const [freq, setFreq] = useState(String(p.features.Frequency));
  const [mon, setMon] = useState(String(p.features.Monetary));
  const [ipi, setIpi] = useState(p.features.IPI == null ? "" : String(p.features.IPI));
  const [status, setStatus] = useState<Status>(API_URL ? "checking" : "unset");
  const [out, setOut] = useState<Out>(null);
  const [msg, setMsg] = useState("");

  useEffect(() => {
    if (!API_URL) return;
    let alive = true;
    const t = setTimeout(() => alive && setStatus((s) => (s === "checking" ? "waking" : s)), 2500);
    fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(90000) })
      .then((r) => alive && setStatus(r.ok ? "online" : "offline"))
      .catch(() => alive && setStatus("offline"));
    return () => { alive = false; clearTimeout(t); };
  }, []);

  function choose(i: number) {
    const q = presets[i];
    setSel(i); setRec(String(q.features.Recency)); setFreq(String(q.features.Frequency));
    setMon(String(q.features.Monetary)); setIpi(q.features.IPI == null ? "" : String(q.features.IPI));
    setOut(null); setMsg("");
  }

  const unchanged =
    Number(rec) === p.features.Recency && Number(freq) === p.features.Frequency &&
    Number(mon) === p.features.Monetary && (ipi === "" ? p.features.IPI == null : Number(ipi) === p.features.IPI);

  async function score() {
    setMsg("");
    const body = { Recency: Number(rec), Frequency: Number(freq), Monetary: Number(mon), ...(ipi === "" ? {} : { IPI: Number(ipi) }) };
    if (status === "online" || status === "waking" || status === "checking") {
      try {
        const r = await fetch(`${API_URL}/score`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: AbortSignal.timeout(90000) });
        if (!r.ok) { setMsg("The API rejected these values (check recency ≥ 0, frequency ≥ 1)."); return; }
        setOut({ score: (await r.json()) as Score, source: "api" }); setStatus("online"); return;
      } catch { setStatus("offline"); }
    }
    if (unchanged) {
      setOut({ score: { predicted_spend: p.predicted_spend, multiplier: p.predicted_spend / Math.max(p.features.Monetary, 50), probability_buy_again: p.probability_buy_again, segment: p.segment }, source: "saved" });
    } else {
      setOut(null); setMsg("The live API is not reachable, so only the saved example scores can be shown. Pick an example without editing it.");
    }
  }

  const label: Record<Status, string> = {
    unset: "Live API not configured: showing saved scores for the examples.",
    checking: "Checking the live API…",
    waking: "The API is waking up (free hosting sleeps when idle; this can take about a minute)…",
    online: "Live API online.",
    offline: "Live API not reachable: showing saved scores for the examples.",
  };

  return (
    <div className="scorer">
      <p className={`status ${status}`} role="status">{label[status]}</p>
      <div className="presets" role="group" aria-label="Example customers">
        {presets.map((q, i) => (
          <button key={q.customer_id} className={i === sel ? "chip on" : "chip"} onClick={() => choose(i)}>{q.segment}</button>
        ))}
      </div>
      <div className="fields">
        <label>Days since last purchase<input inputMode="decimal" value={rec} onChange={(e) => setRec(e.target.value)} /></label>
        <label>Purchase days in 12 months<input inputMode="numeric" value={freq} onChange={(e) => setFreq(e.target.value)} /></label>
        <label>Net spend in 12 months (£)<input inputMode="decimal" value={mon} onChange={(e) => setMon(e.target.value)} /></label>
        <label>Usual days between purchases<input inputMode="decimal" placeholder="blank = one purchase day" value={ipi} onChange={(e) => setIpi(e.target.value)} /></label>
      </div>
      <button className="go" onClick={score}>Score this customer</button>
      {msg && <p className="warn">{msg}</p>}
      {out && (
        <div className="result" aria-live="polite">
          <div><span className="big">{gbp(out.score.predicted_spend)}</span><span className="sub"> expected spend, next 6 months</span></div>
          <div><span className="big">{pct(out.score.probability_buy_again)}</span><span className="sub"> chance of buying again</span></div>
          <div><span className="big">{out.score.segment}</span><span className="sub"> segment</span></div>
          <p className="small">Source: {out.source === "api" ? "live API" : "saved score from the backtest"}.
            {unchanged && out.source === "saved" ? ` This customer actually spent ${gbp(p.actual_spend)}.` : ""}</p>
        </div>
      )}
    </div>
  );
}
