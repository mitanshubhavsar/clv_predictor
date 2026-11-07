import { useEffect, useState } from "react";
import Scorer from "./Scorer";
import { gbp, img, loadJson, num, pct } from "./lib";
import type { Dataset, Interval, ScorerData, Summary } from "./types";

const SECTIONS = [["about", "About"], ["dataset", "Dataset"], ["architecture", "Architecture"], ["results", "Results"], ["try", "Try it"]];

const ci = (m: Interval, f: (x: number) => string) => `${f(m.value)} [${f(m.ci95[0])}, ${f(m.ci95[1])}]`;

function Fig({ src, alt, caption }: { src: string; alt: string; caption: string }) {
  return <figure><img src={img(src)} alt={alt} /><figcaption>{caption}</figcaption></figure>;
}

export default function App() {
  const [s, setS] = useState<Summary | null>(null);
  const [d, setD] = useState<Dataset | null>(null);
  const [sc, setSc] = useState<ScorerData | null>(null);
  const [err, setErr] = useState("");
  const [active, setActive] = useState("about");

  useEffect(() => {
    Promise.all([loadJson<Summary>("summary"), loadJson<Dataset>("dataset"), loadJson<ScorerData>("scorer")])
      .then(([a, b, c]) => { setS(a); setD(b); setSc(c); })
      .catch((e: Error) => setErr(e.message));
  }, []);

  useEffect(() => {
    const obs = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setActive(e.target.id)), { rootMargin: "-40% 0px -55% 0px" });
    SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) obs.observe(el); });
    return () => obs.disconnect();
  }, [s]);

  if (err) return <main><p className="warn">Could not load the saved results: {err}</p></main>;
  if (!s || !d || !sc) return <main><p>Loading…</p></main>;
  const C = s.test.model_C_headline, P = s.test.past_spend, diff = s.diff_vs_past_spend;

  return (
    <>
      <a className="skip" href="#about">Skip to content</a>
      <header>
        <strong>Customer spend forecast</strong>
        <nav aria-label="Sections">
          {SECTIONS.map(([id, l]) => <a key={id} href={`#${id}`} className={active === id ? "cur" : ""} aria-current={active === id ? "true" : undefined}>{l}</a>)}
        </nav>
      </header>
      <main>
        <section id="about">
          <h1>How much will each customer spend in the next 6 months?</h1>
          <p className="lead">A model for an online gift retailer that predicts each customer's net spend (purchases minus returns) over the next 6 months, and groups customers into four action segments. Built and tested honestly: scored once on a later period it never saw.</p>
          <div className="tiles">
            <div className="tile"><b>{num(C.R2.value, 2)}</b><span>R² on the test period</span></div>
            <div className="tile"><b>{gbp(C.MAE.value)}</b><span>average error per customer</span></div>
            <div className="tile"><b>{pct(C["Capture@20%"].value)}</b><span>of future money held by the model's top 20%</span></div>
          </div>
          <p className="callout"><strong>The honest summary.</strong> A plain rule, "sort by past spend", is almost as good. The model is about {gbp(diff["MAE gain"].point)} ({pct(diff["MAE gain"].point / P.MAE.value)}) more accurate per customer, and identical for ranking. Use it to size and prioritise customers, not as a big upgrade. It is not causal and says nothing about margin.</p>
        </section>

        <section id="dataset">
          <h2>Dataset</h2>
          <p>UCI Online Retail II: {s.raw_rows.toLocaleString("en-GB")} invoice lines, {s.raw_customers.toLocaleString("en-GB")} customers, {s.raw_first_date.slice(0, 10)} to {s.raw_last_date.slice(0, 10)}. Rows without a customer ID (23%) are dropped; cancellations become returns.</p>
          <h3>Raw columns</h3>
          <div className="scroll"><table><thead><tr><th>Column</th><th>Meaning</th><th>Example</th></tr></thead>
            <tbody>{d.raw.schema.map((r) => <tr key={r.column}><td>{r.column}</td><td>{r.meaning}</td><td>{r.example}</td></tr>)}</tbody></table></div>
          <h3>One customer, followed through</h3>
          <p>Customer {d.followed_customer}'s raw invoice lines (first rows):</p>
          <div className="scroll"><table><thead><tr>{d.raw.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
            <tbody>{d.raw.rows.slice(0, 6).map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j}>{String(v)}</td>)}</tr>)}</tbody></table></div>
          <h3>Features the model uses</h3>
          <div className="scroll"><table><thead><tr><th>Feature</th><th>Meaning</th><th>Typical (5th / median / 95th)</th></tr></thead>
            <tbody>{d.features.schema.map((r) => <tr key={r.column}><td>{r.column}</td><td>{r.meaning}</td><td>{r.p5} / {r.median} / {r.p95}</td></tr>)}</tbody></table></div>
        </section>

        <section id="architecture">
          <h2>Architecture</h2>
          <Fig src="pipeline.png" alt="Pipeline: raw invoices, clean, ledger, snapshot, features, model C" caption="Every step uses the same code in training and in the API, so features cannot drift." />
          <Fig src="snapshot_timeline.png" alt="Train and test snapshots as bars on a timeline" caption={`Features come only from before the cutoff. Train cutoff ${s.cutoff_train}, test cutoff ${s.cutoff_test}; the test target period is entirely later than anything the model saw.`} />
          <p><strong>Model C.</strong> prediction = multiplier × the customer's past spend (at least £50). The multiplier comes from a small boosted-tree model on recency, frequency, monetary, purchase interval and overdue ratio.</p>
          <Fig src="serving.png" alt="Serving: browser, GitHub Pages, FastAPI, model, reply" caption="The site is static; the live scorer calls a small FastAPI service." />
        </section>

        <section id="results">
          <h2>Results</h2>
          <p>Test period, {s.n_test.toLocaleString("en-GB")} customers ({pct(s.zero_share_test)} spent nothing). Intervals are 95% bootstraps over customers.</p>
          <div className="scroll"><table><thead><tr><th>Model</th><th>R²</th><th>MAE</th><th>Capture of money, top 20%</th></tr></thead><tbody>
            <tr><td>Sort by past spend</td><td>{ci(P.R2, (x) => num(x))}</td><td>{ci(P.MAE, gbp)}</td><td>{ci(P["Capture@20%"], (x) => num(x))}</td></tr>
            <tr><td><strong>Model C</strong></td><td>{ci(C.R2, (x) => num(x))}</td><td>{ci(C.MAE, gbp)}</td><td>{ci(C["Capture@20%"], (x) => num(x))}</td></tr>
          </tbody></table></div>
          <p>Model C minus past spend: R² {diff.R2.point >= 0 ? "+" : ""}{num(diff.R2.point)} [{num(diff.R2.ci[0])}, {num(diff.R2.ci[1])}] · MAE better by {gbp(diff["MAE gain"].point)} [{gbp(diff["MAE gain"].ci[0])}, {gbp(diff["MAE gain"].ci[1])}] · capture {num(diff["Capture@20%"].point)} (no difference). Perfect ranking would capture {pct(s.capture_ceiling)}.</p>
          <Fig src="capture_curves.png" alt="Revenue capture curves" caption="Contacting customers best-first: the model and the past-spend sort lie on top of each other." />
          <Fig src="model_comparison.png" alt="Error of each model tried" caption="Every model tried, 5-fold CV on the training period." />
          <h3>Targeting view</h3>
          <p>The top 20% by model C spend {gbp(s.targeting.model_C_mean_future_spend)} each in the next 6 months against {gbp(s.targeting.average_customer)} for the average customer, and hold {pct(s.targeting.model_C_share_of_money)} of future money. Past spend gives {gbp(s.targeting.past_spend_mean_future_spend)}. This measures ranking, not campaign ROI.</p>
          <h3>Segments</h3>
          <p>High value = predicted spend of {gbp(s.high_value_cutoff_gbp)} or more; likely to return = probability of buying again above 50%.</p>
          <div className="scroll"><table><thead><tr><th>Segment</th><th>Customers</th><th>Share of money</th><th>Mean future spend</th><th>Bought again</th></tr></thead>
            <tbody>{s.segments.map((g) => <tr key={g.segment}><td>{g.segment}</td><td>{g.customers.toLocaleString("en-GB")} ({pct(g["share of customers"])})</td><td>{pct(g["share of future money"], 1)}</td><td>{gbp(g.mean_future_spend)}</td><td>{pct(g.bought_again)}</td></tr>)}</tbody></table></div>
          <Fig src="shap_importance.png" alt="Feature importance" caption="What the model uses. This shows use, not proven accuracy gain." />
          <h3>Limits</h3>
          <ul>
            <li>Ranks who will spend, not who is persuadable (not causal); revenue, not margin.</li>
            <li>Ten giant customers hold about a fifth of the money and cause most of the error; treat them as named accounts.</li>
            <li>Two snapshots from one retailer with Christmas-heavy seasons.</li>
            <li>The test period was scored once for the model, and later also used to describe segments.</li>
          </ul>
        </section>

        <section id="try">
          <h2>Try it</h2>
          <p>Pick an example customer or type your own numbers.</p>
          <Scorer presets={sc.presets} />
        </section>
      </main>
      <footer>Data: UCI Online Retail II (CC BY 4.0). Every number on this page is read from files written by the export script.</footer>
    </>
  );
}
