import { gbp, img, num, pct } from "./lib";
import type { Decision, FlowStage, Heuristics, Story, Summary } from "./types";

// Fill {{placeholders}} in the hand-written text from the saved summary, so these numbers cannot drift.
function fill(text: string, s: Summary): string {
  const vals: Record<string, string> = {
    top_c: gbp(s.targeting.model_C_mean_future_spend),
    top_p: gbp(s.targeting.past_spend_mean_future_spend),
    avg: gbp(s.targeting.average_customer),
    share: pct(s.targeting.model_C_share_of_money),
  };
  return text.replace(/\{\{(\w+)\}\}/g, (_, k: string) => vals[k] ?? "");
}

function StageTable({ stage, story, s }: { stage: FlowStage; story: Story; s: Summary }) {
  if (stage.table === "models") {
    return (
      <div className="scroll">
        <table>
          <caption>Every model tried, 5-fold cross-validation on the training period</caption>
          <thead><tr><th>Model</th><th>R²</th><th>MAE</th><th>Capture, top 20%</th></tr></thead>
          <tbody>{s.model_comparison.map((m) => (
            <tr key={m.model}><td>{m.model}</td><td>{Math.abs(m.R2) < 0.005 ? "0.00" : num(m.R2, 2)}</td><td>{gbp(m.MAE)}</td><td>{num(m["Capture@20%"], 2)}</td></tr>
          ))}</tbody>
        </table>
      </div>
    );
  }
  if (stage.table === "features") {
    return (
      <div className="scroll">
        <table>
          <caption>Candidate features against the model with the same settings</caption>
          <thead><tr><th>Feature</th><th>Verdict</th><th>Evidence</th></tr></thead>
          <tbody>{story.features_table.map((r) => <tr key={r[0]}><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>)}</tbody>
        </table>
      </div>
    );
  }
  return null;
}

const TURNING = new Set(["models", "test"]);

export function Flow({ story, s }: { story: Story; s: Summary }) {
  return (
    <>
      <p>How the project went from raw invoices to a served model, in the order it happened, including the dead ends. Each stage says what was done, why, and what it showed.</p>
      <figure>
        <img src={img("flow.png")} alt="Thirteen stages in order: raw data, clean, time split, RFM features, study the target, baselines, try models, test features one by one, final model scored once, segments, SHAP, targeting, package" />
        <figcaption>The thirteen stages. Orange marks the two turning points: no model beat the simple rule, and the test period was scored exactly once.</figcaption>
      </figure>
      <ol className="stages">
        {story.flow.map((st, i) => (
          <li key={st.id} id={`stage-${st.id}`}>
            <h3><span className="num">{i + 1}</span> {st.title}{TURNING.has(st.id) ? <span className="turn">Turning point</span> : null}</h3>
            <dl>
              <dt>What we did</dt><dd>{fill(st.did, s)}</dd>
              <dt>Why</dt><dd>{fill(st.why, s)}</dd>
              <dt>What we found</dt><dd className="found">{fill(st.found, s)}</dd>
            </dl>
            <StageTable stage={st} story={story} s={s} />
            <p className="src">Notes: {st.source}</p>
          </li>
        ))}
      </ol>
    </>
  );
}

function Card({ d, s }: { d: Decision; s: Summary }) {
  const weak = d.evidence.startsWith("ARITHMETIC");
  return (
    <article className="decision">
      <h3>{d.title}</h3>
      <p className={weak ? "tag weakTag" : "tag"}>Evidence: {d.evidence}</p>
      <dl>
        <dt>How we got there</dt><dd>{fill(d.how, s)}</dd>
        <dt>Result</dt><dd>{fill(d.result, s)}</dd>
        <dt>How far to trust it</dt><dd>{fill(d.limit, s)}</dd>
      </dl>
    </article>
  );
}

export function Decisions({ story, s }: { story: Story; s: Summary }) {
  return (
    <>
      <p>What the analysis supports, and how each conclusion was reached. <strong>DATA</strong> means measured on held-out data. <strong>RULE OF THUMB</strong> is a recommendation from a label that the data does not test. <strong>ARITHMETIC ON ASSUMPTIONS</strong> is a calculation on a made-up input.</p>
      {story.decisions.map((d) => <Card key={d.id} d={d} s={s} />)}
      <h3>What would strengthen these conclusions</h3>
      <ul>{story.not_done.map((t) => <li key={t}>{t}</li>)}</ul>
    </>
  );
}

export function HeuristicsSection({ h }: { h: Heuristics }) {
  const t = h.textbook, tr = h.ours.train, d = h.diagnostic;
  const naive = h.table[1], model = h.table[3];
  return (
    <>
      <p>You can estimate customer lifetime value with no machine learning at all, and even without any transaction history, using a formula and a few averages from industry benchmarks or early-stage estimates. This section shows the two standard formulas, runs them on this dataset, and scores them on the same test period as model C.</p>

      <h3>1. Averaged formula (needs ARPU, margin and churn)</h3>
      <p className="formula">CLV = (average revenue per user × gross margin) ÷ churn rate</p>
      <p>Textbook example: ${t.arpu} per month × {pct(t.margin)} margin ÷ {pct(t.churn)} monthly churn = <strong>${t.clv.toLocaleString("en-GB")}</strong>.</p>
      <p>On this data, with 6-month periods and revenue only (there is no margin or cost data): the average customer who buys in a period spends {gbp(tr.arpu_active)}, and {pct(tr.churn)} of customers buy nothing in the next 6 months, so CLV = {gbp(tr.arpu_active)} ÷ {num(tr.churn, 2)} ≈ <strong>{gbp(tr.clv_formula1)}</strong> of revenue over a lifetime of about {num(h.ours.lifetimes_periods_train, 1)} periods.</p>
      <p><strong>Limits.</strong> It gives one number for everyone, so it cannot say who to prioritise: scored as a prediction it is the "average for everyone" row below (R² of zero). A shop customer does not announce that they have left: "buys nothing in the next 6 months" is not the same as gone for good, so churn measured this way is only a proxy and can overstate leaving. And it needs a margin that revenue data cannot give.</p>

      <h3>2. Fixed-horizon formula (needs order value, frequency and lifespan)</h3>
      <p className="formula">CLV = average order value × orders per period × lifespan × profit margin</p>
      <p>For a 6-month horizon with each customer's own history: order value = net spend ÷ orders, orders per 6 months = orders in the last 12 months ÷ 2, lifespan = 1 period (assume the customer stays the whole 6 months), margin = 1. That works out to <strong>half of past spend</strong>. It is a per-customer number, so it can rank customers.</p>

      <h3>How the heuristics score on the test period</h3>
      <div className="scroll">
        <table>
          <caption>Test period, scored once, same customers and measures as model C</caption>
          <thead><tr><th>Method</th><th>R²</th><th>MAE</th><th>Capture, top 20%</th><th>Predicted total ÷ actual</th></tr></thead>
          <tbody>{h.table.map((r) => (
            <tr key={r.name}><td><strong>{r.name}</strong><br /><span className="src">{r.how}</span></td><td>{num(r.R2)}</td><td>{gbp(r.MAE)}</td><td>{num(r["Capture@20%"])}</td><td>{num(r.total_pred_over_actual, 2)}</td></tr>
          ))}</tbody>
        </table>
      </div>

      <h3>What to take from it</h3>
      <ul>
        <li><strong>Ranking comes from past spend, not from machine learning.</strong> The unfitted rule captures {num(naive["Capture@20%"], 3)} of future money in its top 20%, model C {num(model["Capture@20%"], 3)}. Its average error is {gbp(naive.MAE)} against {gbp(model.MAE)}.</li>
        <li><strong>The heuristic's higher R² ({num(naive.R2, 2)} vs {num(model.R2, 2)}) is a level effect, not skill.</strong> Spending per pound of past spend was {num(d.past_spend_ratio_train, 2)} in the training period and {num(d.past_spend_ratio_test, 2)} in the test period. The "everyone stays" rule happens to use 0.50, so its totals come out right ({num(naive.total_pred_over_actual, 2)}), while anything fitted on train under-predicts. As a diagnostic only, since it uses test outcomes: model C rescaled to the actual total would reach R² {num(d.model_C_rescaled_to_actual_total.R2, 2)}. So the real gap between methods is small, and the level of spending shifting between periods matters more than the choice of model.</li>
        <li><strong>The lifespan assumption is the whole forecast.</strong> Fitting it on train gives {num(h.implied_lifespan_fraction, 2)} of the period instead of 1, and moving it moves every prediction. It is the weakest assumption in the heuristics.</li>
        <li><strong>When heuristics are enough:</strong> no usable history, a quick estimate for a board or a budget, or a sanity check on a model. <strong>When a model earns its place:</strong> here only about 3% lower average error than sorting by past spend, so the case for a model rests on explanation, segments and the calibrated probability of buying again, not on ranking.</li>
      </ul>
      <p className="src">Numbers computed by scripts/export_heuristics.py on the same snapshots as the rest of the site. No margin data exists, so everything is revenue.</p>
    </>
  );
}
