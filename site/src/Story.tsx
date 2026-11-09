import { gbp, img, num, pct } from "./lib";
import type { Decision, FlowStage, Story, Summary } from "./types";

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
            <tr key={m.model}><td>{m.model}</td><td>{num(m.R2, 2)}</td><td>{gbp(m.MAE)}</td><td>{num(m["Capture@20%"], 2)}</td></tr>
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
            <h3><span className="num">{i + 1}</span> {st.title}</h3>
            <dl>
              <dt>What we did</dt><dd>{fill(st.did, s)}</dd>
              <dt>Why</dt><dd>{fill(st.why, s)}</dd>
              <dt>What we found</dt><dd>{fill(st.found, s)}</dd>
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
