import type { ReactElement } from "react";
import { IconGrow, IconLow, IconProtect, IconWinBack } from "./Icons";
import { gbp, pct } from "./lib";
import type { Segment } from "./types";

// Colour class, icon and suggested action per segment. The actions are rules of thumb from the labels;
// the data shows the groups differ, not that the actions work (see Business decisions).
const META: Record<string, { cls: string; icon: ReactElement; action: string; note?: string }> = {
  Protect: { cls: "protect", icon: <IconProtect />, action: "Keep them happy. Avoid discounts: they buy anyway." },
  "Win back": { cls: "winback", icon: <IconWinBack />, action: "A tiny box in the model. A better list: high past spend, no purchase in 90 days.", note: "lead, not confirmed" },
  Grow: { cls: "grow", icon: <IconGrow />, action: "Upsell and cross-sell." },
  "Low priority": { cls: "low", icon: <IconLow />, action: "Do not spend effort here." },
};

export function SegmentCards({ segments }: { segments: Segment[] }) {
  return (
    <ul className="segcards">
      {segments.map((g) => {
        const m = META[g.segment] ?? { cls: "low", icon: <IconLow />, action: "" };
        return (
          <li key={g.segment} className={`segcard ${m.cls}`}>
            <div className="segHead"><span className="segIcon">{m.icon}</span><h4>{g.segment}</h4></div>
            <p className="segMoney">{pct(g["share of future money"])}<span> of future money</span></p>
            <p className="segSub">from {pct(g["share of customers"])} of customers ({g.customers.toLocaleString("en-GB")})</p>
            <div className="bars" aria-hidden="true">
              <div><span>customers</span><b><i style={{ width: `${Math.max(2, g["share of customers"] * 100)}%` }} /></b></div>
              <div><span>money</span><b><i style={{ width: `${Math.max(2, g["share of future money"] * 100)}%` }} /></b></div>
            </div>
            <dl className="segStats">
              <div><dt>Mean future spend</dt><dd>{gbp(g.mean_future_spend)}</dd></div>
              <div><dt>Bought again</dt><dd>{pct(g.bought_again)}</dd></div>
            </dl>
            <p className="segAction">{m.action}{m.note ? <em> ({m.note})</em> : null}</p>
          </li>
        );
      })}
    </ul>
  );
}
