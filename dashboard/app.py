"""Backtest inspector (Streamlit). Reads only saved files in outputs/ and site/public/data/.
It never refits a model or re-runs the pipeline, so it cannot disagree with the saved results.
Run:  streamlit run dashboard/app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "metrics": ROOT / "outputs/metrics.json",
    "summary": ROOT / "site/public/data/summary.json",
    "curve": ROOT / "site/public/data/capture_curve.json",
    "predictions": ROOT / "outputs/test_predictions.csv",
    "comparison": ROOT / "outputs/model_comparison.csv",
    "shap": ROOT / "outputs/shap_importance.csv",
}
IMG = ROOT / "site/public/img"
SEG_ORDER = ["Protect", "Win back", "Grow", "Low priority"]

st.set_page_config(page_title="CLV backtest inspector", layout="wide")
st.title("Customer spend forecast: backtest inspector")
st.caption("Reads saved files only. Rebuild them with src/step15_save.py and scripts/export_site_data.py.")

with st.sidebar:
    st.header("Saved artifacts")
    for name, path in FILES.items():
        st.write(("✅ " if path.exists() else "❌ ") + path.relative_to(ROOT).as_posix())
missing = [n for n, p in FILES.items() if not p.exists()]
if missing:
    st.error("Missing files: " + ", ".join(missing) + ". Run src/step15_save.py then scripts/export_site_data.py.")
    st.stop()


@st.cache_data
def load():
    j = {k: json.loads(FILES[k].read_text()) for k in ("metrics", "summary", "curve")}
    c = {k: pd.read_csv(FILES[k]) for k in ("predictions", "shap")}
    c["comparison"] = pd.read_csv(FILES["comparison"], index_col=0)
    return j, c


J, C = load()
S, preds = J["summary"], C["predictions"]
T = S["test"]


def fmt(m, money=False):
    v, (lo, hi) = m["value"], m["ci95"]
    p = (lambda x: f"£{x:,.0f}") if money else (lambda x: f"{x:.3f}")
    return p(v), f"95% interval {p(lo)} to {p(hi)}"


tab_over, tab_models, tab_seg, tab_cust, tab_explain, tab_limits = st.tabs(
    ["Overview", "Model comparison", "Segments", "Customers", "Explain", "Limits"])

with tab_over:
    st.subheader("Headline model C on the test snapshot (scored once)")
    st.write(f"Train cutoff {S['cutoff_train']} ({S['n_train']:,} customers), test cutoff {S['cutoff_test']} "
             f"({S['n_test']:,} customers, {S['zero_share_test']:.0%} spent nothing).")
    a, b, c = st.columns(3)
    for col, key, label, money in ((a, "R2", "R²", False), (b, "MAE", "MAE", True), (c, "Capture@20%", "Capture of money, top 20%", False)):
        val, note = fmt(T["model_C_headline"][key], money)
        col.metric(label, val, help=note)
        col.caption(note)
    st.markdown("**Against the simple rule 'sort by past spend'** (model C minus past spend, bootstrap over customers):")
    d = S["diff_vs_past_spend"]
    st.table(pd.DataFrame({k: {"difference": f"{v['point']:+.3f}", "95% interval": f"{v['ci'][0]:+.3f} to {v['ci'][1]:+.3f}"}
                           for k, v in d.items()}).T)
    st.info("The model is about 3% better than past spend in MAE, and equal in ranking. Treat it as a sizing tool, not a big upgrade.")
    t = S["targeting"]
    st.markdown(f"**Targeting the top 20%** by model C: £{t['model_C_mean_future_spend']:,.0f} each in the next 6 months vs "
                f"£{t['average_customer']:,.0f} for the average customer; they hold {t['model_C_share_of_money']:.0%} of future money. "
                f"Sorting by past spend gives £{t['past_spend_mean_future_spend']:,.0f}: no gain.")
    cur = J["curve"]
    cdf = pd.DataFrame(cur["curves"], index=cur["grid"])
    st.line_chart(cdf, x_label="Share of customers contacted", y_label="Share of future money")

with tab_models:
    st.subheader("Models tried (5-fold CV on train)")
    st.dataframe(C["comparison"].round(3), width="stretch")
    st.subheader("What drives the predictions (SHAP on the multiplier)")
    st.bar_chart(C["shap"].set_index("feature")["share"])
    for f in ("model_comparison.png", "shap_importance.png"):
        if (IMG / f).exists():
            st.image(str(IMG / f))

with tab_seg:
    st.subheader("Segments on the test snapshot")
    st.write(f"High value = predicted spend of £{S['high_value_cutoff_gbp']:,.0f} or more (fixed on train). "
             "Likely to return = probability of buying again above 0.5.")
    st.dataframe(pd.DataFrame(S["segments"]).set_index("segment"), width="stretch")

with tab_cust:
    st.subheader("Scored customers (test snapshot)")
    segs = st.multiselect("Segment", SEG_ORDER, default=SEG_ORDER)
    top = st.checkbox("Top 20% by predicted spend only")
    v = preds[preds["segment"].isin(segs)]
    if top:
        v = v[v["predicted_spend"] >= preds["predicted_spend"].quantile(0.8)]
    v = v.sort_values("predicted_spend", ascending=False)
    st.write(f"{len(v):,} customers")
    st.dataframe(v, width="stretch", hide_index=True)
    st.download_button("Download CSV", v.to_csv(index=False), "customers.csv", "text/csv")

with tab_explain:
    st.subheader("Explain one customer")
    cid = st.selectbox("Customer ID", preds.sort_values("predicted_spend", ascending=False)["Customer ID"].head(500))
    r = preds[preds["Customer ID"] == cid].iloc[0]
    st.write(f"Predicted spend **£{r.predicted_spend:,.0f}** = multiplier **{r.multiplier:.3f}** × max(past spend, £50) "
             f"(past spend £{r.Monetary:,.0f}). Probability of buying again {r.probability_buy_again:.0%}. Segment: **{r.segment}**. "
             f"Actual spend in the next 6 months: £{r.actual_spend:,.0f}.")
    st.dataframe(r[["Recency", "Frequency", "Monetary", "IPI", "OverdueRatio"]].to_frame("value"))
    st.caption("Why the multiplier moves is shown live by the API (/score → reasons).")

with tab_limits:
    st.subheader("Known limits")
    st.markdown("""
- Ranks who will spend; it does **not** say who is persuadable (no campaign data, not causal).
- Revenue, not margin.
- Only about 3% better than past spend in MAE; no gain in ranking.
- Ten giant customers hold about a fifth of the money and cause most of the error; treat them as named accounts.
- Two snapshots only; one retailer with Christmas-heavy seasons.
- The test snapshot was scored once for the model and later also used to describe segments and targeting.
""")
