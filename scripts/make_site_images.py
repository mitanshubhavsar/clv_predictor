"""Step 17b: figures for the dashboard and site, drawn from saved files only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "public" / "img"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e1",
                     "grid.linewidth": 0.6, "figure.dpi": 110, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor="white")
    plt.close(fig)


def capture():
    d = json.loads((ROOT / "site/public/data/capture_curve.json").read_text())
    style = {"Model C": (BLUE, "-", 2.2), "Sort by past spend": (ORANGE, "--", 1.8),
             "Recency only": (AQUA, "-", 1.6), "Perfect foresight": (GREY, ":", 1.4), "Random": (GREY, "-", 1)}
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for k, ys in d["curves"].items():
        c, ls, lw = style[k]
        ax.plot([g * 100 for g in d["grid"]], [y * 100 for y in ys], color=c, ls=ls, lw=lw, label=k)
    ax.axvline(20, color=GREY, lw=0.8)
    ax.set(xlabel="Customers contacted, best first (%)", ylabel="Future money captured (%)",
           title="Revenue capture on the test snapshot", xlim=(0, 100), ylim=(0, 102))
    ax.title.set_color(INK)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "capture_curves.png")


def comparison():
    df = pd.read_csv(ROOT / "outputs/model_comparison.csv", index_col=0)
    df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    cols = [BLUE if "Model C" in n else (ORANGE if "Past" in n else GREY) for n in df.index]
    ax.barh(df.index, df["MAE"], color=cols, height=0.6)
    for i, v in enumerate(df["MAE"]):
        ax.text(v + 8, i, f"£{v:,.0f}", va="center", color=MUTED, fontsize=9)
    ax.set(xlabel="Mean absolute error, 5-fold CV on train (GBP, lower is better)",
           title="Model error comparison", xlim=(0, df["MAE"].max() * 1.15))
    ax.title.set_color(INK); ax.grid(axis="y", visible=False)
    save(fig, "model_comparison.png")


def shap():
    df = pd.read_csv(ROOT / "outputs/shap_importance.csv").sort_values("share")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.barh(df["feature"], df["share"] * 100, color=BLUE, height=0.55)
    for i, v in enumerate(df["share"] * 100):
        ax.text(v + 0.6, i, f"{v:.0f}%", va="center", color=MUTED, fontsize=9)
    ax.set(xlabel="Share of total SHAP effect on the multiplier (%)", title="What drives the predictions",
           xlim=(0, df["share"].max() * 100 * 1.15))
    ax.title.set_color(INK); ax.grid(axis="y", visible=False)
    save(fig, "shap_importance.png")


def boxes(ax, items, y, w=2.0, h=0.9, gap=0.35, fill="#eef4fc"):
    x = 0.2
    for i, (title, sub) in enumerate(items):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fill, ec=BLUE, lw=1.2))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=10, color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8, color=MUTED)
        if i < len(items) - 1:
            ax.annotate("", xy=(x + w + gap, y + h / 2), xytext=(x + w, y + h / 2),
                        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.3))
        x += w + gap


def blank(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis("off"); ax.set_xlim(0, 14.4); ax.set_ylim(0, 2)
    return fig, ax


def pipeline():
    fig, ax = blank(12, 1.9)
    boxes(ax, [("Raw invoices", "1.07M lines"), ("Clean", "returns split off"), ("Ledger", "one row per line"),
               ("Snapshot", "cutoff date"), ("Features", "R, F, M, IPI, overdue"), ("Model C", "multiplier x spend")], 0.55)
    save(fig, "pipeline.png")


def timeline():
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    spans = [(0, "Train features", "2009-12 to 2010-11", BLUE, 0, 12), (1, "Train target", "2010-12 to 2011-05", ORANGE, 12, 18),
             (2, "Test features", "2010-06 to 2011-05", BLUE, 6, 18), (3, "Test target", "2011-06 to 2011-11", ORANGE, 18, 24)]
    for row, name, dates, c, a, b in spans:
        y = 3 - row
        ax.barh(y, b - a, left=a, height=0.6, color=c, alpha=0.85)
        if b - a >= 12:
            ax.text(a + 0.2, y, f"{name}  ({dates})", va="center", color="white", fontsize=9, weight="bold")
        else:
            ax.text(a - 0.2, y, f"{name}  ({dates})", va="center", ha="right", color=INK, fontsize=9, weight="bold")
    ax.set_xlim(0, 24); ax.set_ylim(-0.5, 3.6)
    ax.text(0, -0.45, "Dec 2009", color=MUTED, fontsize=8); ax.text(23.9, -0.45, "Nov 2011", color=MUTED, fontsize=8, ha="right")
    save(fig, "snapshot_timeline.png")


def serving():
    fig, ax = blank(12, 1.9)
    ax.set_xlim(0, 14.2)
    boxes(ax, [("Browser", "React site"), ("GitHub Pages", "static files"), ("FastAPI", "/score"),
               ("Model C", "loaded once"), ("Reply", "spend, segment, why")], 0.55, w=2.2, gap=0.45, fill="#fdf1ec")
    save(fig, "serving.png")


def flow():
    """The whole project as one picture: 13 stages in two rows."""
    labels = ["1 Raw data", "2 Clean,\nnet returns", "3 Time split", "4 RFM\nfeatures", "5 Study\nthe target", "6 Baselines",
              "7 Try models", "8 Test features\none by one", "9 Final model,\ntest once", "10 Segments", "11 Explain\n(SHAP)",
              "12 Targeting", "13 Package:\nAPI, dashboard,\nsite"]
    rows = [labels[:7], labels[7:]]
    w, h, gap = 1.7, 0.95, 0.35
    fig, ax = plt.subplots(figsize=(14, 4.4))
    ax.axis("off"); ax.set_xlim(0, 7 * (w + gap)); ax.set_ylim(0, 4.2)
    pos = []
    for r, row in enumerate(rows):
        y = 2.7 - r * 2.0
        for i, lab in enumerate(row):
            x = 0.1 + i * (w + gap)
            hi = lab.startswith(("7", "9"))
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc="#eef4fc" if not hi else "#fdf1ec", ec=BLUE if not hi else ORANGE, lw=1.3))
            ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=9.5, color=INK, weight="bold")
            pos.append((x, y))
    for (x1, y1), (x2, y2) in zip(pos, pos[1:]):
        if y1 == y2:
            ax.annotate("", xy=(x2, y2 + h / 2), xytext=(x1 + w, y1 + h / 2), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.3))
        else:  # wrap to the second row: down, left, down
            ym = (y1 + y2 + h) / 2
            ax.plot([x1 + w / 2, x1 + w / 2, x2 + w / 2], [y1, ym, ym], color=GREY, lw=1.3)
            ax.annotate("", xy=(x2 + w / 2, y2 + h), xytext=(x2 + w / 2, ym), arrowprops=dict(arrowstyle="->", color=GREY, lw=1.3))
    ax.text(0.1, 0.35, "Orange = where the plan changed: models only matched the simple rule, and the final model was scored on the later test period exactly once.",
            fontsize=9, color=MUTED)
    save(fig, "flow.png")


if __name__ == "__main__":
    for f in (capture, comparison, shap, pipeline, timeline, serving, flow):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
