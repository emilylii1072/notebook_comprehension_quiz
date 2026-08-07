"""Build an HTML report from the notebooks graded by the Notebook Grader.

`render_report()` returns ONE self-contained page (no external CSS, JS, fonts or
images), which is what lets the same output serve three uses: written to disk by
this script, embedded in the app's Report tab, and downloaded as a single file.
It contains:
    - score tiles, a sorted totals bar chart, per-section spread
    - a notebook x criterion heatmap and a hardest-criteria ranking
    - a PCA + k-means clustering of scoring profiles
    - a per-notebook section: every rubric item, its score, the grader's
      reasoning, and the transcript that was actually sent to the model

Reads the gradings the app already stores in Supabase (table `graded_notebooks`),
or a JSON file exported from them, so it needs no extra bookkeeping.

Usage:
    python build_report.py                          # rubric attrition_v1 -> report/
    python build_report.py --rubric my_rubric --open
    python build_report.py --json rows.json         # offline, no database
    python build_report.py --report-dir out/review
"""

import argparse
import html
import json
import math
import random
import re
import statistics
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RUBRIC = "attrition_v1"
DEFAULT_REPORT_DIR = HERE / "report"

# Nicer names for the section slugs the grader emits; anything not listed here is
# humanized automatically, so a new rubric needs no edit.
SECTION_LABELS = {
    "completeness": "Completeness",
    "simulating_attrition": "Simulating attrition",
    "model": "Model",
    "features": "Features",
    "evaluation": "Evaluation",
    "style_discussion": "Style & discussion",
}

# Sequential blue ramp, light->dark: magnitude encoding for every heatmap cell.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]

MAX_CLUSTERS = 3  # the categorical slots that clear the all-pairs CVD gate


def section_label(sec):
    return SECTION_LABELS.get(sec, sec.replace("_", " ").capitalize())


CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --baseline: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
    --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    --accent: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --baseline: #383835;
  --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  --accent: #3987e5;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 20px 24px 60px; }
h1 { font-size: 22px; margin: 8px 0 2px; }
h2 { font-size: 16px; margin: 34px 0 4px; }
h3 { font-size: 14px; margin: 18px 0 4px; }
.sub { color: var(--ink-2); margin: 0 0 12px; }
.hint { color: var(--muted); font-size: 12.5px; margin: 2px 0 10px; }
.card { background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px; overflow-x: auto; }
.tiles { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0 6px; }
.tile { background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 18px; min-width: 132px; }
.tile .v { font-size: 24px; font-weight: 650; }
.tile .l { color: var(--ink-2); font-size: 12px; }
svg { max-width: 100%; }
svg text { font: 11.5px system-ui, -apple-system, "Segoe UI", sans-serif; }
.gridline { stroke: var(--grid); stroke-width: 1; }
.baseline { stroke: var(--baseline); stroke-width: 1; }
.axislab { fill: var(--muted); }
.marklab { fill: var(--ink-2); }
table.scores { border-collapse: collapse; width: 100%; }
table.scores th { text-align: left; color: var(--ink-2); font-weight: 600;
  font-size: 12px; padding: 6px 10px; border-bottom: 1px solid var(--baseline); }
table.scores td { padding: 6px 10px; border-bottom: 1px solid var(--grid);
  vertical-align: top; }
table.scores td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
table.scores tr:hover td { background: color-mix(in srgb, var(--accent) 7%, transparent); }
.minibar { display: inline-block; width: 68px; height: 7px; background: var(--grid);
  border-radius: 4px; overflow: hidden; vertical-align: 1px; margin-right: 6px; }
.minibar i { display: block; height: 100%; background: var(--series-1);
  border-radius: 4px; }
.chip { display: inline-block; width: 10px; height: 10px; border-radius: 3px;
  margin-right: 6px; vertical-align: -1px; }
.lg { display: inline-flex; align-items: center; gap: 4px; color: var(--ink-2);
  font-size: 12.5px; margin-right: 12px; }
pre.nb { background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px; overflow-x: auto; white-space: pre-wrap;
  font: 12.5px/1.55 ui-monospace, SFMono-Regular, Consolas, monospace;
  color: var(--ink-2); }
details > summary { cursor: pointer; color: var(--accent); margin: 10px 0; }
#tip { position: fixed; display: none; pointer-events: none; z-index: 50;
  background: var(--surface); color: var(--ink); border: 1px solid var(--border);
  box-shadow: 0 4px 14px rgba(0,0,0,.18); border-radius: 8px;
  padding: 7px 10px; font-size: 12.5px; max-width: 340px; }
"""

TIP_JS = """
<div id="tip"></div>
<script>
const tip = document.getElementById('tip');
document.addEventListener('mouseover', e => {
  const el = e.target.closest('[data-tip]');
  if (!el) { tip.style.display = 'none'; return; }
  tip.innerHTML = el.dataset.tip;
  tip.style.display = 'block';
});
document.addEventListener('mousemove', e => {
  if (tip.style.display !== 'block') return;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth - 8) x = e.clientX - w - pad;
  if (y + h > innerHeight - 8) y = e.clientY - h - pad;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
});
</script>
"""

esc = html.escape


def attr(s):
    return esc(str(s), quote=True)


def page(title, body):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{esc(title)}</title><style>{CSS}</style></head><body>'
            f'<div class="wrap">{body}</div>{TIP_JS}</body></html>')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def slug(filename):
    """Notebook filename -> safe page name ('A.ipynb' -> 'A')."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", Path(filename).stem) or "notebook"


def load_from_db(rubric_name):
    """Pull gradings through db.py, so table names and credential handling stay
    in one place. Streamlit is imported as a side effect of that module and warns
    about running outside a script run; the warning is meaningless here."""
    import logging

    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")  # db.py reads credentials from the environment
    sys.path.insert(0, str(HERE))
    from db import get_graded_notebooks

    # Streamlit reconfigures logging when imported, so quiet it only afterwards.
    logging.getLogger(
        "streamlit.runtime.scriptrunner_utils.script_run_context"
    ).setLevel(logging.ERROR)
    rows = get_graded_notebooks(rubric_name, include_text=True)
    if not rows:
        sys.exit(f"No graded notebooks for rubric '{rubric_name}'. Grade some in "
                 f"the app first, or pass --json.")
    return rows


def load_from_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["rows"] if isinstance(data, dict) else data


def normalize(rows):
    """DB rows -> records the report uses, sorted by total score descending."""
    records = []
    for row in rows:
        items = [{"section": it["section"], "criterion": it["criterion"],
                  "score": float(it["score"]), "max_pts": float(it["max_pts"]),
                  "reasoning": (it.get("reasoning") or "").strip()}
                 for it in row["results"]]
        total = float(row.get("total_score") or sum(i["score"] for i in items))
        max_score = float(row.get("max_score") or sum(i["max_pts"] for i in items))
        records.append({
            "name": row["notebook_filename"],
            "slug": slug(row["notebook_filename"]),
            "items": items,
            "total": total,
            "max": max_score,
            "pct": 100 * total / max_score if max_score else 0.0,
            "text": row.get("notebook_text") or "",
            "by_key": {(i["section"], i["criterion"]): i for i in items},
        })
    return sorted(records, key=lambda r: -r["total"])


def records_from_graded(graded):
    """The app's in-session shape ({filename: {results, total_score, max_score}})
    -> report records, so the Report tab and this CLI share one code path."""
    return normalize([{"notebook_filename": name,
                       "results": rec["results"],
                       "total_score": rec.get("total_score"),
                       "max_score": rec.get("max_score")}
                      for name, rec in graded.items()])


def columns_of(records):
    """Union of (section, criterion) across notebooks, first-seen order, with
    each section's items kept contiguous so the heatmap groups cleanly."""
    order, seen = [], set()
    for rec in records:
        for it in rec["items"]:
            key = (it["section"], it["criterion"])
            if key not in seen:
                seen.add(key)
                order.append(key)
    grouped, sections = [], []
    for sec, crit in order:
        if sec not in sections:
            sections.append(sec)
    for sec in sections:
        grouped.extend((s, c) for s, c in order if s == sec)
    return grouped, sections


def col_max(records, key):
    """Max points for a column — the max seen, so one odd grading can't shrink it."""
    return max((rec["by_key"][key]["max_pts"] for rec in records
                if key in rec["by_key"]), default=0.0)


def section_totals(rec, sections):
    out = {}
    for sec in sections:
        got = [i for i in rec["items"] if i["section"] == sec]
        out[sec] = {"score": sum(i["score"] for i in got),
                    "max": sum(i["max_pts"] for i in got)}
    return out


# ---------------------------------------------------------------------------
# Analytics — PCA + k-means in plain Python, so the report needs no extra deps
# ---------------------------------------------------------------------------

def score_matrix(records, columns):
    """Rows = notebooks, cols = criteria, values = fraction of the item's max."""
    mat = []
    for rec in records:
        row = []
        for key in columns:
            mx = col_max(records, key)
            item = rec["by_key"].get(key)
            row.append(item["score"] / mx if item and mx else 0.0)
        mat.append(row)
    return mat


def _center(mat):
    n, m = len(mat), len(mat[0])
    means = [sum(r[j] for r in mat) / n for j in range(m)]
    return [[r[j] - means[j] for j in range(m)] for r in mat]


def _covariance(x):
    m = len(x[0])
    return [[sum(r[a] * r[b] for r in x) for b in range(m)] for a in range(m)]


def _top_eigenvector(cov, iters=300):
    """Power iteration — enough for the leading component of a 20ish x 20ish matrix."""
    m = len(cov)
    v = [1.0 / math.sqrt(m)] * m
    for _ in range(iters):
        w = [sum(cov[a][b] * v[b] for b in range(m)) for a in range(m)]
        norm = math.sqrt(sum(c * c for c in w))
        if norm < 1e-12:
            return v, 0.0
        v = [c / norm for c in w]
    lam = sum(v[a] * sum(cov[a][b] * v[b] for b in range(m)) for a in range(m))
    return v, lam


def pca_2d(mat):
    """Project onto the first two principal components. Returns (xy, evr)."""
    x = _center(mat)
    cov = _covariance(x)
    trace = sum(cov[a][a] for a in range(len(cov))) or 1.0
    v1, l1 = _top_eigenvector(cov)
    deflated = [[cov[a][b] - l1 * v1[a] * v1[b] for b in range(len(cov))]
                for a in range(len(cov))]
    v2, l2 = _top_eigenvector(deflated)
    xy = [[sum(r[j] * v1[j] for j in range(len(r))),
           sum(r[j] * v2[j] for j in range(len(r)))] for r in x]
    return xy, [l1 / trace, l2 / trace]


def _dist2(a, b):
    return sum((p - q) ** 2 for p, q in zip(a, b))


def _kmeans(mat, k, seed=0, restarts=10, iters=60):
    """k-means++ seeding, best of `restarts` by inertia. Deterministic."""
    rng = random.Random(seed)
    best = None
    for _ in range(restarts):
        centers = [list(mat[rng.randrange(len(mat))])]
        while len(centers) < k:
            d = [min(_dist2(p, c) for c in centers) for p in mat]
            total = sum(d)
            if total <= 0:
                centers.append(list(mat[rng.randrange(len(mat))]))
                continue
            pick, acc = rng.random() * total, 0.0
            for i, dv in enumerate(d):
                acc += dv
                if acc >= pick:
                    centers.append(list(mat[i]))
                    break
        labels = [0] * len(mat)
        for _ in range(iters):
            new = [min(range(k), key=lambda c: _dist2(p, centers[c])) for p in mat]
            if new == labels:
                break
            labels = new
            for c in range(k):
                members = [mat[i] for i in range(len(mat)) if labels[i] == c]
                if members:
                    centers[c] = [sum(vals) / len(members) for vals in zip(*members)]
        inertia = sum(_dist2(mat[i], centers[labels[i]]) for i in range(len(mat)))
        if best is None or inertia < best[0]:
            best = (inertia, labels)
    return best[1]


def _silhouette(mat, labels):
    n = len(mat)
    groups = {}
    for i, c in enumerate(labels):
        groups.setdefault(c, []).append(i)
    if len(groups) < 2:
        return -1.0
    scores = []
    for i in range(n):
        own = groups[labels[i]]
        if len(own) <= 1:
            scores.append(0.0)
            continue
        a = sum(math.sqrt(_dist2(mat[i], mat[j])) for j in own if j != i) / (len(own) - 1)
        b = min(sum(math.sqrt(_dist2(mat[i], mat[j])) for j in idx) / len(idx)
                for c, idx in groups.items() if c != labels[i])
        scores.append(0.0 if max(a, b) == 0 else (b - a) / max(a, b))
    return sum(scores) / n


def cluster(mat):
    """PCA + k-means with k picked by silhouette. None when too few notebooks."""
    if len(mat) < 5:
        return None
    xy, evr = pca_2d(mat)
    best = None
    for k in range(2, min(MAX_CLUSTERS, len(mat) - 1) + 1):
        labels = _kmeans(mat, k)
        if len(set(labels)) < 2:
            continue
        sil = _silhouette(mat, labels)
        if best is None or sil > best["sil"]:
            best = {"k": k, "sil": sil, "labels": labels}
    if best is None:
        return None
    return {"xy": xy, "evr": evr, **best}


def cluster_traits(records, mat, columns, labels, k):
    """Per cluster: members, plus the criteria furthest above/below the mean."""
    m = len(columns)
    overall = [sum(r[j] for r in mat) / len(mat) for j in range(m)]
    out = []
    for c in range(k):
        idx = [i for i in range(len(records)) if labels[i] == c]
        rows = [mat[i] for i in idx]
        mean = [sum(r[j] for r in rows) / len(rows) for j in range(m)]
        diffs = sorted(range(m), key=lambda j: mean[j] - overall[j])
        fmt = lambda j: f"{columns[j][1]} ({(mean[j]-overall[j])*100:+.0f}pp)"
        out.append({
            "members": [records[i]["name"] for i in idx],
            "mean_pct": sum(records[i]["pct"] for i in idx) / len(idx),
            "high": [fmt(j) for j in reversed(diffs[-2:])],
            "low": [fmt(j) for j in diffs[:2]],
        })
    return out


# ---------------------------------------------------------------------------
# SVG charts
# ---------------------------------------------------------------------------

def ramp_color(frac):
    idx = min(len(BLUE_RAMP) - 1, max(0, int(frac * len(BLUE_RAMP))))
    return BLUE_RAMP[idx]


def rounded_bar_path(x, y, w, h, r=4):
    """Bar with rounded data-end, square where it meets the baseline."""
    r = max(0.0, min(r, w / 2, h))
    return (f"M{x:.1f},{y+h:.1f} L{x:.1f},{y+r:.1f} Q{x:.1f},{y:.1f} "
            f"{x+r:.1f},{y:.1f} L{x+w-r:.1f},{y:.1f} Q{x+w:.1f},{y:.1f} "
            f"{x+w:.1f},{y+r:.1f} L{x+w:.1f},{y+h:.1f} Z")


def chart_totals(records):
    """Total score per notebook, sorted; bars link to their detail page."""
    n = len(records)
    W, H, ml, mb, mt = 960, 300, 42, 30, 18
    plot_w, plot_h = W - ml - 14, H - mt - mb
    maxv = max(r["max"] for r in records) or 100
    step = plot_w / n
    bw = max(6.0, min(34.0, step - 8))  # >=2px surface gap between adjacent bars
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             f'aria-label="Total score per notebook">']
    tick = 25 if maxv >= 50 else 10
    for gv in range(0, int(maxv) + 1, tick):
        y = mt + plot_h * (1 - gv / maxv)
        parts.append(f'<line class="gridline" x1="{ml}" y1="{y:.1f}" '
                     f'x2="{W-14}" y2="{y:.1f}"/>')
        parts.append(f'<text class="axislab" x="{ml-6}" y="{y+4:.1f}" '
                     f'text-anchor="end">{gv}</text>')
    for i, rec in enumerate(records):
        frac = rec["total"] / maxv if maxv else 0
        bh = max(2.0, plot_h * frac)
        x = ml + i * step + (step - bw) / 2
        y = mt + plot_h - bh
        tipt = (f"<b>{esc(rec['name'])}</b> — {rec['total']:g}/{rec['max']:g} "
                f"({rec['pct']:.0f}%)<br>click to open its review page")
        parts.append(f'<a href="#{rec["slug"]}">'
                     f'<path d="{rounded_bar_path(x, y, bw, bh)}" '
                     f'fill="var(--series-1)" data-tip="{attr(tipt)}"/></a>')
        if i in (0, n - 1):  # direct-label the extremes only
            parts.append(f'<text class="marklab" x="{x+bw/2:.1f}" y="{y-6:.1f}" '
                         f'text-anchor="middle">{rec["total"]:g}</text>')
        parts.append(f'<text class="axislab" x="{x+bw/2:.1f}" y="{H-10}" '
                     f'text-anchor="middle">{esc(Path(rec["name"]).stem[:8])}</text>')
    yb = mt + plot_h
    parts.append(f'<line class="baseline" x1="{ml}" y1="{yb}" x2="{W-14}" y2="{yb}"/>')
    parts.append("</svg>")
    return "".join(parts)


def chart_section_spread(records, sections):
    """One row per section: a dot per notebook at % of that section's max."""
    W, row_h, ml = 960, 40, 176
    H = row_h * len(sections) + 32
    plot_w = W - ml - 44
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             f'aria-label="Score spread by rubric section">']
    for gv in (0, 25, 50, 75, 100):
        x = ml + plot_w * gv / 100
        parts.append(f'<line class="gridline" x1="{x:.1f}" y1="8" x2="{x:.1f}" '
                     f'y2="{H-24}"/>')
        parts.append(f'<text class="axislab" x="{x:.1f}" y="{H-8}" '
                     f'text-anchor="middle">{gv}%</text>')
    for si, sec in enumerate(sections):
        cy = 8 + row_h * si + row_h / 2
        parts.append(f'<text class="marklab" x="{ml-10}" y="{cy+4:.1f}" '
                     f'text-anchor="end">{esc(section_label(sec))}</text>')
        fracs = []
        for rec in records:
            st = section_totals(rec, [sec])[sec]
            if not st["max"]:
                continue
            f = st["score"] / st["max"]
            fracs.append(f)
            tipt = (f"<b>{esc(rec['name'])}</b> — {esc(section_label(sec))}: "
                    f"{st['score']:g}/{st['max']:g} ({f*100:.0f}%)")
            parts.append(f'<a href="#{rec["slug"]}"><circle '
                         f'cx="{ml + plot_w*f:.1f}" cy="{cy:.1f}" r="5" '
                         f'fill="var(--series-1)" fill-opacity="0.55" '
                         f'stroke="var(--surface)" stroke-width="2" '
                         f'data-tip="{attr(tipt)}"/></a>')
        if fracs:
            mean = sum(fracs) / len(fracs)
            mx = ml + plot_w * mean
            parts.append(f'<line x1="{mx:.1f}" y1="{cy-10:.1f}" x2="{mx:.1f}" '
                         f'y2="{cy+10:.1f}" stroke="var(--ink)" stroke-width="2" '
                         f'data-tip="{attr(f"cohort mean {mean*100:.0f}%")}"/>')
    parts.append("</svg>")
    legend = ('<p class="hint"><span class="lg"><span class="chip" '
              'style="background:var(--series-1)"></span>one notebook</span>'
              '<span class="lg"><span class="chip" style="background:var(--ink); '
              'width:3px; height:12px; border-radius:1px"></span>cohort mean</span>'
              'Dots are jittered only by their value, so overlap means agreement.</p>')
    return "".join(parts) + legend


def chart_heatmap(records, columns, sections):
    """Notebooks (rows, sorted by total) x criteria; cell shade = % of item max."""
    cw, ch, ml, mt, gap, secgap = 46, 21, 128, 128, 2, 10
    xs, x, prev = [], ml, None
    for sec, crit in columns:
        if prev is not None and sec != prev:
            x += secgap
        xs.append(x)
        x += cw + gap
        prev = sec
    # Right pad for the -45 degree labels, which run up and to the RIGHT of the
    # last column and would otherwise be clipped by the viewBox.
    W, H = x + 104, mt + len(records) * (ch + gap) + 12
    parts = [f'<svg viewBox="0 0 {W} {H}" width="{W}" role="img" '
             f'aria-label="Notebook by criterion score heatmap">']
    for j, (sec, crit) in enumerate(columns):
        cx = xs[j] + cw / 2
        tip = f"<b>{esc(crit)}</b><br>{esc(section_label(sec))} · " \
              f"{col_max(records, (sec, crit)):g} pts max"
        parts.append(f'<text class="axislab" transform="rotate(-45 {cx:.1f} '
                     f'{mt-8})" x="{cx:.1f}" y="{mt-8}" text-anchor="start" '
                     f'data-tip="{attr(tip)}">{esc(crit[:22])}</text>')
    for i, rec in enumerate(records):
        y = mt + i * (ch + gap)
        parts.append(f'<a href="#{rec["slug"]}"><text class="marklab" '
                     f'x="{ml-8}" y="{y+ch-6}" text-anchor="end">'
                     f'{esc(rec["name"][:18])}</text></a>')
        for j, key in enumerate(columns):
            item = rec["by_key"].get(key)
            mx = col_max(records, key)
            if item is None:
                parts.append(f'<rect x="{xs[j]}" y="{y}" width="{cw}" '
                             f'height="{ch}" rx="3" fill="var(--grid)" '
                             f'data-tip="{attr("not graded for this notebook")}"/>')
                continue
            frac = item["score"] / mx if mx else 0.0
            tipt = (f"<b>{esc(rec['name'])} · {esc(key[1])}</b> — "
                    f"{item['score']:g}/{item['max_pts']:g}<br>"
                    f"{esc(item['reasoning'][:200])}")
            parts.append(f'<a href="#{rec["slug"]}">'
                         f'<rect x="{xs[j]}" y="{y}" width="{cw}" height="{ch}" '
                         f'rx="3" fill="{ramp_color(frac)}" '
                         f'data-tip="{attr(tipt)}"/></a>')
    parts.append("</svg>")
    steps = "".join(f'<span style="display:inline-block; width:16px; height:10px; '
                    f'background:{c}"></span>' for c in BLUE_RAMP)
    legend = (f'<p class="hint">0% {steps} 100% of the item\'s max. '
              f'Hover a cell for the grader\'s reasoning; click to open that '
              f'notebook\'s page at the item. Gray = the grader returned no such '
              f'item for that notebook.</p>')
    return "".join(parts) + legend


def chart_criterion_difficulty(records, columns):
    """Criteria ranked by cohort mean — which parts of the task went worst."""
    rows = []
    for key in columns:
        mx = col_max(records, key)
        got = [rec["by_key"][key]["score"] / mx for rec in records
               if key in rec["by_key"] and mx]
        if got:
            rows.append((key, sum(got) / len(got), min(got), max(got), len(got)))
    rows.sort(key=lambda r: r[1])
    W, row_h, ml = 960, 26, 210
    H = row_h * len(rows) + 34
    plot_w = W - ml - 60
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             f'aria-label="Criteria ranked by cohort mean score">']
    for gv in (0, 25, 50, 75, 100):
        x = ml + plot_w * gv / 100
        parts.append(f'<line class="gridline" x1="{x:.1f}" y1="6" x2="{x:.1f}" '
                     f'y2="{H-26}"/>')
        parts.append(f'<text class="axislab" x="{x:.1f}" y="{H-10}" '
                     f'text-anchor="middle">{gv}%</text>')
    for i, (key, mean, lo, hi, n) in enumerate(rows):
        y = 6 + i * row_h
        bh = row_h - 8  # >=2px surface gap between adjacent bars
        bw = max(1.5, plot_w * mean)
        parts.append(f'<text class="marklab" x="{ml-10}" y="{y+bh-3:.1f}" '
                     f'text-anchor="end">{esc(key[1][:26])}</text>')
        tipt = (f"<b>{esc(key[1])}</b> — {esc(section_label(key[0]))}<br>"
                f"cohort mean {mean*100:.0f}% of {col_max(records, key):g} pts"
                f"<br>range {lo*100:.0f}%–{hi*100:.0f}% across {n} notebooks")
        # Horizontal bar: rounded data-end at the right, square at the baseline.
        r = min(4.0, bw)
        parts.append(f'<path d="M{ml},{y:.1f} L{ml+bw-r:.1f},{y:.1f} '
                     f'Q{ml+bw:.1f},{y:.1f} {ml+bw:.1f},{y+r:.1f} '
                     f'L{ml+bw:.1f},{y+bh-r:.1f} Q{ml+bw:.1f},{y+bh:.1f} '
                     f'{ml+bw-r:.1f},{y+bh:.1f} L{ml},{y+bh:.1f} Z" '
                     f'fill="{ramp_color(mean)}" data-tip="{attr(tipt)}"/>')
        parts.append(f'<text class="marklab" x="{ml+bw+8:.1f}" '
                     f'y="{y+bh-3:.1f}">{mean*100:.0f}%</text>')
    parts.append(f'<line class="baseline" x1="{ml}" y1="6" x2="{ml}" '
                 f'y2="{H-26}"/>')
    parts.append("</svg>")
    return "".join(parts)


def chart_clusters(records, cl):
    """PCA projection of scoring profiles, colored by k-means cluster."""
    xy, labels = cl["xy"], cl["labels"]
    W, H, m = 760, 460, 52
    xsv = [p[0] for p in xy]
    ysv = [p[1] for p in xy]
    xmin, xmax = min(xsv), max(xsv)
    ymin, ymax = min(ysv), max(ysv)
    xpad = (xmax - xmin or 1) * 0.14
    ypad = (ymax - ymin or 1) * 0.14
    xmin, xmax, ymin, ymax = xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad
    sx = lambda v: m + (v - xmin) / (xmax - xmin) * (W - 2 * m)
    sy = lambda v: H - m - (v - ymin) / (ymax - ymin) * (H - 2 * m)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             f'aria-label="Notebook clustering, PCA projection">',
             f'<line class="baseline" x1="{m}" y1="{H-m}" x2="{W-m}" y2="{H-m}"/>',
             f'<line class="baseline" x1="{m}" y1="{m}" x2="{m}" y2="{H-m}"/>',
             f'<text class="axislab" x="{W/2}" y="{H-14}" text-anchor="middle">'
             f'PC1 ({cl["evr"][0]*100:.0f}% of variance)</text>',
             f'<text class="axislab" transform="rotate(-90 18 {H/2})" x="18" '
             f'y="{H/2}" text-anchor="middle">PC2 ({cl["evr"][1]*100:.0f}%)</text>']
    for i, rec in enumerate(records):
        x, y = sx(xy[i][0]), sy(xy[i][1])
        c = labels[i] % MAX_CLUSTERS + 1
        tipt = (f"<b>{esc(rec['name'])}</b> — cluster {labels[i]+1}, "
                f"{rec['total']:g}/{rec['max']:g} ({rec['pct']:.0f}%)")
        parts.append(f'<a href="#{rec["slug"]}"><circle cx="{x:.1f}" '
                     f'cy="{y:.1f}" r="7" fill="var(--series-{c})" '
                     f'stroke="var(--surface)" stroke-width="2" '
                     f'data-tip="{attr(tipt)}"/></a>')
        # Direct label every point: identity never rests on hue alone.
        parts.append(f'<text class="marklab" x="{x+11:.1f}" y="{y+4:.1f}">'
                     f'{esc(Path(rec["name"]).stem[:8])}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def tiles(records):
    pcts = [r["pct"] for r in records]
    spread = f"{min(pcts):.0f}–{max(pcts):.0f}%"
    stdev = statistics.pstdev(pcts) if len(pcts) > 1 else 0.0
    cells = [("Notebooks", str(len(records))),
             ("Mean", f"{statistics.mean(pcts):.0f}%"),
             ("Median", f"{statistics.median(pcts):.0f}%"),
             ("Spread", spread),
             ("Std dev", f"{stdev:.1f}pp")]
    return ('<div class="tiles">' + "".join(
        f'<div class="tile"><div class="v">{esc(v)}</div>'
        f'<div class="l">{esc(l)}</div></div>' for l, v in cells) + "</div>")


def index_table(records, sections):
    head = ("<tr><th>Notebook</th><th>Total</th><th>%</th>"
            + "".join(f"<th>{esc(section_label(s))}</th>" for s in sections)
            + "</tr>")
    rows = []
    for rec in records:
        st = section_totals(rec, sections)
        cells = "".join(
            f'<td class="num">{st[s]["score"]:g}/{st[s]["max"]:g}</td>'
            for s in sections)
        rows.append(
            f'<tr><td><a href="#{rec["slug"]}">{esc(rec["name"])}</a></td>'
            f'<td class="num"><span class="minibar"><i style="width:'
            f'{rec["pct"]:.0f}%"></i></span>{rec["total"]:g}/{rec["max"]:g}</td>'
            f'<td class="num">{rec["pct"]:.0f}%</td>{cells}</tr>')
    return f'<table class="scores">{head}{"".join(rows)}</table>'


def build_dashboard(records, columns, sections, rubric_name):
    mat = score_matrix(records, columns)
    cl = cluster(mat)
    body = [f'<h1>Notebook grading report</h1>',
            f'<p class="sub">Rubric <b>{esc(rubric_name)}</b> · '
            f'{len(records)} notebooks · {len(columns)} rubric items · built '
            f'{datetime.now():%Y-%m-%d %H:%M}</p>',
            tiles(records),
            '<h2>Total score by notebook</h2>',
            '<p class="hint">Sorted high to low. Click a bar to jump to that '
            'notebook\'s breakdown below.</p>',
            f'<div class="card">{chart_totals(records)}</div>',
            '<h2>Where the scores spread</h2>',
            '<p class="hint">Each section as a percentage of its own maximum, so '
            'sections with different point weights are comparable.</p>',
            f'<div class="card">{chart_section_spread(records, sections)}</div>',
            '<h2>Notebook × criterion</h2>',
            f'<div class="card">{chart_heatmap(records, columns, sections)}</div>',
            '<h2>Hardest criteria</h2>',
            '<p class="hint">Cohort mean per rubric item, worst first — the items '
            'at the top are where this set of notebooks lost the most.</p>',
            f'<div class="card">{chart_criterion_difficulty(records, columns)}</div>']
    if cl:
        traits = cluster_traits(records, mat, columns, cl["labels"], cl["k"])
        legend = "".join(
            f'<span class="lg"><span class="chip" style="background:'
            f'var(--series-{c % MAX_CLUSTERS + 1})"></span>Cluster {c+1} '
            f'(n={len(traits[c]["members"])})</span>' for c in range(cl["k"]))
        cards = "".join(
            f'<div class="tile" style="flex:1; min-width:260px">'
            f'<div class="l">Cluster {c+1} · mean {t["mean_pct"]:.0f}%</div>'
            f'<div style="margin:4px 0">{esc(", ".join(t["members"]))}</div>'
            f'<div class="l">above cohort: {esc(", ".join(t["high"]))}</div>'
            f'<div class="l">below cohort: {esc(", ".join(t["low"]))}</div></div>'
            for c, t in enumerate(traits))
        body += ['<h2>Scoring profiles</h2>',
                 f'<p class="hint">Each notebook is a point in 19-dimensional '
                 f'rubric space, projected to 2D by PCA and grouped by k-means '
                 f'(k={cl["k"]}, silhouette {cl["sil"]:.2f}). Clusters describe '
                 f'this sample only — with {len(records)} notebooks, read them as '
                 f'a reading aid, not a finding.</p>',
                 f'<p>{legend}</p>',
                 f'<div class="card">{chart_clusters(records, cl)}</div>',
                 f'<div class="tiles">{cards}</div>']
    else:
        body += ['<h2>Scoring profiles</h2>',
                 '<p class="hint">Clustering needs at least 5 graded notebooks.</p>']
    body += ['<h2>All notebooks</h2>',
             f'<div class="card">{index_table(records, sections)}</div>']
    return "".join(body)


def detail_section(rec, sections):
    """One notebook's full breakdown, as an anchored in-page section."""
    st = section_totals(rec, sections)
    sec_rows = "".join(
        f'<tr><td>{esc(section_label(s))}</td>'
        f'<td class="num"><span class="minibar"><i style="width:'
        f'{100*st[s]["score"]/st[s]["max"] if st[s]["max"] else 0:.0f}%"></i>'
        f'</span>{st[s]["score"]:g}/{st[s]["max"]:g}</td></tr>'
        for s in sections)
    item_rows = []
    for sec in sections:
        items = [i for i in rec["items"] if i["section"] == sec]
        if not items:
            continue
        item_rows.append(f'<tr><td colspan="3" style="padding-top:14px">'
                         f'<b>{esc(section_label(sec))}</b></td></tr>')
        for it in items:
            frac = it["score"] / it["max_pts"] if it["max_pts"] else 0
            item_rows.append(
                f'<tr><td>{esc(it["criterion"])}</td>'
                f'<td class="num"><span class="minibar"><i style="width:'
                f'{frac*100:.0f}%"></i></span>{it["score"]:g}/{it["max_pts"]:g}</td>'
                f'<td>{esc(it["reasoning"])}</td></tr>')
    transcript = (f'<details><summary>Transcript as sent to the grader '
                  f'({len(rec["text"]):,} chars)</summary>'
                  f'<pre class="nb">{esc(rec["text"])}</pre></details>'
                  if rec["text"] else "")
    return (f'<section id="{rec["slug"]}">'
            f'<h3 style="font-size:17px; margin-top:30px">{esc(rec["name"])} '
            f'<span class="hint" style="display:inline">{rec["total"]:g}/'
            f'{rec["max"]:g} ({rec["pct"]:.0f}%) · '
            f'<a href="#top">back to top</a></span></h3>'
            f'<div class="card"><table class="scores">'
            f'<tr><th>Criterion</th><th>Score</th><th>Grader reasoning</th></tr>'
            f'{"".join(item_rows)}'
            f'<tr><td><b>Section totals</b></td><td colspan="2">'
            f'<table class="scores" style="width:auto">{sec_rows}</table>'
            f'</td></tr></table>{transcript}</div></section>')


def render_report(records, rubric_name):
    """The whole report as one self-contained HTML page."""
    columns, sections = columns_of(records)
    body = (f'<a id="top"></a>'
            + build_dashboard(records, columns, sections, rubric_name)
            + '<h2>Per-notebook breakdown</h2>'
            + '<p class="hint">Every rubric item with the grader\'s reasoning.</p>'
            + "".join(detail_section(rec, sections) for rec in records))
    return page(f"Grading report — {rubric_name}", body)


def build(records, rubric_name, report_dir):
    """Write the report to <report_dir>/index.html. Returns that path."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "index.html"
    out.write_text(render_report(records, rubric_name), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rubric", default=DEFAULT_RUBRIC,
                    help=f"rubric name to report on (default {DEFAULT_RUBRIC})")
    ap.add_argument("--json", help="read rows from a JSON file instead of Supabase")
    ap.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    ap.add_argument("--open", action="store_true", help="open the report when done")
    args = ap.parse_args()

    rows = load_from_json(args.json) if args.json else load_from_db(args.rubric)
    records = normalize(rows)
    index = build(records, args.rubric, Path(args.report_dir))
    print(f"Wrote {index} ({len(records)} notebooks, {index.stat().st_size:,} bytes).")
    if args.open:
        webbrowser.open(index.resolve().as_uri())


if __name__ == "__main__":
    main()
