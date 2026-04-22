"""Build a self-contained HTML dashboard from the derived CSVs.

Uses Plotly JS via CDN — no Python plotly dep needed. Output is
world_spec/viz/dashboard.html, openable in any browser.

Sections:
    1. ARR over time (quarterly step + IPO marker)
    2. Headcount over time
    3. GPU fleet over time
    4. Valuation (pre-IPO) + market cap (post-IPO) on one chart
    5. Daily stock price post-IPO with funding-round markers
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

VIZ_DIR = Path(__file__).parent
FINANCIALS_CSV = VIZ_DIR / "financials.csv"
VALUATION_CSV = VIZ_DIR / "valuation.csv"
STOCK_CSV = VIZ_DIR / "stock_prices.csv"
OUT_HTML = VIZ_DIR / "dashboard.html"


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    fin = load_csv(FINANCIALS_CSV)
    val = load_csv(VALUATION_CSV)
    stk = load_csv(STOCK_CSV)

    # Series
    fin_dates = [r["date"] for r in fin]
    arr = [float(r["arr_m"]) for r in fin]
    fte = [int(r["fte"]) for r in fin]
    gpus = [int(r["gpus_own"]) for r in fin]
    events = [r["event"] for r in fin]
    notes = [r["notes"] for r in fin]

    val_dates = [r["date"] for r in val]
    post_money = [float(r["post_money_m"]) for r in val]

    stk_dates = [r["date"] for r in stk]
    stk_close = [float(r["close"]) for r in stk]
    stk_high = [float(r["high"]) for r in stk]
    stk_low = [float(r["low"]) for r in stk]
    stk_mcap = [float(r["mcap_m"]) for r in stk]

    # Round markers on stock chart — big vertical lines for funding events
    rounds = [
        ("2030-03-18", "IPO $32/share"),
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Echoblast canonical financials dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 1400px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }}
  h1 {{ font-weight: 600; margin: 0 0 4px 0; }}
  h2 {{ font-weight: 500; margin: 28px 0 8px 0; border-bottom: 1px solid #eee; padding-bottom: 4px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 20px; }}
  .chart {{ margin-bottom: 32px; }}
  table {{ border-collapse: collapse; font-size: 13px; }}
  td, th {{ padding: 4px 10px; border: 1px solid #ddd; text-align: right; }}
  th {{ background: #f5f5f5; text-align: left; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .event {{ color: #444; font-style: italic; }}
</style>
</head>
<body>

<h1>Echoblast — derived financials dashboard</h1>
<p class="meta">Source of truth: <code>docs/spec.md §3.1</code>. Regenerate: <code>python -m world_spec.derived.financials &amp;&amp; python -m world_spec.derived.prices --allow-synthetic &amp;&amp; python -m world_spec.viz.build_dashboard</code>. NOT manually edited.</p>

<h2>1. ARR ($M, annualized run-rate)</h2>
<div id="arr" class="chart" style="height:380px;"></div>

<h2>2. Headcount (FTE)</h2>
<div id="fte" class="chart" style="height:320px;"></div>

<h2>3. Owned GPU fleet</h2>
<div id="gpus" class="chart" style="height:320px;"></div>

<h2>4. Valuation pre-IPO and market cap post-IPO (log scale, $M)</h2>
<div id="mcap" class="chart" style="height:420px;"></div>

<h2>5. Daily stock price post-IPO (EBLA, Q1 2030 → Q4 2033)</h2>
<p class="meta">Derived via time-shift of CRWV (CoreWeave) real-daily series — CRWV's Mar 2025 listing becomes Echoblast's Mar 2030 IPO; percent-change series rolled forward. Tail extended with deterministic ARR-anchored synthetic walk where CRWV data doesn't exist. $32 IPO price; 438M shares outstanding.</p>
<div id="stock" class="chart" style="height:420px;"></div>

<h2>6. Anchor table (read-only)</h2>
<div id="anchor-table"></div>

<script>
const finDates = {json.dumps(fin_dates)};
const arr = {json.dumps(arr)};
const fte = {json.dumps(fte)};
const gpus = {json.dumps(gpus)};
const events = {json.dumps(events)};
const notes = {json.dumps(notes)};
const valDates = {json.dumps(val_dates)};
const postMoney = {json.dumps(post_money)};
const stkDates = {json.dumps(stk_dates)};
const stkClose = {json.dumps(stk_close)};
const stkHigh = {json.dumps(stk_high)};
const stkLow = {json.dumps(stk_low)};
const stkMcap = {json.dumps(stk_mcap)};

const layoutBase = {{
  margin: {{l: 60, r: 30, t: 10, b: 40}},
  hovermode: 'x unified',
  xaxis: {{title: '', tickfont: {{size: 11}}}},
  yaxis: {{tickfont: {{size: 11}}}},
  plot_bgcolor: '#fafafa',
  paper_bgcolor: '#ffffff',
  showlegend: false,
}};

Plotly.newPlot('arr', [{{
  x: finDates,
  y: arr,
  mode: 'lines+markers',
  line: {{color: '#2b6cb0', width: 2}},
  marker: {{size: 6}},
  text: events.map((e, i) => e + (notes[i] ? ' — ' + notes[i] : '')),
  hovertemplate: '%{{x}}: $%{{y:.1f}}M ARR<br><i>%{{text}}</i><extra></extra>',
}}], {{...layoutBase, yaxis: {{...layoutBase.yaxis, title: 'ARR $M', tickformat: '.0f'}}}}, {{displayModeBar: false}});

Plotly.newPlot('fte', [{{
  x: finDates, y: fte, mode: 'lines+markers',
  line: {{color: '#9f7aea', width: 2}}, marker: {{size: 6}},
  hovertemplate: '%{{x}}: %{{y}} FTE<extra></extra>',
}}], {{...layoutBase, yaxis: {{...layoutBase.yaxis, title: 'FTE'}}}}, {{displayModeBar: false}});

Plotly.newPlot('gpus', [{{
  x: finDates, y: gpus, mode: 'lines+markers',
  line: {{color: '#d69e2e', width: 2}}, marker: {{size: 6}},
  hovertemplate: '%{{x}}: %{{y:,}} GPUs owned<extra></extra>',
}}], {{...layoutBase, yaxis: {{...layoutBase.yaxis, title: 'Owned GPUs', tickformat: ',d'}}}}, {{displayModeBar: false}});

Plotly.newPlot('mcap', [
  {{x: valDates, y: postMoney, mode: 'lines', name: 'Pre-IPO post-money ($M)', line: {{color: '#2c7a7b', width: 2, shape: 'hv'}}, hovertemplate: '%{{x}}: $%{{y:.0f}}M post-money<extra></extra>'}},
  {{x: stkDates, y: stkMcap, mode: 'lines', name: 'Post-IPO market cap ($M)', line: {{color: '#c53030', width: 1.5}}, hovertemplate: '%{{x}}: $%{{y:,.0f}}M mcap<extra></extra>'}},
], {{...layoutBase, yaxis: {{...layoutBase.yaxis, title: '$M (log)', type: 'log'}}, showlegend: true, legend: {{x: 0.01, y: 0.99}}}}, {{displayModeBar: false}});

Plotly.newPlot('stock', [
  {{x: stkDates, y: stkHigh, mode: 'lines', line: {{color: 'rgba(200,40,40,0.15)', width: 0}}, showlegend: false, hoverinfo: 'skip'}},
  {{x: stkDates, y: stkLow, mode: 'lines', fill: 'tonexty', fillcolor: 'rgba(200,40,40,0.12)', line: {{color: 'rgba(200,40,40,0.15)', width: 0}}, showlegend: false, hoverinfo: 'skip'}},
  {{x: stkDates, y: stkClose, mode: 'lines', name: 'EBLA close', line: {{color: '#c53030', width: 1.5}}, hovertemplate: '%{{x}}: $%{{y:.2f}}<extra></extra>'}},
], {{...layoutBase, yaxis: {{...layoutBase.yaxis, title: 'EBLA $/share'}}, shapes: [{{type: 'line', x0: '2030-03-18', x1: '2030-03-18', y0: 0, y1: 1, yref: 'paper', line: {{color: '#718096', width: 1, dash: 'dot'}}}}], annotations: [{{x: '2030-03-18', y: 1, yref: 'paper', yanchor: 'bottom', text: 'IPO', showarrow: false, font: {{size: 10, color: '#718096'}}}}]}}, {{displayModeBar: false}});

// Anchor table
let t = '<table><thead><tr><th>Date</th><th>Event</th><th>ARR $M</th><th>FTE</th><th>Owned GPUs</th><th>Funding total $M</th><th>Post-money $M</th><th class="event">Notes</th></tr></thead><tbody>';
for (let i = 0; i < finDates.length; i++) {{
  t += '<tr><td>' + finDates[i] + '</td><td>' + events[i] + '</td><td>' + arr[i].toLocaleString() + '</td><td>' + fte[i].toLocaleString() + '</td><td>' + gpus[i].toLocaleString() + '</td><td>' + (i < {len(fin)} ? {json.dumps([float(r['funding_total_m']) for r in fin])}[i].toLocaleString() : '') + '</td><td>' + (i < {len(fin)} ? {json.dumps([float(r['post_money_m']) for r in fin])}[i].toLocaleString() : '') + '</td><td class="event">' + (notes[i] || '') + '</td></tr>';
}}
t += '</tbody></table>';
document.getElementById('anchor-table').innerHTML = t;
</script>

</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_HTML}")
    print(f"open: start file:///{OUT_HTML.as_posix()}")


if __name__ == "__main__":
    main()
