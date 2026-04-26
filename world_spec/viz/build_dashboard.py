"""Build a self-contained HTML dashboard from the derived CSVs.

Uses Plotly JS via CDN — no Python plotly dep needed. Output is
world_spec/viz/dashboard.html, openable in any browser.

Sections:
    1. ARR over time (quarterly step + IPO marker)
    2. Headcount over time
    3. GPU fleet over time
    4. Valuation (pre-IPO) + market cap (post-IPO) on one chart
    5. Daily stock price post-IPO with funding-round markers
    6. EBLA vs SMH vs NVDA (normalized to 1.0 at IPO date) with macro
       event annotations — visual sanity check that EBLA tracks factor
       indices
    7. Anchor table
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

VIZ_DIR = Path(__file__).parent
FINANCIALS_CSV = VIZ_DIR / "financials.csv"
VALUATION_CSV = VIZ_DIR / "valuation.csv"
STOCK_CSV = VIZ_DIR / "stock_prices.csv"
REAL_DIR = VIZ_DIR / "real_indices"
FACTOR_FIT_JSON = VIZ_DIR / "factor_fit.json"
MACRO_CSV = VIZ_DIR / "macro_calendar.csv"
OUT_HTML = VIZ_DIR / "dashboard.html"

ECHOBLAST_IPO_DATE_STR = "2030-03-18"


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_real_smh_nvda_projected() -> dict:
    """Build SMH + NVDA series spanning the same window as EBLA (IPO →
    Dec 2033) by combining real data with projected data the same way
    prices.py does. Rather than re-running the projection here, we
    reconstruct SMH/NVDA closes by reading the real CSVs and then
    overlaying the projected segment from a fresh call to
    world_spec.derived.prices.project_factors.
    """
    from world_spec.derived import prices as P
    import pandas as pd

    reals = P.load_real_indices()
    smh = reals["SMH"]["Close"].sort_index()
    nvda = reals["NVDA"]["Close"].sort_index()
    factor_df, _ = P.project_factors(smh, nvda, P.PROJECTION_START, P.PROJECTION_END)
    # Keep only rows from EBLA IPO onward
    factor_df["date_d"] = pd.to_datetime(factor_df["date"]).dt.date
    ebla_ipo = dt.date.fromisoformat(ECHOBLAST_IPO_DATE_STR)
    sub = factor_df[factor_df["date_d"] >= ebla_ipo].reset_index(drop=True)
    return {
        "dates": [d.isoformat() for d in sub["date_d"].tolist()],
        "smh_close": [float(x) for x in sub["smh_close"].tolist()],
        "nvda_close": [float(x) for x in sub["nvda_close"].tolist()],
    }


def main() -> None:
    fin = load_csv(FINANCIALS_CSV)
    val = load_csv(VALUATION_CSV)
    stk = load_csv(STOCK_CSV)
    macro = load_csv(MACRO_CSV) if MACRO_CSV.exists() else []
    factor_fit = json.loads(FACTOR_FIT_JSON.read_text(encoding="utf-8")) if FACTOR_FIT_JSON.exists() else {}

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

    # GPU mix by generation (from monthly_financials)
    monthly = load_csv(VIZ_DIR / "monthly_financials.csv")
    gpu_mix_dates = [r["month_end"] for r in monthly]
    gpu_h100 = [int(r.get("fleet_h100", 0)) for r in monthly]
    gpu_h200 = [int(r.get("fleet_h200", 0)) for r in monthly]
    gpu_blackwell = [int(r.get("fleet_blackwell", 0)) for r in monthly]
    gpu_marketplace = [int(r.get("fleet_marketplace", 0)) for r in monthly]

    # P/E ratio: stock_mcap / trailing-12-month net income (use ARR × 0.20 net margin as proxy when net income unavailable)
    # Build a date → annualized net income map from monthly
    monthly_by_date = {r["month_end"]: r for r in monthly}
    pe_ratios = []
    for stk_date, mcap in zip(stk_dates, stk_mcap):
        # Find monthly row at or before stk_date
        best = None
        for d, row in monthly_by_date.items():
            if d <= stk_date:
                best = row
        if best:
            # Annualize current month's net income (×12)
            try:
                ni = float(best.get("net_income_m", 0)) * 12
                pe = mcap / ni if ni > 0 else None
            except (ValueError, ZeroDivisionError):
                pe = None
        else:
            pe = None
        pe_ratios.append(pe)

    # Normalized comparison — EBLA / SMH / NVDA all indexed to 1.0 at IPO date
    try:
        factor_series = _load_real_smh_nvda_projected()
        ebla_norm_dates = stk_dates
        ebla_base = stk_close[0]
        ebla_norm = [c / ebla_base for c in stk_close]
        # Align SMH/NVDA to same dates
        smh_by_date = dict(zip(factor_series["dates"], factor_series["smh_close"]))
        nvda_by_date = dict(zip(factor_series["dates"], factor_series["nvda_close"]))
        smh_aligned = [smh_by_date.get(d) for d in ebla_norm_dates]
        nvda_aligned = [nvda_by_date.get(d) for d in ebla_norm_dates]
        # Fill any gaps with forward-fill
        last_smh = None
        for i in range(len(smh_aligned)):
            if smh_aligned[i] is None:
                smh_aligned[i] = last_smh
            else:
                last_smh = smh_aligned[i]
        last_nvda = None
        for i in range(len(nvda_aligned)):
            if nvda_aligned[i] is None:
                nvda_aligned[i] = last_nvda
            else:
                last_nvda = nvda_aligned[i]
        smh_base = smh_aligned[0] if smh_aligned[0] else 1.0
        nvda_base = nvda_aligned[0] if nvda_aligned[0] else 1.0
        smh_norm = [s / smh_base if s else None for s in smh_aligned]
        nvda_norm = [n / nvda_base if n else None for n in nvda_aligned]
    except Exception as e:
        print(f"WARNING: could not build normalized comparison: {e!r}")
        ebla_norm_dates = stk_dates
        ebla_norm = []
        smh_norm = []
        nvda_norm = []

    # Macro event annotations: only those with dates within the stock window
    stk_start = stk_dates[0] if stk_dates else ECHOBLAST_IPO_DATE_STR
    stk_end = stk_dates[-1] if stk_dates else "2033-12-31"
    macro_in_window = [
        m for m in macro
        if stk_start <= m["date"] <= stk_end
    ]

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
  .fitbox {{ background: #f7fafc; border: 1px solid #e2e8f0; padding: 10px 14px; margin: 10px 0; font-size: 13px; font-family: monospace; white-space: pre-wrap; }}
</style>
</head>
<body>

<h1>Echoblast — derived financials dashboard</h1>
<p class="meta">Source of truth: <code>docs/spec.md §3.1</code>. Regenerate: <code>python -m world_spec.derived.prices &amp;&amp; python -m world_spec.viz.build_dashboard</code>. NOT manually edited.</p>

<h2>1. ARR ($M, annualized run-rate)</h2>
<div id="arr" class="chart" style="height:380px;"></div>

<h2>2. Headcount (FTE)</h2>
<p class="meta">Hiring-plan annotations:
  <b>Mar 2025</b> 4 founders + first eng hire (YC W25 batch) ·
  <b>Late 2025</b> 14 FTE post-seed (mostly engineering) ·
  <b>2026</b> first SRE (Jacob Evans), first sales (Carmen Russell), Diego Baker as recruiting head Jan 2027 ·
  <b>2027 hiring plan</b> "scale eng to 50 + first PMs + first BD/datacenter-leases hire (Roman Munoz)" ·
  <b>2028 (post-Series B)</b> "scale to 90 — first dedicated training-platform team, first GTM segments split" ·
  <b>2029 (Series C/D)</b> "scale to 230 — pre-IPO hires: CISO Matthew Stewart, IR head Youssef Wilson queued for IPO" ·
  <b>2030 (post-IPO)</b> 500 FTE — Austin support center opens, Tier-1 customer-success org built ·
  <b>2031-32</b> "30% YoY hiring; multi-region ops org for EU expansion" ·
  <b>2033 (recession)</b> Q1 hiring freeze, Q2 4% RIF (~100 cut, fact F-0202), Q3-Q4 modest re-hire to 1,850.
  Deltas vs plan are tracked in board minutes; FY2034 plan: ~22% revenue growth at flat-to-modest headcount growth.</p>
<div id="fte" class="chart" style="height:320px;"></div>

<h2>3. Owned GPU fleet</h2>
<p class="meta">Fleet composition by year-end:
  <b>2026</b> 300× H100 SXM (NVIDIA via Supermicro PO-001/002) at EBL-VA1 Sterling VA (Equinix colo) ·
  <b>2027</b> 1,200× H100 across VA1 ·
  <b>2028</b> 3,200 mixed (2,800 H100 + 400 H200, NVIDIA direct PO-004 thru PO-007) at VA1 + VA2 Ashburn (Equinix) + OR1 Hillsboro (QTS) ·
  <b>2029</b> 28,000 (3K H100 + 10K H200 + 15K Blackwell B200, PO-008 thru PO-012) — first Blackwell ramp at NV1 Las Vegas (Switch) ·
  <b>2030</b> 75,000 (mostly Blackwell GB200, NVLink-connected pods, PO-013/014) post-IPO ·
  <b>2031</b> 120,000 — H100 retiring, EU1 Frankfurt (Interxion) opens ·
  <b>2032</b> 180,000 — first Rubin R100 orders (PO-017/018) at NV2 ·
  <b>2033</b> 240,000 — Rubin R100/R200 dominant, H100 fully retired, EU2 Amsterdam opens ·
  AMD MI300X/MI325X/MI350X/MI400X (~1,500 GPUs): added 2028+ for AMD-optimized workloads at OR1 + NV1.
  All POs in <code>derived/gpu_purchase_orders.csv</code>. DC details in <code>derived/dc_buildout.csv</code>.</p>
<div id="gpus" class="chart" style="height:320px;"></div>

<h2>3a. GPU fleet by generation (stacked area)</h2>
<div id="gpu_mix" class="chart" style="height:320px;"></div>

<h2>4. Valuation pre-IPO and market cap post-IPO (log scale, $M)</h2>
<div id="mcap" class="chart" style="height:420px;"></div>

<h2>5. Daily stock price post-IPO (EBLA, Q1 2030 → Q4 2033)</h2>
<p class="meta">Derived from a two-factor model:
  <code>r_EBLA = alpha + beta_SMH·r_SMH + beta_NVDA·r_NVDA + eps</code>.
  Factor loadings fit by OLS on real CRWV + NBIS daily log-returns vs SMH + NVDA (2024-2026). Where real
  SMH/NVDA data exists (through ~April 2026), those are the real factor inputs; beyond that, SMH/NVDA are
  projected by correlated GBM (fitted 2024-2026 drift/vol) with a deterministic macro-event calendar
  overlaid. EBLA's idiosyncratic noise is a fixed-seed (2030) draw. $32 IPO price; 438M shares out;
  target end-2033 close ≈ $150 (4.8× IPO).
  <b>Hover for P/E ratio</b> at each date (TTM-proxy: mcap / annualized current month net income).
  Sell-side analyst notes typically reference these multiples: at IPO ~10× ARR / N/M P/E (early losses);
  by Q4 2033 ~22× ARR and ~25-40× forward P/E typical for a 22% growth neocloud.</p>
<div id="stock" class="chart" style="height:460px;"></div>

<h2>5a. P/E ratio over time (TTM-proxy)</h2>
<p class="meta">Annualized monthly net income × 12 vs. market cap. Use as an approximate metric for analyst-doc
  references. Capped at 100× display ceiling (early post-IPO P/E is very high or undefined when net income near zero).</p>
<div id="pe_chart" class="chart" style="height:280px;"></div>

<h2>6. EBLA vs SMH vs NVDA (normalized to 1.0 at IPO)</h2>
<p class="meta">Sanity check — EBLA (red) should move with its factor indices (SMH blue, NVDA green). Vertical dashed lines mark macro events from <code>macro_calendar.csv</code>.</p>
<div id="compare" class="chart" style="height:540px;"></div>

<h2>7. Fitted factor parameters</h2>
<div class="fitbox">{json.dumps(factor_fit, indent=2) if factor_fit else '(no factor_fit.json found)'}</div>

<h2>8. Anchor table (read-only)</h2>
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
const peRatios = {json.dumps(pe_ratios)};
const gpuMixDates = {json.dumps(gpu_mix_dates)};
const gpuH100 = {json.dumps(gpu_h100)};
const gpuH200 = {json.dumps(gpu_h200)};
const gpuBlackwell = {json.dumps(gpu_blackwell)};
const gpuMarketplace = {json.dumps(gpu_marketplace)};
const eblaNormDates = {json.dumps(ebla_norm_dates)};
const eblaNorm = {json.dumps(ebla_norm)};
const smhNorm = {json.dumps(smh_norm)};
const nvdaNorm = {json.dumps(nvda_norm)};
const macroEvents = {json.dumps(macro_in_window)};

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

// GPU mix by generation (stacked area)
Plotly.newPlot('gpu_mix', [
  {{x: gpuMixDates, y: gpuH100, mode: 'lines', stackgroup: 'mix', name: 'H100', line: {{width: 0.5, color: '#3182ce'}}, fillcolor: 'rgba(49,130,206,0.6)', hovertemplate: '%{{x}}: %{{y:,}} H100<extra></extra>'}},
  {{x: gpuMixDates, y: gpuH200, mode: 'lines', stackgroup: 'mix', name: 'H200', line: {{width: 0.5, color: '#38a169'}}, fillcolor: 'rgba(56,161,105,0.6)', hovertemplate: '%{{x}}: %{{y:,}} H200<extra></extra>'}},
  {{x: gpuMixDates, y: gpuBlackwell, mode: 'lines', stackgroup: 'mix', name: 'Blackwell + Rubin', line: {{width: 0.5, color: '#d69e2e'}}, fillcolor: 'rgba(214,158,46,0.6)', hovertemplate: '%{{x}}: %{{y:,}} Blackwell/Rubin<extra></extra>'}},
  {{x: gpuMixDates, y: gpuMarketplace, mode: 'lines', stackgroup: 'mix', name: 'Marketplace 3P', line: {{width: 0.5, color: '#9f7aea'}}, fillcolor: 'rgba(159,122,234,0.4)', hovertemplate: '%{{x}}: %{{y:,}} marketplace<extra></extra>'}},
], {{...layoutBase, yaxis: {{...layoutBase.yaxis, title: 'GPUs', tickformat: ',d'}}, showlegend: true, legend: {{x: 0.01, y: 0.99}}}}, {{displayModeBar: false}});

Plotly.newPlot('mcap', [
  {{x: valDates, y: postMoney, mode: 'lines', name: 'Pre-IPO post-money ($M)', line: {{color: '#2c7a7b', width: 2, shape: 'hv'}}, hovertemplate: '%{{x}}: $%{{y:.0f}}M post-money<extra></extra>'}},
  {{x: stkDates, y: stkMcap, mode: 'lines', name: 'Post-IPO market cap ($M)', line: {{color: '#c53030', width: 1.5}}, hovertemplate: '%{{x}}: $%{{y:,.0f}}M mcap<extra></extra>'}},
], {{...layoutBase, yaxis: {{...layoutBase.yaxis, title: '$M (log)', type: 'log'}}, showlegend: true, legend: {{x: 0.01, y: 0.99}}}}, {{displayModeBar: false}});

// Build hover text with P/E for each day
const stockHoverText = stkDates.map((d, i) => {{
  const close = stkClose[i].toFixed(2);
  const mcap = (stkMcap[i] / 1000).toFixed(2);
  const pe = peRatios[i] != null ? peRatios[i].toFixed(1) : 'N/A';
  return `${{d}}<br>Close: $${{close}}<br>Mcap: $${{mcap}}B<br>P/E (TTM-proxy): ${{pe}}×`;
}});

Plotly.newPlot('stock', [
  {{x: stkDates, y: stkHigh, mode: 'lines', line: {{color: 'rgba(200,40,40,0.15)', width: 0}}, showlegend: false, hoverinfo: 'skip'}},
  {{x: stkDates, y: stkLow, mode: 'lines', fill: 'tonexty', fillcolor: 'rgba(200,40,40,0.12)', line: {{color: 'rgba(200,40,40,0.15)', width: 0}}, showlegend: false, hoverinfo: 'skip'}},
  {{x: stkDates, y: stkClose, mode: 'lines', name: 'EBLA close', line: {{color: '#c53030', width: 1.5}}, text: stockHoverText, hovertemplate: '%{{text}}<extra></extra>'}},
], {{...layoutBase, yaxis: {{...layoutBase.yaxis, title: 'EBLA $/share'}}, shapes: [{{type: 'line', x0: '2030-03-18', x1: '2030-03-18', y0: 0, y1: 1, yref: 'paper', line: {{color: '#718096', width: 1, dash: 'dot'}}}}], annotations: [{{x: '2030-03-18', y: 1, yref: 'paper', yanchor: 'bottom', text: 'IPO', showarrow: false, font: {{size: 10, color: '#718096'}}}}]}}, {{displayModeBar: false}});

// P/E ratio chart
Plotly.newPlot('pe_chart', [{{
  x: stkDates, y: peRatios, mode: 'lines', line: {{color: '#805ad5', width: 1.5}},
  hovertemplate: '%{{x}}: %{{y:.1f}}× P/E<extra></extra>',
}}], {{...layoutBase, yaxis: {{...layoutBase.yaxis, title: 'P/E (TTM-proxy)', range: [0, 100]}}}}, {{displayModeBar: false}});

// Normalized comparison chart with macro event annotations
const macroShapes = macroEvents.map(m => ({{
  type: 'line', x0: m.date, x1: m.date, y0: 0, y1: 1, yref: 'paper',
  line: {{color: parseFloat(m.signed_impact_pct) < 0 ? 'rgba(197,48,48,0.35)' : 'rgba(47,133,90,0.35)', width: 1, dash: 'dash'}},
}}));
const macroAnnotations = macroEvents.map((m, i) => ({{
  x: m.date, y: 1 - 0.04 * (i % 6), yref: 'paper',
  text: (parseFloat(m.signed_impact_pct) >= 0 ? '+' : '') + parseFloat(m.signed_impact_pct).toFixed(0) + '% ' + m.target + ': ' + m.description.slice(0, 38),
  showarrow: false,
  font: {{size: 9, color: '#4a5568'}},
  bgcolor: 'rgba(255,255,255,0.75)',
  bordercolor: '#cbd5e0', borderwidth: 0.5, borderpad: 2,
  xanchor: 'left',
}}));

Plotly.newPlot('compare', [
  {{x: eblaNormDates, y: smhNorm, mode: 'lines', name: 'SMH (normalized)', line: {{color: '#3182ce', width: 1.2}}, hovertemplate: '%{{x}}: %{{y:.2f}}x<extra>SMH</extra>'}},
  {{x: eblaNormDates, y: nvdaNorm, mode: 'lines', name: 'NVDA (normalized)', line: {{color: '#38a169', width: 1.2}}, hovertemplate: '%{{x}}: %{{y:.2f}}x<extra>NVDA</extra>'}},
  {{x: eblaNormDates, y: eblaNorm, mode: 'lines', name: 'EBLA (normalized)', line: {{color: '#c53030', width: 2}}, hovertemplate: '%{{x}}: %{{y:.2f}}x<extra>EBLA</extra>'}},
], {{...layoutBase, yaxis: {{...layoutBase.yaxis, title: 'multiple of IPO-day close'}}, showlegend: true, legend: {{x: 0.01, y: 0.99}}, shapes: macroShapes, annotations: macroAnnotations}}, {{displayModeBar: false}});

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
