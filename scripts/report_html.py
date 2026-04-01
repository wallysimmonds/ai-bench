#!/usr/bin/env python3
"""
report_html.py — Generate HTML benchmark report from results JSON files

Usage:
    python scripts/report_html.py --results results/
    python scripts/report_html.py --results results/ --output reports/benchmark.html
"""

import argparse
import json
import glob
from datetime import datetime
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_results(results_dir):
    records = []
    for f in glob.glob(str(Path(results_dir) / "*.json")):
        with open(f) as fp:
            data = json.load(fp)
            if isinstance(data, list):
                records.extend(data)
    return records


def build_report(records):
    nodes = sorted(set(r["node"] for r in records))
    models = sorted(set(r["model"] for r in records))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Aggregate best tok/s per model/node
    perf = defaultdict(dict)
    model_best = defaultdict(lambda: {"tps": 0, "node": "-", "ttft_vals": [], "tps_vals": []})
    for r in records:
        if r.get("error") or not r.get("tokens_per_sec"):
            continue
        tps = r["tokens_per_sec"]
        model = r["model"]
        node = r["node"]
        existing = perf[model].get(node, 0)
        if tps > existing:
            perf[model][node] = tps
        model_best[model]["tps_vals"].append(tps)
        if tps > model_best[model]["tps"]:
            model_best[model]["tps"] = tps
            model_best[model]["best_node"] = node
        if r.get("ttft_ms"):
            model_best[model]["ttft_vals"].append(r["ttft_ms"])

    max_tps = max((v for m in perf.values() for v in m.values()), default=1)

    def tps_bar(tps, max_v=None):
        if not tps or tps == "—":
            return '<span style="color:#666">—</span>'
        mv = max_v or max_tps
        pct = min(100, int(tps / mv * 100))
        colour = "#06d6a0" if pct > 66 else "#ffd166" if pct > 33 else "#ef476f"
        return f'''<div style="display:flex;align-items:center;gap:8px">
            <div style="flex:1;background:#1a1a2e;border-radius:4px;height:16px;overflow:hidden">
                <div style="width:{pct}%;height:100%;background:{colour};border-radius:4px"></div>
            </div>
            <span style="font-weight:600;min-width:52px;color:{colour}">{tps}</span>
        </div>'''

    # Matrix table
    matrix_rows = ""
    for model in models:
        cells = ""
        for node in nodes:
            tps = perf[model].get(node)
            cells += f'<td style="padding:10px 14px">{tps_bar(tps) if tps else tps_bar("—")}</td>'
        matrix_rows += f'<tr><td style="padding:10px 14px;font-weight:600;color:#a0aec0;font-size:13px">{model}</td>{cells}</tr>'

    # Summary table
    summary_rows = ""
    for model, stats in sorted(model_best.items(), key=lambda x: -x[1]["tps"]):
        avg_tps = round(sum(stats["tps_vals"]) / len(stats["tps_vals"]), 1) if stats["tps_vals"] else "—"
        avg_ttft = round(sum(stats["ttft_vals"]) / len(stats["ttft_vals"])) if stats["ttft_vals"] else "—"
        best = stats["tps"]
        summary_rows += f"""<tr>
            <td style="padding:10px 14px;color:#e2e8f0;font-weight:600">{model}</td>
            <td style="padding:10px 14px;color:#a0aec0">{stats.get('best_node','—')}</td>
            <td style="padding:10px 14px">{tps_bar(best)}</td>
            <td style="padding:10px 14px;color:#a0aec0;text-align:center">{avg_tps}</td>
            <td style="padding:10px 14px;color:#a0aec0;text-align:center">{avg_ttft}</td>
        </tr>"""

    # Raw results table
    raw_rows = ""
    for r in sorted(records, key=lambda x: x.get("timestamp",""), reverse=True):
        err = r.get("error")
        tps = r.get("tokens_per_sec")
        ttft = r.get("ttft_ms")
        row_colour = "#2d1515" if err else ""
        raw_rows += f"""<tr style="background:{row_colour}">
            <td style="padding:8px 12px;color:#718096;font-size:12px">{r.get('timestamp','')[:19]}</td>
            <td style="padding:8px 12px;color:#90cdf4">{r.get('node','')}</td>
            <td style="padding:8px 12px;color:#e2e8f0;font-size:12px">{r.get('model','')}</td>
            <td style="padding:8px 12px;color:#a0aec0;font-size:12px">{r.get('test_name','')}</td>
            <td style="padding:8px 12px;text-align:center">{'<span style="color:#06d6a0;font-weight:600">'+str(tps)+'</span>' if tps else '—'}</td>
            <td style="padding:8px 12px;text-align:center;color:#a0aec0">{ttft if ttft else '—'}</td>
            <td style="padding:8px 12px;color:#ef476f;font-size:11px">{err or ''}</td>
        </tr>"""

    node_headers = "".join(f'<th style="padding:12px 14px;color:#90cdf4;font-weight:600">{n}</th>' for n in nodes)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Fleet Benchmark Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
             padding: 40px 48px; border-bottom: 1px solid #0f3460; }}
  .header h1 {{ font-size: 28px; font-weight: 700; color: #fff; letter-spacing: -0.5px; }}
  .header p {{ color: #718096; margin-top: 6px; font-size: 14px; }}
  .stats-row {{ display: flex; gap: 16px; margin-top: 24px; flex-wrap: wrap; }}
  .stat-card {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
                border-radius: 10px; padding: 14px 20px; min-width: 140px; }}
  .stat-card .val {{ font-size: 26px; font-weight: 700; color: #00b4d8; }}
  .stat-card .lbl {{ font-size: 12px; color: #718096; margin-top: 2px; }}
  .content {{ padding: 32px 48px; max-width: 1400px; }}
  .section {{ margin-bottom: 40px; }}
  .section-title {{ font-size: 16px; font-weight: 700; color: #90cdf4;
                    margin-bottom: 16px; padding-bottom: 8px;
                    border-bottom: 1px solid #1a1a2e; letter-spacing: 0.5px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  thead {{ background: #1a1a2e; }}
  thead th {{ padding: 12px 14px; text-align: left; font-size: 12px;
              font-weight: 600; color: #718096; letter-spacing: 0.5px; text-transform: uppercase; }}
  tbody tr {{ border-bottom: 1px solid #1a1a2e; transition: background 0.15s; }}
  tbody tr:hover {{ background: rgba(255,255,255,0.03); }}
  .card {{ background: #161b22; border: 1px solid #1a1a2e; border-radius: 12px;
           overflow: hidden; }}
  .pill {{ display: inline-block; padding: 3px 10px; border-radius: 20px;
           font-size: 11px; font-weight: 600; }}
  .pill-green {{ background: rgba(6,214,160,0.15); color: #06d6a0; }}
  .pill-blue  {{ background: rgba(0,180,216,0.15); color: #00b4d8; }}
</style>
</head>
<body>
<div class="header">
  <h1>⚡ AI Fleet Benchmark Report</h1>
  <p>Local inference fleet performance — Sovereign AI Lab</p>
  <div class="stats-row">
    <div class="stat-card"><div class="val">{len(nodes)}</div><div class="lbl">Nodes</div></div>
    <div class="stat-card"><div class="val">{len(models)}</div><div class="lbl">Models</div></div>
    <div class="stat-card"><div class="val">{len(records)}</div><div class="lbl">Benchmark Runs</div></div>
    <div class="stat-card"><div class="val">{max_tps}</div><div class="lbl">Peak tok/s</div></div>
    <div class="stat-card"><div class="val" style="font-size:16px;padding-top:4px">{ts}</div><div class="lbl">Generated</div></div>
  </div>
</div>

<div class="content">

  <div class="section">
    <div class="section-title">PERFORMANCE SUMMARY</div>
    <div class="card">
      <table>
        <thead><tr>
          <th>Model</th><th>Best Node</th><th style="min-width:200px">Peak tok/s</th>
          <th>Avg tok/s</th><th>Avg TTFT (ms)</th>
        </tr></thead>
        <tbody>{summary_rows or '<tr><td colspan="5" style="padding:20px;text-align:center;color:#718096">No results yet — run benchmark.py first</td></tr>'}</tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <div class="section-title">NODE × MODEL MATRIX (tokens/sec)</div>
    <div class="card">
      <table>
        <thead><tr><th>Model</th>{node_headers}</tr></thead>
        <tbody>{matrix_rows or '<tr><td colspan="10" style="padding:20px;text-align:center;color:#718096">No data</td></tr>'}</tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <div class="section-title">RAW RESULTS</div>
    <div class="card">
      <table>
        <thead><tr>
          <th>Timestamp</th><th>Node</th><th>Model</th><th>Test</th>
          <th>tok/s</th><th>TTFT (ms)</th><th>Error</th>
        </tr></thead>
        <tbody>{raw_rows or '<tr><td colspan="7" style="padding:20px;text-align:center;color:#718096">No results yet</td></tr>'}</tbody>
      </table>
    </div>
  </div>

</div>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate HTML benchmark report")
    parser.add_argument("--results", default="results/", help="Results directory")
    parser.add_argument("--output", help="Output HTML file path")
    args = parser.parse_args()

    records = load_results(args.results)
    print(f"Loaded {len(records)} benchmark records")

    html = build_report(records)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or str(REPORTS_DIR / f"benchmark_{ts}.html")
    with open(output, "w") as f:
        f.write(html)
    print(f"HTML report saved: {output}")


if __name__ == "__main__":
    main()
