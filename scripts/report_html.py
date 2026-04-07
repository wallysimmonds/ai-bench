#!/usr/bin/env python3
"""
report_html.py — Generate HTML benchmark report from results JSON files

Handles two result types:
  - benchmark.py (Ollama): tg_tokens_per_sec, pp_tokens_per_sec, ttft_ms
  - llama_bench.sh (llama-bench): avg_ts, n_gen==0 → pp512, n_gen>0 → tg128

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


# ── Model metadata ────────────────────────────────────────────────────────────

def load_model_meta():
    """Return dict: ollama_tag → {arch, active_params} from models.yaml."""
    import yaml
    meta = {}
    try:
        with open(ROOT / "config" / "models.yaml") as f:
            cfg = yaml.safe_load(f)
        for tier in cfg.get("models", {}).values():
            for m in tier:
                tag = m.get("ollama_tag") or m.get("name")
                if tag:
                    meta[tag] = {
                        "arch":          m.get("arch", "dense"),
                        "active_params": m.get("active_params"),
                    }
    except Exception:
        pass
    return meta


def model_label(tag, meta):
    """Return display string e.g. 'qwen3.5:122b — MoE (10B active)'."""
    info = meta.get(tag, {})
    arch = info.get("arch", "dense")
    if arch == "moe":
        active = info.get("active_params")
        suffix = f"MoE ({active} active)" if active else "MoE"
        return tag, f'<span style="color:#ffd166;font-size:11px;font-weight:400"> — {suffix}</span>'
    return tag, '<span style="color:#4a5568;font-size:11px;font-weight:400"> — dense</span>'


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(results_dir):
    ollama, llama = [], []
    for f in sorted(glob.glob(str(Path(results_dir) / "*.json"))):
        with open(f) as fp:
            try:
                data = json.load(fp)
            except json.JSONDecodeError:
                continue
        if not isinstance(data, list):
            continue
        for r in data:
            if r.get("bench_tool") == "llama-bench":
                llama.append(r)
            elif "test_id" in r:
                ollama.append(r)
    return ollama, llama


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_ollama(records):
    tg_perf  = defaultdict(dict)
    pp_perf  = defaultdict(dict)
    ttft_all = defaultdict(list)
    for r in records:
        if r.get("error"):
            continue
        model, node = r["model"], r["node"]
        tg, pp, ttft = r.get("tg_tokens_per_sec"), r.get("pp_tokens_per_sec"), r.get("ttft_ms")
        if tg:
            tg_perf[model][node] = max(tg_perf[model].get(node, 0), tg)
        if pp:
            pp_perf[model][node] = max(pp_perf[model].get(node, 0), pp)
        if ttft:
            ttft_all[model].append(ttft)
    return tg_perf, pp_perf, ttft_all


def aggregate_llama(records):
    pp_perf = defaultdict(dict)
    tg_perf = defaultdict(dict)
    for r in records:
        model = r.get("model_tag", r.get("model", "unknown"))
        node  = r.get("node", "unknown")
        ts    = r.get("avg_ts")
        if ts is None:
            continue
        if r.get("n_gen", 1) == 0:
            pp_perf[model][node] = round(ts, 1)
        else:
            tg_perf[model][node] = round(ts, 1)
    return tg_perf, pp_perf


# ── HTML helpers ──────────────────────────────────────────────────────────────

def fmt(val, unit=""):
    return "—" if val is None else f"{val}{unit}"


def bar(val, max_v, colour=None):
    if val is None:
        return '<span style="color:#4a5568">—</span>'
    pct = min(100, int(float(val) / max_v * 100)) if max_v else 0
    c = colour or ("#06d6a0" if pct > 66 else "#ffd166" if pct > 33 else "#ef476f")
    return (f'<div style="display:flex;align-items:center;gap:8px">'
            f'<div style="flex:1;background:#1a1a2e;border-radius:4px;height:14px;overflow:hidden">'
            f'<div style="width:{pct}%;height:100%;background:{c};border-radius:4px"></div></div>'
            f'<span style="font-weight:600;min-width:60px;color:{c};font-size:13px">{val}</span></div>')


def empty(cols):
    return (f'<tr><td colspan="{cols}" style="padding:20px;text-align:center;'
            f'color:#4a5568;font-style:italic">No data</td></tr>')


def card(html):
    return f'<div class="card">{html}</div>'


def table(thead, tbody):
    return f'<table><thead>{thead}</thead><tbody>{tbody}</tbody></table>'


def th(*cols):
    return "<tr>" + "".join(
        f'<th style="padding:12px 14px">{c}</th>' for c in cols
    ) + "</tr>"


def section(title, content, anchor="", subtitle=""):
    sub = f'<span class="section-sub">{subtitle}</span>' if subtitle else ""
    a   = f'<a name="{anchor}"></a>' if anchor else ""
    return f'''{a}
  <div class="section">
    <div class="section-title">{title}{sub}</div>
    {content}
  </div>'''


def collapsible(label, content):
    return f'''<details class="drill">
  <summary>{label}</summary>
  <div class="drill-content">{content}</div>
</details>'''


# ── Summary tables ────────────────────────────────────────────────────────────

def ollama_summary_table(tg_perf, pp_perf, ttft_all, meta):
    all_tg = [v for m in tg_perf.values() for v in m.values()]
    all_pp = [v for m in pp_perf.values() for v in m.values()]
    max_tg = max(all_tg, default=1)
    max_pp = max(all_pp, default=1)

    models = sorted(set(list(tg_perf.keys()) + list(pp_perf.keys())),
                    key=lambda m: -max(tg_perf[m].values(), default=0))
    rows = ""
    for model in models:
        best_tg   = max(tg_perf[model].values(), default=None)
        best_pp   = max(pp_perf[model].values(), default=None)
        best_node = (max(tg_perf[model], key=tg_perf[model].get)
                     if tg_perf[model] else "—")
        avg_ttft  = (round(sum(ttft_all[model]) / len(ttft_all[model]))
                     if ttft_all.get(model) else None)
        _, badge = model_label(model, meta)
        rows += (f'<tr>'
                 f'<td class="td-model">{model}{badge}</td>'
                 f'<td class="td-dim">{best_node}</td>'
                 f'<td class="td-bar">{bar(best_tg, max_tg)}</td>'
                 f'<td class="td-bar">{bar(best_pp, max_pp, "#00b4d8")}</td>'
                 f'<td class="td-num">{fmt(avg_ttft, "ms")}</td>'
                 f'</tr>')

    thead = th("Model", "Best Node",
               'TG tok/s <span class="col-sub">(generation)</span>',
               'PP tok/s <span class="col-sub">(prefill)</span>',
               "Avg TTFT")
    return card(table(thead, rows or empty(5))), max_tg


def llama_summary_table(tg_perf, pp_perf, meta):
    all_tg = [v for m in tg_perf.values() for v in m.values()]
    all_pp = [v for m in pp_perf.values() for v in m.values()]
    max_tg = max(all_tg, default=1)
    max_pp = max(all_pp, default=1)

    models = sorted(set(list(tg_perf.keys()) + list(pp_perf.keys())),
                    key=lambda m: -max(tg_perf[m].values(), default=0))
    rows = ""
    for model in models:
        best_tg   = max(tg_perf[model].values(), default=None)
        best_pp   = max(pp_perf[model].values(), default=None)
        best_node = (max(tg_perf[model], key=tg_perf[model].get)
                     if tg_perf[model] else "—")
        _, badge = model_label(model, meta)
        rows += (f'<tr>'
                 f'<td class="td-model">{model}{badge}</td>'
                 f'<td class="td-dim">{best_node}</td>'
                 f'<td class="td-bar">{bar(best_tg, max_tg)}</td>'
                 f'<td class="td-bar">{bar(best_pp, max_pp, "#00b4d8")}</td>'
                 f'<td class="td-num" style="color:#4a5568">pp512 / tg128</td>'
                 f'</tr>')

    thead = th("Model", "Best Node",
               'TG tok/s <span class="col-sub">(tg128)</span>',
               'PP tok/s <span class="col-sub">(pp512)</span>',
               "Test")
    return card(table(thead, rows or empty(5))), max_tg


# ── Detail tables (for collapsible drill-down) ────────────────────────────────

def ollama_matrix(tg_perf):
    all_tg  = [v for m in tg_perf.values() for v in m.values()]
    max_tg  = max(all_tg, default=1)
    models  = sorted(tg_perf.keys())
    nodes   = sorted(set(n for m in tg_perf.values() for n in m.keys()))
    n_heads = "".join(f'<th style="padding:12px 14px">{n}</th>' for n in nodes)
    rows    = ""
    for model in models:
        cells = "".join(
            f'<td class="td-bar">{bar(tg_perf[model].get(n), max_tg)}</td>'
            for n in nodes
        )
        rows += f'<tr><td class="td-dim" style="white-space:nowrap">{model}</td>{cells}</tr>'
    thead = f'<tr><th style="padding:12px 14px">Model</th>{n_heads}</tr>'
    return card(table(thead, rows or empty(len(nodes) + 1)))


def ollama_raw(records):
    rows = ""
    for r in sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True):
        err  = r.get("error")
        tg   = r.get("tg_tokens_per_sec")
        pp   = r.get("pp_tokens_per_sec")
        ttft = r.get("ttft_ms")
        bg   = 'style="background:#2d1515"' if err else ""
        rows += (f'<tr {bg}>'
                 f'<td class="td-ts">{r.get("timestamp","")[:19]}</td>'
                 f'<td class="td-dim">{r.get("node","")}</td>'
                 f'<td class="td-dim" style="white-space:nowrap">{r.get("model","")}</td>'
                 f'<td class="td-dim">{r.get("test_name","")}</td>'
                 f'<td class="td-num" style="color:#06d6a0">{fmt(tg)}</td>'
                 f'<td class="td-num" style="color:#00b4d8">{fmt(pp)}</td>'
                 f'<td class="td-num">{fmt(ttft, "ms")}</td>'
                 f'<td class="td-err">{err or ""}</td>'
                 f'</tr>')
    thead = th("Timestamp", "Node", "Model", "Test",
               "TG tok/s", "PP tok/s", "TTFT", "Error")
    return card(table(thead, rows or empty(8)))


def llama_matrix(tg_perf):
    all_tg = [v for m in tg_perf.values() for v in m.values()]
    max_tg = max(all_tg, default=1)
    models = sorted(tg_perf.keys())
    nodes  = sorted(set(n for m in tg_perf.values() for n in m.keys()))
    n_heads = "".join(f'<th style="padding:12px 14px">{n}</th>' for n in nodes)
    rows   = ""
    for model in models:
        cells = "".join(
            f'<td class="td-bar">{bar(tg_perf[model].get(n), max_tg)}</td>'
            for n in nodes
        )
        rows += f'<tr><td class="td-dim" style="white-space:nowrap">{model}</td>{cells}</tr>'
    thead = f'<tr><th style="padding:12px 14px">Model</th>{n_heads}</tr>'
    return card(table(thead, rows or empty(len(nodes) + 1)))


def llama_raw(records):
    rows = ""
    for r in sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True):
        model = r.get("model_tag", r.get("model", ""))
        ts    = r.get("avg_ts")
        ttype = "pp512" if r.get("n_gen", 1) == 0 else "tg128"
        col   = "#00b4d8" if ttype == "pp512" else "#06d6a0"
        rows += (f'<tr>'
                 f'<td class="td-ts">{r.get("timestamp","")[:19]}</td>'
                 f'<td class="td-dim">{r.get("node","")}</td>'
                 f'<td class="td-dim" style="white-space:nowrap">{model}</td>'
                 f'<td class="td-num" style="color:{col}">{ttype}</td>'
                 f'<td class="td-num">{r.get("n_prompt","—")}p / {r.get("n_gen","—")}g</td>'
                 f'<td class="td-num" style="color:{col};font-weight:600">'
                 f'{round(ts, 1) if ts else "—"}</td>'
                 f'</tr>')
    thead = th("Timestamp", "Node", "Model", "Test", "Tokens", "tok/s")
    return card(table(thead, rows or empty(6)))


# ── Main report assembly ──────────────────────────────────────────────────────

def build_report(ollama_records, llama_records):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    all_nodes  = sorted(set(
        r.get("node", "") for r in ollama_records + llama_records if r.get("node")
    ))
    all_models = sorted(set(
        r.get("model") or r.get("model_tag", "")
        for r in ollama_records + llama_records
        if r.get("model") or r.get("model_tag")
    ))

    # Aggregate
    meta = load_model_meta()
    o_tg, o_pp, o_ttft = aggregate_ollama(ollama_records)
    l_tg, l_pp          = aggregate_llama(llama_records)

    # Summary tables
    o_sum_html, o_peak_tg = ollama_summary_table(o_tg, o_pp, o_ttft, meta)
    l_sum_html, l_peak_tg = llama_summary_table(l_tg, l_pp, meta)
    peak_tg = max(o_peak_tg, l_peak_tg)

    # Detail tables
    o_matrix_html = ollama_matrix(o_tg)
    o_raw_html    = ollama_raw(ollama_records)
    l_matrix_html = llama_matrix(l_tg)
    l_raw_html    = llama_raw(llama_records)

    # Nav links
    nav_links = ""
    if ollama_records:
        nav_links += '<a class="nav-link" href="#ollama">Ollama</a>'
    if llama_records:
        nav_links += '<a class="nav-link" href="#llamabench">llama-bench</a>'

    # Page
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Fleet Benchmark Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0d1117; color: #e2e8f0; }}

  /* Header */
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
             padding: 40px 48px; border-bottom: 1px solid #0f3460; }}
  .header h1 {{ font-size: 28px; font-weight: 700; color: #fff; letter-spacing: -0.5px; }}
  .header p  {{ color: #718096; margin-top: 6px; font-size: 14px; }}
  .stats-row {{ display: flex; gap: 16px; margin-top: 24px; flex-wrap: wrap; }}
  .stat-card {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
                border-radius: 10px; padding: 14px 20px; min-width: 130px; }}
  .stat-card .val {{ font-size: 26px; font-weight: 700; color: #00b4d8; }}
  .stat-card .lbl {{ font-size: 12px; color: #718096; margin-top: 2px; }}

  /* Sticky nav */
  .nav {{ position: sticky; top: 0; z-index: 100;
          background: #161b22; border-bottom: 1px solid #1a1a2e;
          padding: 0 48px; display: flex; gap: 4px; align-items: center; }}
  .nav-label {{ font-size: 11px; color: #4a5568; font-weight: 600;
                letter-spacing: 1px; text-transform: uppercase;
                padding: 14px 0; margin-right: 8px; }}
  .nav-link {{ display: inline-block; padding: 14px 16px; font-size: 13px;
               color: #718096; text-decoration: none; font-weight: 500;
               border-bottom: 2px solid transparent; transition: all 0.15s; }}
  .nav-link:hover {{ color: #e2e8f0; border-bottom-color: #00b4d8; }}

  /* Content */
  .content {{ padding: 32px 48px; max-width: 1400px; }}
  .section {{ margin-bottom: 36px; }}
  .section-title {{ font-size: 13px; font-weight: 700; color: #90cdf4;
                    letter-spacing: 1px; text-transform: uppercase;
                    margin-bottom: 16px; padding-bottom: 8px;
                    border-bottom: 1px solid #1a1a2e; display: flex;
                    align-items: baseline; gap: 10px; }}
  .section-sub {{ font-size: 11px; font-weight: 400; color: #4a5568;
                  text-transform: none; letter-spacing: 0; }}

  /* Divider between summary blocks */
  .summary-group {{ display: flex; flex-direction: column; gap: 32px; }}
  .summary-label {{ font-size: 11px; font-weight: 700; color: #4a5568;
                    letter-spacing: 1px; text-transform: uppercase;
                    margin-bottom: 10px; }}

  /* Collapsible drill-down */
  .drill {{ margin-bottom: 16px; border: 1px solid #1a1a2e; border-radius: 12px;
            overflow: hidden; background: #161b22; }}
  .drill > summary {{ padding: 14px 20px; font-size: 13px; font-weight: 600;
                      color: #a0aec0; cursor: pointer; list-style: none;
                      display: flex; align-items: center; gap: 10px;
                      user-select: none; }}
  .drill > summary::-webkit-details-marker {{ display: none; }}
  .drill > summary::before {{ content: "▶"; font-size: 10px; color: #4a5568;
                               transition: transform 0.2s; }}
  .drill[open] > summary::before {{ transform: rotate(90deg); }}
  .drill > summary:hover {{ color: #e2e8f0; background: rgba(255,255,255,0.02); }}
  .drill-content {{ padding: 0 0 0 0; }}

  /* Tables */
  .card {{ background: #161b22; border: 1px solid #1a1a2e; border-radius: 12px;
           overflow: hidden; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  thead {{ background: #0d1117; }}
  thead th {{ padding: 12px 14px; text-align: left; font-size: 11px;
              font-weight: 600; color: #4a5568; letter-spacing: 0.5px;
              text-transform: uppercase; }}
  tbody tr {{ border-bottom: 1px solid #1a1a2e; transition: background 0.12s; }}
  tbody tr:last-child {{ border-bottom: none; }}
  tbody tr:hover {{ background: rgba(255,255,255,0.025); }}
  .td-model {{ padding: 10px 14px; color: #e2e8f0; font-weight: 600;
               white-space: nowrap; font-size: 13px; }}
  .td-bar   {{ padding: 10px 14px; min-width: 180px; }}
  .td-dim   {{ padding: 10px 14px; color: #718096; font-size: 12px; }}
  .td-num   {{ padding: 10px 14px; color: #a0aec0; font-size: 12px;
               text-align: center; }}
  .td-ts    {{ padding: 8px 12px; color: #4a5568; font-size: 11px; }}
  .td-err   {{ padding: 8px 12px; color: #ef476f; font-size: 11px; }}
  .col-sub  {{ color: #4a5568; font-weight: 400; }}
</style>
</head>
<body>

<div class="header">
  <h1>&#9889; AI Fleet Benchmark Report</h1>
  <p>Local inference fleet — Sovereign AI Lab</p>
  <div class="stats-row">
    <div class="stat-card"><div class="val">{len(all_nodes)}</div><div class="lbl">Nodes</div></div>
    <div class="stat-card"><div class="val">{len(all_models)}</div><div class="lbl">Models</div></div>
    <div class="stat-card"><div class="val">{len(ollama_records)}</div><div class="lbl">Ollama Runs</div></div>
    <div class="stat-card"><div class="val">{len(llama_records)}</div><div class="lbl">llama-bench Runs</div></div>
    <div class="stat-card"><div class="val">{peak_tg}</div><div class="lbl">Peak TG tok/s</div></div>
    <div class="stat-card"><div class="val" style="font-size:15px;padding-top:5px">{ts}</div><div class="lbl">Generated</div></div>
  </div>
</div>

<nav class="nav">
  <span class="nav-label">Jump to</span>
  <a class="nav-link" href="#summary">Summary</a>
  {nav_links}
</nav>

<div class="content">

  <a name="summary"></a>
  <div class="section">
    <div class="section-title">Performance Summary
      <span class="section-sub">best result per model across all nodes and tests</span>
    </div>
    <div class="summary-group">

      {'<div><div class="summary-label">Ollama Benchmark</div>' + o_sum_html + '</div>'
       if ollama_records else ''}

      {'<div><div class="summary-label">llama-bench &mdash; standardised pp512 / tg128 &mdash; 3-rep median</div>' + l_sum_html + '</div>'
       if llama_records else ''}

    </div>
  </div>

  {'<a name="ollama"></a><div class="section"><div class="section-title">Ollama Benchmark <span class="section-sub">drill-down</span></div>' +
   collapsible("Node × Model Matrix — TG tok/s", o_matrix_html) +
   collapsible("Raw Results — all tests", o_raw_html) +
   '</div>' if ollama_records else ''}

  {'<a name="llamabench"></a><div class="section"><div class="section-title">llama-bench <span class="section-sub">drill-down</span></div>' +
   collapsible("Node × Model Matrix — TG tok/s", l_matrix_html) +
   collapsible("Raw Results — all runs", l_raw_html) +
   '</div>' if llama_records else ''}

</div>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate HTML benchmark report")
    parser.add_argument("--results", default="results/", help="Results directory")
    parser.add_argument("--output", help="Output HTML file path")
    args = parser.parse_args()

    ollama_records, llama_records = load_results(args.results)
    print(f"Loaded {len(ollama_records)} Ollama records, {len(llama_records)} llama-bench records")

    html = build_report(ollama_records, llama_records)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or str(REPORTS_DIR / f"benchmark_{ts}.html")
    with open(output, "w") as f:
        f.write(html)
    print(f"Report saved: {output}")


if __name__ == "__main__":
    main()
