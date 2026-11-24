"""Computed vs. reported values -> outputs/COMPARISON.md and comparison.json.

Status per item:
  MATCH         the computed value rounds to the printed value (or satisfies
                the printed inequality); counts and strings match exactly
  MISMATCH      otherwise
  NOT COMPUTED  the key is absent from results.json
"""
from __future__ import annotations

import json
from pathlib import Path

from . import reference_values as RV


def _status(key, ref, val):
    if val is None:
        return "NOT COMPUTED", None
    if key in RV.UPPER_BOUND:
        return ("MATCH" if val <= ref else "MISMATCH"), ref
    if key in RV.LOWER_BOUND:
        return ("MATCH" if val >= ref else "MISMATCH"), ref
    if isinstance(ref, (bool, str)) or (isinstance(ref, int) and not isinstance(val, float)):
        return ("MATCH" if val == ref else "MISMATCH"), 0
    tol = 0.5 * 10 ** (-RV.decimals(key)) + 1e-9
    return ("MATCH" if abs(float(val) - float(ref)) <= tol else "MISMATCH"), tol


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.0f}"
    return str(v)


def run(out_dir: Path, log=print):
    out_dir = Path(out_dir)
    R = json.loads((out_dir / "results.json").read_text())
    rows, counts = [], {"MATCH": 0, "MISMATCH": 0, "NOT COMPUTED": 0}
    for key, ref in RV.ARTICLE.items():
        st, tol = _status(key, ref, R.get(key))
        counts[st] += 1
        rows.append({"key": key, "article": ref, "computed": R.get(key), "status": st,
                     "tolerance": tol})
    lines = ["# Computed vs. reported values", "",
             "Article: `TII-Articles-LaTeX-template/tii-articles-template.tex`. "
             "A printed value matches when the computed value rounds to it.", "",
             f"**MATCH {counts['MATCH']} | MISMATCH {counts['MISMATCH']} | "
             f"NOT COMPUTED {counts['NOT COMPUTED']}**", "",
             "| Key | Article | Computed | Status |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['key']}` | {_fmt(r['article'])} | {_fmt(r['computed'])} | "
                     f"{'**' + r['status'] + '**' if r['status'] != 'MATCH' else r['status']} |")
    (out_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    (out_dir / "comparison.json").write_text(json.dumps({"counts": counts, "rows": rows},
                                                        indent=1, default=str))
    log(f"comparison: {counts}")
    return counts, rows
