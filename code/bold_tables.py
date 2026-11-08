"""
Mark the best value in each column of the generated result tables.

Runs after make_tables_v2.py. Rows are grouped by \\midrule. Within a
group, the best value of each listed column is set in bold. For the
detection table, bold marks the best compressed model; exploratory
cells (dagger) are not compared.

Usage: python code/bold_tables.py --dir "<manuscript>/tables"
"""
import argparse
import os
import re

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
COMPRESSED = ("GF-IDS (ours)", "MLP-INT8", "BNN-INT8IO")

# file: {column index: "max" | "min"}
SPECS = {
    "tab_int8.tex": {2: "min", 3: "max", 4: "max", 5: "min"},
    "tab_readiness.tex": {3: "min", 4: "min"},
    "tab_stability.tex": {2: "max", 3: "max", 4: "max", 5: "min"},
    "tab_controls_nfton.tex": {2: "max", 3: "max", 4: "min"},
    "tab_classweight_nfton.tex": {1: "max", 2: "min", 3: "max", 4: "min"},
    "tab_noniid_nfton.tex": {1: "max", 2: "max", 3: "min", 4: "max",
                             5: "max", 6: "min", 7: "max", 8: "max", 9: "min"},
    "tab_nonoverlap_nfton.tex": {1: "max", 2: "max", 3: "max", 4: "max"},
}
DETECTION = {1: "max", 2: "max", 3: "min", 4: "min",
             5: "max", 6: "max", 7: "min", 8: "min"}


def value(cell):
    if "dagger" in cell or "multicolumn" in cell:
        return None
    m = NUM.search(cell.replace("$", " "))
    return float(m.group(0).replace(",", "")) if m else None


def best_rows(rows, col, mode, allowed=None):
    vals = [(i, value(r[col])) for i, r in enumerate(rows)
            if col < len(r)
            # A spanning cell changes the relationship between split('&')
            # indices and logical columns (e.g. readiness 'not reached').
            and not any("multicolumn" in cell for cell in r)
            and (allowed is None or allowed(r))]
    vals = [(i, v) for i, v in vals if v is not None]
    if len(vals) < 2:
        return set()
    target = max(v for _, v in vals) if mode == "max" else min(v for _, v in vals)
    return {i for i, v in vals if v == target}


def wrap(cell, cmd):
    s = cell.strip()
    return f" \\{cmd}{{{s}}} "


def process(path, spec, detection=False):
    lines = open(path, encoding="utf-8").read().split("\n")
    out, group, body = [], [], False

    def flush():
        rows = [[c for c in g.rstrip("\\ ").split("&")] for g in group]
        if rows:
            for col, mode in (DETECTION if detection else spec).items():
                if detection:
                    for i in best_rows(rows, col, mode,
                                       lambda r: r[0].strip() in COMPRESSED):
                        rows[i][col] = wrap(rows[i][col], "textbf")
                else:
                    for i in best_rows(rows, col, mode):
                        rows[i][col] = wrap(rows[i][col], "textbf")
        out.extend("&".join(r) + " \\\\" for r in rows)
        group.clear()

    for ln in lines:
        s = ln.strip()
        if s == "\\midrule":
            if body:
                flush()
            out.append(ln)
            body = True
        elif s == "\\bottomrule":
            flush()
            out.append(ln)
            body = False
        elif body and s.endswith("\\\\") and "&" in s:
            group.append(s)
        else:
            out.append(ln)
    open(path, "w", encoding="utf-8").write("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    for f, spec in SPECS.items():
        p = os.path.join(a.dir, f)
        if os.path.exists(p):
            process(p, spec)
    p = os.path.join(a.dir, "tab_detection.tex")
    if os.path.exists(p):
        process(p, None, detection=True)
    print("bolded")


if __name__ == "__main__":
    main()
