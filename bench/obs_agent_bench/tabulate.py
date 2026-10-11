"""Per-task pass counts across trial transcripts, as a markdown table.

    python3 tabulate.py base=run/a-t0.jsonl,run/a-t1.jsonl iter1=run/b-t0.jsonl,...

Each column is a label; each cell is passes/trials for that task.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hard_tasks  # noqa: E402

CHECKS = {t["id"]: t["check"] for t in hard_tasks.HARD_TASKS}


def load(paths):
    """Re-checks hard-suite answers with the current checkers (the task's
    data dir sits next to the transcript), so a checker fix rescores old runs."""
    out = {}
    for p in paths:
        label = os.path.basename(p)[: -len(".jsonl")]
        for line in open(p):
            r = json.loads(line)
            ok = bool(r["pass"])
            d = os.path.join(os.path.dirname(p), label, r["task"], "obs")
            if r["task"] in CHECKS and os.path.isdir(d):
                ok = bool(CHECKS[r["task"]](r["answer"], d)[0])
            out.setdefault(r["task"], []).append(ok)
    return out


def main():
    cols = []
    for arg in sys.argv[1:]:
        label, paths = arg.split("=", 1)
        cols.append((label, load(paths.split(","))))
    tasks = []
    for _, d in cols:
        for t in d:
            if t not in tasks:
                tasks.append(t)
    print("| task | " + " | ".join(lbl for lbl, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for t in tasks:
        cells = []
        for _, d in cols:
            v = d.get(t)
            cells.append(f"{sum(v)}/{len(v)}" if v else "-")
        print(f"| {t} | " + " | ".join(cells) + " |")
    tot = []
    for _, d in cols:
        p = sum(sum(v) for v in d.values())
        n = sum(len(v) for v in d.values())
        tot.append(f"**{p}/{n}** ({100 * p / max(n, 1):.0f}%)")
    print("| total | " + " | ".join(tot) + " |")


if __name__ == "__main__":
    main()
