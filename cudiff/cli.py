from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .gen import kernel
from .harness import check, default_configs
from .reduce import reduce


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cudiff")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("configs")
    p = sub.add_parser("gen"); p.add_argument("--seed", type=int, default=1)
    p = sub.add_parser("run")
    p.add_argument("-n", type=int, default=100); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="results/latest"); p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--reduce", action="store_true"); p.add_argument("--stmts", type=int, default=12)
    a = ap.parse_args(argv)

    if a.cmd == "configs":
        for c in default_configs():
            print(f"{c.name:16s} {' '.join(c.cmd)}")
        return 0
    if a.cmd == "gen":
        sys.stdout.write(kernel(a.seed).source); return 0
    if a.cmd == "run":
        cfgs = default_configs()
        os.makedirs(os.path.join(a.out, "candidates"), exist_ok=True)
        counts = {}
        cands = []
        t0 = time.time()
        log = open(os.path.join(a.out, "campaign.jsonl"), "a")

        def one(seed):
            k = kernel(seed, a.stmts)
            return seed, k, check(k, cfgs)

        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            futs = [ex.submit(one, a.seed + i) for i in range(a.n)]
            done = 0
            for fut in as_completed(futs):
                seed, k, v = fut.result()
                done += 1
                counts[v.kind] = counts.get(v.kind, 0) + 1
                log.write(json.dumps({"seed": seed, "kind": v.kind, "note": v.note[:400]}) + "\n"); log.flush()
                if v.kind in ("diverge", "compiler-crash"):
                    path = os.path.join(a.out, "candidates", f"k{seed}.cu")
                    with open(path, "w") as f:
                        f.write(k.source)
                    rec = {"seed": seed, "kind": v.kind, "note": v.note, "source": path,
                           "results": [{"config": r.config, "status": r.status, "cpu": r.cpu, "gpu": r.gpu} for r in v.results]}
                    if a.reduce and v.kind == "diverge":
                        r = reduce(k, cfgs)
                        rec["reduced_stmts"] = r.stmts
                        with open(os.path.join(a.out, "candidates", f"k{seed}_reduced.cu"), "w") as f:
                            f.write(r.source)
                    cands.append(rec)
                if done % 10 == 0 or done == a.n:
                    print(f"[{done}/{a.n}] {counts}", file=sys.stderr, flush=True)
        summary = {"n": a.n, "seed": a.seed, "stmts": a.stmts, "configs": [c.name for c in cfgs],
                   "counts": counts, "seconds": round(time.time() - t0, 1), "candidates": cands}
        json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
        print(json.dumps(counts))
        return 0
    return 1
