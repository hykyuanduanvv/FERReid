"""Every-5-rounds record of the 20-round re-runs: mAP / R1 / purity at rounds 5, 10, 15, 20 per group (parsed from
run.log while running), plus seed means and paired differences once rounds are available. NZDT times."""
import glob, re, os, time, datetime
import numpy as np
os.chdir("/data1/yangbin/dz/code/exp_1008_eps/experiments")
pat = re.compile(r"\[(\S+) r(\d+)\] queries (\d+) .*?pos (\d+) .*?mAP ([\d.]+|-) R1 ([\d.]+|-)")
R = (5, 10, 15, 20)
def parse(d):
    out = {}
    for line in open(d + "/run.log", errors="ignore"):
        m = pat.search(line)
        if m and m.group(5) != "-":
            out[int(m.group(2))] = (float(m.group(5)), float(m.group(6)), int(m.group(4)))
    return out
runs = {d: parse(d) for d in sorted(glob.glob("l20_*")) if os.path.exists(d + "/run.log")}
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=13))).strftime("%Y-%m-%d %H:%M")
L = ["# 20 轮复跑：每 5 轮记录（%s NZDT 更新）" % now, "",
     "题只在第 1–5 轮出（CUHK03 共 605 题，MSMT17 共 1490 题），第 6–20 轮只训练。每格是 mAP / Rank-1；括号里是种子数。", "",
     "| 组 | 已到轮次 | R5 | R10 | R15 | R20 | R16–20 均值 | 累计\"是\" |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
for d, r in runs.items():
    last = max(r) if r else 0
    cell = lambda k: "%.2f / %.1f" % r[k][:2] if k in r else ""
    plat = "%.2f / %.1f" % tuple(np.mean([r[k][:2] for k in range(16, 21)], 0)) if all(k in r for k in range(16, 21)) else ""
    done = " ✓" if os.path.exists(d + "/.done") else ""
    L.append("| %s%s | %d | %s | %s | %s | %s | %s | %s |" % (d, done, last, cell(5), cell(10), cell(15), cell(20), plat, r[last][2] if r else ""))
L += ["", "## 按种子平均（只统计该轮已有的种子）", "", "| 组 | R5 | R10 | R15 | R20 | R16–20 平台 |", "|---|---|---|---|---|---|"]
groups = {}
for d in runs:
    groups.setdefault(re.sub(r"_s\d$", "", d), []).append(runs[d])
for g, rs in groups.items():
    def agg(k):
        v = [r[k][:2] for r in rs if k in r]
        return "%.2f / %.1f (%d)" % (*np.mean(v, 0), len(v)) if v else ""
    pv = [np.mean([r[k][:2] for k in range(16, 21)], 0) for r in rs if all(k in r for k in range(16, 21))]
    L.append("| %s | %s | %s | %s | %s | %s |" % (g, agg(5), agg(10), agg(15), agg(20), "%.2f / %.1f (%d)" % (*np.mean(pv, 0), len(pv)) if pv else ""))
open("L20_每5轮记录.md.tmp", "w").write("\n".join(L) + "\n")
os.replace("L20_每5轮记录.md.tmp", "L20_每5轮记录.md")
