import os
import json
from common import RESULTS, load_json

EHS = ["2.41421356", "3.03150807", "3.12880294", "3.15225457", "3.15698186", "3.15802086",
       "3.15824198", "3.15828833", "3.15829845", "3.15830062"]


def fmt(x, d=9):
    return ("%." + str(d) + "f") % x


def cascade_rows():
    c = load_json("cascade.json")
    long = load_json("cascade_long.json")["0.5"]
    rows = []
    for n in range(1, len(long) + 1):
        cells = [str(n)]
        for nu in ["0.2"]:
            cells.append(fmt(c[nu][n - 1]) if n <= len(c[nu]) else "")
        cells.append(fmt(long[n - 1]))
        cells.append(EHS[n - 1])
        for nu in ["0.8", "1.0"]:
            cells.append(fmt(c[nu][n - 1]) if n <= len(c[nu]) else "")
        if n >= 3:
            cells.append("%.4f" % ((long[n - 2] - long[n - 3]) / (long[n - 1] - long[n - 2])))
        else:
            cells.append("")
        rows.append(" & ".join(cells) + r"\\")
    return "\n".join(rows)


def count_rows(Ts=(10 ** 3, 10 ** 7), deltas=(1e-4, 1e-6, 1e-8, 1e-10)):
    kc = load_json("kernel_counts.json")
    rows = []
    for T in Ts:
        for d in deltas:
            cells = ["$10^{%d}$" % round(__import__("math").log10(T)), "$10^{%d}$" % round(__import__("math").log10(d))]
            for nu in (0.2, 0.5, 0.8):
                r = [x for x in kc if x["nu"] == nu and x["T"] == T and x["delta"] == d][0]
                cells += [str(r["lb"]), r"\textbf{%d}" % r["r"], str(r["K0"]), str(r["soe"]),
                          str(r["focq"]) if r["focq"] else "--"]
            rows.append(" & ".join(cells) + r"\\")
    return "\n".join(rows)


def sci(x):
    if x is None:
        return "--"
    if x == float("inf"):
        return "diverges"
    if x == 0:
        return "$0$"
    e = int(__import__("math").floor(__import__("math").log10(abs(x))))
    m = x / 10 ** e
    return r"$%.1f\cdot10^{%d}$" % (m, e)


def num(x, digits=3):
    if x is None:
        return "--"
    if x >= 100:
        return "%.0f" % x
    if x >= 10:
        return "%.1f" % x
    if x >= 1:
        return "%.2f" % x
    return "%.3f" % x


def bench_rows(Ts=(2 ** 12, 2 ** 14, 2 ** 16, 2 ** 20), errT=(2 ** 14, 2 ** 16)):
    b = load_json("bench_T.json")
    names = {"DS": "DS", "SM": r"SM, $L_{\rm s}=1000$", "NM": r"NM, $W=32$", "FFT": "FFT", "SOE": "SOE",
             "FOCQ": "FOCQ", "WHC": r"\textbf{WHC (ours)}"}
    rows = []
    for m in ["DS", "SM", "NM", "FFT", "SOE", "FOCQ", "WHC"]:
        def get(T):
            r = [x for x in b if x["method"] == m and x["T"] == T]
            return r[0] if r else None
        cells = [names[m]]
        for T in Ts:
            r = get(T)
            cells.append(num(r["time"]) if r and not r["skipped"] else "--")
        for T in Ts:
            r = get(T)
            cells.append(num(r["mem"] / 2 ** 20) if r and not r["skipped"] else "--")
        for T in errT:
            r = get(T)
            if m == "FFT" and r and not r["skipped"]:
                cells.append("reference")
            else:
                cells.append(sci(r["err"]) if r and not r["skipped"] else "--")
        rows.append(" & ".join(cells) + r"\\")
    return "\n".join(rows)


if __name__ == "__main__":
    out = {"cascade": cascade_rows(), "counts": count_rows()}
    if os.path.exists(os.path.join(RESULTS, "bench_T.json")):
        out["bench"] = bench_rows()
    with open(os.path.join(RESULTS, "tables.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(k)
        print(v)
