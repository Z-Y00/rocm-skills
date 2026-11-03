"""Run-length op sequence of the biggest loop in an AMDGPU ISA dump.

    python isa_seq.py kernel.s

Letters: W wmma/mfma, L ds_load, T ds_load_tr, A s_wait_alu, D s_wait_dscnt, X tensor (TDM) ops,
v anything else. "L16 W32" means 16 LDS loads then 32 matrix ops. Compare a slow kernel's sequence
with a sibling the compiler handles well: per-step reload patterns ("L W L W ...") versus bulk
phases ("L32 W64") explain hazard-wait counts that source reordering cannot change.
"""

import re
import sys


def biggest_loop(lines):
    labels = {re.match(r"^(\.\w+):", l).group(1): i for i, l in enumerate(lines) if re.match(r"^\.\w+:", l)}
    loops = []
    for i, l in enumerate(lines):
        m = re.match(r"^\s+s_cbranch_\w+\s+(\.\w+)", l)
        if m and m.group(1) in labels and labels[m.group(1)] < i:
            loops.append((i - labels[m.group(1)], labels[m.group(1)], i))
    _, a, b = max(loops)
    return lines[a:b]


def cls(line):
    t = line.strip()
    if not t or t.startswith((";", ".")):
        return ""
    s = t.split()[0]
    if s.startswith(("v_wmma", "v_mfma", "v_swmmac", "v_smfmac")):
        return "W"
    if s.startswith("ds_load_tr") or s.startswith("ds_read_b64_tr"):
        return "T"
    if s.startswith(("ds_load", "ds_read")):
        return "L"
    if s == "s_wait_alu":
        return "A"
    if s.startswith("s_wait_dscnt"):
        return "D"
    if "tensor" in s:
        return "X"
    return "v"


def main():
    seq = "".join(cls(l) for l in biggest_loop(open(sys.argv[1]).read().split("\n")))
    out, prev, n = [], "", 0
    for c in seq + "#":
        if c == prev:
            n += 1
            continue
        if prev:
            out.append(prev if n == 1 else f"{prev}{n}")
        prev, n = c, 1
    print(" ".join(out))


if __name__ == "__main__":
    main()
