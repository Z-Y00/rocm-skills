"""Mnemonic histogram diff of the biggest loop of two AMDGPU ISA dumps.

    python isa_loop_hist.py base.s variant.s

Prints the mnemonics whose counts changed most (base, variant, delta); `s_wait_alu` is split by
its depctr field. Diffing a slow kernel against a sibling that runs well is the fastest way to
find compiler-inserted costs: bank switches (`s_set_vgpr_msb`) from register pressure above v255,
hazard waits (`s_wait_alu depctr_va_vdst`), spill traffic (`scratch_*`), or lost CSE (extra
`ds_load_*`).
"""

import re
import sys
from collections import Counter


def loop_hist(path):
    lines = open(path).read().split("\n")
    labels = {re.match(r"^(\.\w+):", l).group(1): i for i, l in enumerate(lines) if re.match(r"^\.\w+:", l)}
    best = max(
        (i - labels[m.group(1)], labels[m.group(1)], i)
        for i, l in enumerate(lines)
        for m in [re.match(r"^\s+s_cbranch_\w+\s+(\.\w+)", l)]
        if m and m.group(1) in labels and labels[m.group(1)] < i
    )
    c = Counter()
    for l in lines[best[1] : best[2]]:
        t = l.strip().split()
        if t and not t[0].startswith((";", ".")):
            k = t[0]
            if k == "s_wait_alu" and len(t) > 1:
                k += " " + t[1].split("(")[0]
            c[k] += 1
    return c


def main(rows=30):
    a, b = loop_hist(sys.argv[1]), loop_hist(sys.argv[2])
    print(f"{'mnemonic':<36} {'base':>6} {'var':>6} {'delta':>6}")
    for k in sorted(set(a) | set(b), key=lambda k: -abs(b[k] - a[k]))[:rows]:
        print(f"{k:<36} {a[k]:>6} {b[k]:>6} {b[k] - a[k]:>+6}")
    print(f"{'TOTAL':<36} {sum(a.values()):>6} {sum(b.values()):>6} {sum(b.values()) - sum(a.values()):>+6}")


if __name__ == "__main__":
    main()
