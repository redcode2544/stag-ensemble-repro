#!/usr/bin/env python3
"""Turn the Thai report into an English one, and prove nothing was missed.

The report generator (build_report.py, 1,477 lines) has Thai prose woven through
its f-strings. Rewriting it would risk the numbers. Instead this works on the
RENDERED html, where every Thai string appears verbatim as a text node:

    extract  runs/report.html            -> local/i18n_en.json   (keys only)
    apply    runs/report.html + the json -> runs/report_en.html  (+ verifies)

"Verifies" means: after substitution, the output contains zero Thai codepoints.
That check is the whole point — a translation table cannot silently half-finish.

Usage:
    python3 local/i18n_report.py extract
    python3 local/i18n_report.py apply
"""
import json
import pathlib
import re
import sys
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent.parent
SRC = HERE / "runs" / "report.html"
MAP = HERE / "local" / "i18n_en.json"
OUT = HERE / "runs" / "report_en.html"

THAI = re.compile(r"[฀-๿]")
# A text node is anything between tags. Attribute values never hold Thai prose
# here (checked: the only attributes with text are class names and hrefs).
NODE = re.compile(r"(?<=>)([^<>]*[฀-๿][^<>]*)(?=<)")


def strings(html):
    """Every distinct Thai-bearing text node, longest first.

    Longest first matters at substitution time: shorter strings are often
    substrings of longer ones, and replacing the short one first would corrupt
    the long one beyond recognition.
    """
    seen = {}
    for m in NODE.finditer(html):
        s = m.group(1)
        if s.strip():
            seen[s] = seen.get(s, 0) + 1
    return sorted(seen, key=lambda s: (-len(s), s)), seen


def cmd_extract():
    html = SRC.read_text(encoding="utf-8")
    keys, counts = strings(html)
    existing = json.loads(MAP.read_text(encoding="utf-8")) if MAP.exists() else {}
    table = {k: existing.get(k, "") for k in keys}
    MAP.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    done = sum(1 for v in table.values() if v)
    print(f"{len(keys)} distinct Thai text nodes "
          f"({sum(counts.values())} occurrences, {sum(len(k) for k in keys)} chars)")
    print(f"{done} already translated, {len(keys) - done} empty -> {MAP}")


def cmd_apply():
    html = SRC.read_text(encoding="utf-8")
    table = json.loads(MAP.read_text(encoding="utf-8"))
    keys, _ = strings(html)

    missing = [k for k in keys if not table.get(k)]
    if missing:
        print(f"REFUSING: {len(missing)} strings have no translation. First 5:")
        for k in missing[:5]:
            print("   ", k[:90])
        sys.exit(1)

    still_thai = {k: v for k, v in table.items() if THAI.search(v or "")}
    if still_thai:
        print(f"REFUSING: {len(still_thai)} translations still contain Thai. First 5:")
        for k, v in list(still_thai.items())[:5]:
            print("   ", v[:90])
        sys.exit(1)

    out = html
    for k in keys:                      # already longest-first
        out = out.replace(k, table[k])

    leftover = [m.group(0) for m in NODE.finditer(out)]
    if leftover:
        print(f"REFUSING: {len(leftover)} Thai nodes survived substitution. First 5:")
        for s in leftover[:5]:
            print("   ", s[:90])
        sys.exit(1)

    # Anything Thai anywhere else (attributes, script, stray) would show as a
    # rendering bug for a reader with no Thai font, so check the whole file.
    stray = THAI.findall(out)
    if stray:
        i = THAI.search(out).start()
        print(f"REFUSING: {len(stray)} Thai characters outside text nodes, "
              f"first near: ...{out[max(0, i-60):i+60]}...")
        sys.exit(1)

    out = out.replace('<html lang="th"', '<html lang="en"')
    if '<html lang="en"' not in out:
        out = re.sub(r"<html\b", '<html lang="en"', out, count=1)
    OUT.write_text(out, encoding="utf-8")
    print(f"wrote {OUT} ({len(out):,} bytes) — 0 Thai characters remain")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "extract"
    {"extract": cmd_extract, "apply": cmd_apply}[what]()
