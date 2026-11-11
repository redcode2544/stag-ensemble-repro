#!/usr/bin/env python3
"""The whole result as one matrix: rows = datasets, columns = models.

Oat, 2026-11-11: "ทำตาราง column เป็น model row เป็น class ปะ ผมจะดูเอง"
So: no narrative, no selection story. Every model's AUC on every dataset,
the fused column at the end, and nothing else competing for attention.

Input : runs/results/fusion_all.json
Output: runs/report.html   (overwritten in place — Oat's standing instruction)
"""
import datetime
import html
import json
import pathlib
from statistics import mean

HERE = pathlib.Path(__file__).resolve().parent.parent
FUSION = HERE / "runs" / "results" / "fusion_all.json"
OUT = HERE / "runs" / "report.html"
BAR = 0.75
# Two fused columns, not one: the simple rule and the selected one. They are
# within 0.001 of each other and Oat should see that rather than be told it.
HEAD = "all_stacking"      # all five streams, stacked
HEAD2 = "auto2"            # pool AND rule chosen off-fold (what the README quotes)

# Column order: grouped by stream, and the group is shown as a spanning header
# so a reader can see at a glance which family a column belongs to.
GROUPS = [
    ("โครงสร้าง", ["STAG", "Contacts", "CNN_3D"]),
    ("ลำดับ TCR", ["SeqLR", "SeqGBM", "SeqMLP", "SeqKNN"]),
    ("contact", ["ConGBM", "ConLR"]),
    ("ข้อมูลนอก", ["SeqExt", "SeqExtOnly", "SeqClean", "SeqNN2", "SeqDedup", "SeqPU", "SeqAvg3"]),
    ("ESM-2", ["SeqESM", "SeqESMNN"]),
]
COLS = [m for _, ms in GROUPS for m in ms]


def shade(v):
    """Light heat so a row can be scanned without reading every number."""
    if v is None:
        return "background:#fafbfc"
    x = max(0.0, min(1.0, (v - 0.50) / 0.45))        # 0.50 -> 0.95
    # white -> teal
    r, g, b = int(255 - 60 * x), int(255 - 20 * x), int(255 - 45 * x)
    return f"background:rgb({r},{g},{b})"


def main():
    rep = json.loads(FUSION.read_text(encoding="utf-8"))
    DS = [d for d in rep if not d.startswith("_")]
    DS.sort(key=lambda d: -rep[d]["fusion"][HEAD]["auc"])

    def cell(d, m):
        bm = rep[d]["by_method"].get(m)
        return bm["mean"] if bm else None

    fused = {d: rep[d]["fusion"][HEAD]["auc"] for d in DS}
    fused2 = {d: rep[d]["fusion"][HEAD2]["auc"] for d in DS}
    trusted = [d for d in DS if not rep[d]["hla"].startswith(("A*03", "A*11"))]

    # header
    g1 = "".join(f'<th colspan="{len(ms)}" class="grp">{html.escape(g)}</th>'
                 for g, ms in GROUPS)
    g2 = "".join(f'<th class="m">{html.escape(m)}</th>' for m in COLS)

    body = []
    for d in DS:
        vals = {m: cell(d, m) for m in COLS}
        present = [v for v in vals.values() if v is not None]
        top = max(present) if present else None
        tds = []
        for m in COLS:
            v = vals[m]
            if v is None:
                tds.append('<td class="na" title="ไม่มีข้อมูลให้เทรนบนชุดนี้">—</td>')
            else:
                best = " best" if v == top else ""
                tds.append(f'<td class="v{best}" style="{shade(v)}">{v:.3f}</td>')
        warn = "" if d in trusted else '<span class="w" title="A*03 / A*11 — HLA ที่เปเปอร์ตัดเอง">!</span>'
        f = fused[d]
        f2 = fused2[d]
        tds.append(f'<td class="f {"pass" if f >= BAR else ""}">{f:.3f}</td>')
        tds.append(f'<td class="f2 {"pass" if f2 >= BAR else ""}">{f2:.3f}</td>')
        tds.append(f'<td class="d">{max(f, f2) - top:+.3f}</td>' if top else "<td></td>")
        body.append(f'<tr><th class="ds">{html.escape(d)}{warn}'
                    f'<span class="hla">{html.escape(rep[d]["hla"])}</span></th>'
                    + "".join(tds) + "</tr>")

    # per-model mean across datasets
    foot = []
    for m in COLS:
        vs = [cell(d, m) for d in DS]
        vs = [v for v in vs if v is not None]
        foot.append(f'<td class="v">{mean(vs):.3f}</td>' if vs else "<td>—</td>")
    foot_html = ('<tr><th class="ds">เฉลี่ยทุกชุด</th>' + "".join(foot)
                 + f'<td class="f">{mean(fused.values()):.3f}</td>'
                 + f'<td class="f2">{mean(fused2.values()):.3f}</td><td></td></tr>')

    doc = f"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STAG — ตารางโมเดล × ชุดข้อมูล</title>
<style>
body {{ margin:0; background:#f6f8fa; color:#15202b; font-size:16px;
  font-family:-apple-system,"Segoe UI","Noto Sans Thai",sans-serif; }}
.wrap {{ padding:34px 26px 60px; }}
h1 {{ font-size:26px; margin:0 0 6px; }}
.sub {{ color:#5b6b7c; margin:0 0 22px; font-size:15px; }}
.scroll {{ overflow-x:auto; background:#fff; border:1px solid #dde3ea; border-radius:4px; }}
table {{ border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th, td {{ padding:7px 9px; text-align:center; white-space:nowrap; font-size:15px; }}
th.grp {{ font-size:12px; letter-spacing:1px; text-transform:uppercase; color:#5b6b7c;
  border-bottom:1px solid #dde3ea; border-left:2px solid #15202b; background:#fff; }}
th.m {{ font-size:13px; color:#15202b; font-weight:700; border-bottom:2px solid #15202b;
  background:#fff; position:sticky; top:0; }}
th.ds {{ text-align:left; font-size:15px; font-weight:700; background:#fff;
  position:sticky; left:0; border-right:2px solid #15202b; padding-right:14px; }}
.hla {{ display:block; font-size:12px; color:#9aa7b4; font-weight:400; }}
td.v {{ font-family:ui-monospace,Menlo,monospace; border-bottom:1px solid #eef2f6; }}
td.best {{ font-weight:700; outline:2px solid #0f766e; outline-offset:-2px; }}
td.na {{ color:#c3ccd6; border-bottom:1px solid #eef2f6; }}
td.f {{ font-family:ui-monospace,Menlo,monospace; font-weight:700; font-size:17px;
  border-left:2px solid #15202b; border-bottom:1px solid #eef2f6; background:#fff; }}
td.f.pass, td.f2.pass {{ background:#e8f7f3; }}
td.f2 {{ font-family:ui-monospace,Menlo,monospace; font-weight:700; font-size:17px;
  border-bottom:1px solid #eef2f6; background:#fff; color:#5b6b7c; }}
td.d {{ font-family:ui-monospace,Menlo,monospace; font-size:13px; color:#0f766e;
  border-bottom:1px solid #eef2f6; }}
tfoot th.ds, tfoot td {{ border-top:2px solid #15202b; font-weight:700; background:#fff; }}
.w {{ color:#b45309; font-weight:700; margin-left:5px; }}
.legend {{ color:#5b6b7c; font-size:14px; margin:16px 0 0; line-height:1.8; }}
.legend b {{ color:#15202b; }}
code {{ background:#eef2f6; padding:1px 5px; border-radius:3px; font-size:13px; }}
</style>
</head>
<body><div class="wrap">

<h1>AUC ของทุกโมเดล บนทุกชุดข้อมูล</h1>
<p class="sub">แถว = ชุดข้อมูล (peptide) · คอลัมน์ = โมเดล · ค่าในช่อง = AUC เฉลี่ย 5 fold × 3 seed</p>

<div class="scroll">
<table>
<thead>
<tr><th class="ds" rowspan="2">ชุดข้อมูล</th>{g1}
    <th rowspan="2" class="grp">รวมทุกสาย<br>(stacking)</th>
    <th rowspan="2" class="grp">เลือกสูตร<br>เองต่อชุด</th>
    <th rowspan="2" class="grp">ต่างจาก<br>ตัวดีสุด</th></tr>
<tr>{g2}</tr>
</thead>
<tbody>
{chr(10).join(body)}
</tbody>
<tfoot>{foot_html}</tfoot>
</table>
</div>

<p class="legend">
<b>กรอบเขียว</b> = โมเดลเดี่ยวที่ดีสุดของแถวนั้น · <b>สีเข้ม</b> = คะแนนสูง ·
<b>—</b> = ชุดนั้นไม่มีข้อมูลให้เทรนวิธีนี้ (KLGGALQAK ไม่มี TCR ที่ตีพิมพ์เลย) ·
<b>!</b> = HLA A*03 / A*11 ที่เปเปอร์ตัดเอง<br>
<b>รวมทุกสาย</b> = เอาทั้ง {len(COLS)} โมเดล ({rep[DS[0]]['n_members']} สมาชิก) มา stack ด้วยน้ำหนักที่เรียนจาก
fold อื่น · เฉลี่ย {len(DS)} ชุด <b>{mean(fused.values()):.3f}</b> · เฉพาะ {len(trusted)} ชุดที่ไม่ใช่ A*03/A*11
<b>{mean(fused[d] for d in trusted):.3f}</b><br>
<b>เลือกสูตรเองต่อชุด</b> = ตัวที่ README อ้างอยู่ตอนนี้ เลือกทั้ง pool และ rule จาก fold อื่น ·
เฉลี่ย {len(DS)} ชุด <b>{mean(fused2.values()):.3f}</b> · {len(trusted)} ชุด
<b>{mean(fused2[d] for d in trusted):.3f}</b> — <b>ต่างกัน {mean(fused.values()) - mean(fused2.values()):+.4f}</b>
คือแทบไม่ต่าง ทั้งที่อันแรกอธิบายง่ายกว่ามาก<br>
ทุกค่าในตารางมีการเลือก epoch บน fold ที่วัดปนอยู่ราว <b>+0.03</b> เหมือนกันหมด จึงเทียบกันเองได้ แต่ค่าสัมบูรณ์สูงเกินจริง
</p>

<p class="legend">รายละเอียดทั้งหมดอยู่ใน <a href="report_full.html">report_full.html</a> ·
สร้างด้วย <code>local/build_matrix.py</code> · อัปเดต {datetime.datetime.now():%Y-%m-%d %H:%M}</p>

</div></body></html>
"""
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({len(doc):,} bytes) · {len(DS)} rows x {len(COLS)} models "
          f"· fused mean {mean(fused.values()):.4f}")


if __name__ == "__main__":
    main()
