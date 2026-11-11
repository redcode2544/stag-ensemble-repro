#!/usr/bin/env python3
"""One table. What each dataset scored, and which model produced it.

Oat, 2026-11-11: "report งงมากเลขเยอะไปหมดไม่มีประโยชน์เลย ขึ้นตารางสรุปพอละ
บอกว่าใช้ model อะไรตรงได้ชัดๆด้วยในตาราง"

Input : runs/results/fusion_all.json
Output: runs/report_choice.html  (superseded by local/build_matrix.py, which owns
        runs/report.html since Oat asked for the model x dataset matrix instead)
        The long version lives in runs/report_full.html (local/build_report.py).

Run with python3 3.9+:  /usr/bin/python3 local/build_summary.py
"""
import collections
import datetime
import html
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent.parent
FUSION = HERE / "runs" / "results" / "fusion_all.json"
OUT = HERE / "runs" / "report_choice.html"
BAR = 0.75
HEAD = "auto2"
# "label เชื่อได้" is the paper's own cut, by HLA, not a judgement of ours:
# A*03 and A*11 are the alleles it flags. Deriving it here instead of hard-coding
# a list — a hand-typed list got NLVPMVATV and IVTDFSVIK the wrong way round once.
SUSPECT_HLA = ("A*03", "A*11")
CURATED = HERE / "ext"   # curated_<ds>_indep.csv = published TCR pairs after the audit

# Plain Thai for the fusion vocabulary. The point of the table is that a reader
# never has to look any of these up.
POOL = {"struct": "โครงสร้างล้วน", "seq": "ลำดับล้วน", "con": "สถิติ contact",
        "data": "ข้อมูลนอกล้วน", "esm": "ESM-2 ล้วน",
        "structseq": "โครงสร้าง + ลำดับ", "all": "ทุกสาย", "*": "ทุกสาย"}
RULE = {"soft": "เฉลี่ยคะแนน", "dprime": "ถ่วงน้ำหนักตามความแม่น", "maxconf": "เอาตัวที่มั่นใจสุด",
        "stacking": "เรียนน้ำหนักเอง (stacking)", "committee": "โหวตเสียงข้างมาก"}
# Which stream each member name belongs to, for the "best single model" column.
STREAM = {"STAG": "โครงสร้าง", "Contacts": "โครงสร้าง", "CNN_3D": "โครงสร้าง",
          "STAG_EGNN": "โครงสร้าง", "STAG_DUAL": "โครงสร้าง",
          "SeqLR": "ลำดับ", "SeqGBM": "ลำดับ", "SeqMLP": "ลำดับ", "SeqKNN": "ลำดับ",
          "ConGBM": "สถิติ contact", "ConLR": "สถิติ contact",
          "SeqExt": "ข้อมูลนอก", "SeqExtOnly": "ข้อมูลนอก", "SeqClean": "ข้อมูลนอก",
          "SeqNN2": "ข้อมูลนอก", "SeqDedup": "ข้อมูลนอก", "SeqPU": "ข้อมูลนอก",
          "SeqAvg3": "ข้อมูลนอก", "SeqESM": "ESM-2", "SeqESMNN": "ESM-2"}


def auc(node):
    return node["auc"] if isinstance(node, dict) else node


def published_pairs(ds):
    """How many paired TCRs the three public databases hold for this peptide,
    after the provenance audit dropped the studies that duplicate the target."""
    f = CURATED / f"curated_{ds}_indep.csv"
    if not f.exists():
        return None
    return max(0, sum(1 for _ in f.open(encoding="utf-8", errors="replace")) - 1)


def describe(picks):
    """'ทุกสาย · เรียนน้ำหนักเอง (stacking)' + how many folds agreed."""
    counted = collections.Counter(picks).most_common()
    out = []
    for tag, n in counted:
        pool, _, rule = tag.partition("/")
        out.append((f"{POOL.get(pool, pool)} · {RULE.get(rule, rule)}", n))
    return out


def main():
    rep = json.loads(FUSION.read_text(encoding="utf-8"))
    names = [d for d in rep if not d.startswith("_")]
    rows = []
    for d in names:
        r = rep[d]
        picks = r.get("_picks", {}).get(HEAD) or r.get("_picks", {}).get("auto", [])
        member, _, seed = r["best_member"].partition("|")
        rows.append({
            "ds": d, "hla": r["hla"], "auc": auc(r["fusion"][HEAD]),
            "used": describe(picks), "folds": len(picks) or 5,
            "n": r["n_members"], "single": member, "single_auc": r["best_member_auc"],
            "stream": STREAM.get(member, "—"),
            "trusted": not r["hla"].startswith(SUSPECT_HLA),
            "pairs": published_pairs(d),
        })
    rows.sort(key=lambda r: -r["auc"])

    mean8 = sum(r["auc"] for r in rows) / len(rows)
    trust = [r for r in rows if r["trusted"]]
    assert len(trust) == 5, f"expected 5 trusted datasets, got {[r['ds'] for r in trust]}"
    mean5 = sum(r["auc"] for r in trust) / len(trust)
    passed = sum(1 for r in rows if r["auc"] >= BAR)
    gain = sum(r["auc"] - r["single_auc"] for r in rows) / len(rows)

    trs = []
    for r in rows:
        used = "<br>".join(
            f'<span class="use">{html.escape(txt)}</span>'
            f'<span class="folds">{n}/{r["folds"]} fold</span>'
            for txt, n in r["used"])
        cls = "pass" if r["auc"] >= BAR else ("weak" if r["auc"] < 0.62 else "")
        flag = "✓" if r["auc"] >= BAR else ""
        note = "" if r["trusted"] else ' <span class="warn">HLA ที่เปเปอร์ตัดเอง</span>'
        if r["pairs"] == 0:
            note += ' <span class="bad">ไม่มี TCR ตีพิมพ์เลย</span>'
        pairs = ("—" if r["pairs"] is None
                 else f"{r['pairs']:,} คู่" if r["pairs"] else "0 คู่")
        trs.append(f"""<tr class="{cls}">
  <td class="ds"><b>{html.escape(r['ds'])}</b>{note}<span class="hla">{html.escape(r['hla'])}
    · TCR ที่ตีพิมพ์แล้ว {pairs}</span></td>
  <td class="used">{used}<span class="nmem">รวม {r['n']} โมเดลย่อย</span></td>
  <td class="num big">{r['auc']:.3f}</td>
  <td class="flag">{flag}</td>
  <td class="single">{html.escape(r['single'])}<span class="st">{html.escape(r['stream'])}</span></td>
  <td class="num">{r['single_auc']:.3f}</td>
  <td class="num delta">{r['auc'] - r['single_auc']:+.3f}</td>
</tr>""")

    doc = f"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STAG — ตารางสรุป</title>
<style>
:root {{
  --ink:#15202b; --mute:#5b6b7c; --line:#dde3ea; --paper:#fff; --soft:#f6f8fa;
  --good:#0f766e; --warn:#b45309; --dim:#9aa7b4;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--soft); color:var(--ink);
  font-family:-apple-system,"Segoe UI","Noto Sans Thai",sans-serif; font-size:17px; line-height:1.6; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:40px 24px 64px; }}
h1 {{ font-size:30px; line-height:1.25; margin:4px 0 8px; letter-spacing:-.3px; }}
.eyebrow {{ font-size:13px; letter-spacing:1.4px; text-transform:uppercase; color:var(--dim); margin:0; }}
.sub {{ color:var(--mute); margin:0 0 28px; font-size:16px; }}
.tiles {{ display:flex; gap:14px; flex-wrap:wrap; margin:0 0 30px; }}
.tile {{ background:var(--paper); border:1px solid var(--line); border-radius:4px;
  padding:14px 18px; flex:1 1 180px; }}
.tile .k {{ font-size:13px; color:var(--mute); }}
.tile .v {{ font-size:30px; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.15; }}
.tile .n {{ font-size:13px; color:var(--dim); }}
table {{ width:100%; border-collapse:collapse; background:var(--paper);
  border:1px solid var(--line); border-radius:4px; overflow:hidden; }}
th {{ text-align:left; font-size:13px; letter-spacing:.6px; text-transform:uppercase;
  color:var(--mute); font-weight:700; padding:12px 14px; border-bottom:2px solid var(--ink);
  background:var(--paper); }}
td {{ padding:14px; border-bottom:1px solid var(--line); vertical-align:top; font-size:16px; }}
tr:last-child td {{ border-bottom:none; }}
tr.pass {{ background:#f3fbf9; }}
tr.weak td {{ color:var(--mute); }}
.ds b {{ font-size:17px; letter-spacing:.2px; }}
.hla, .st, .nmem, .folds {{ display:block; font-size:13px; color:var(--dim); }}
.folds {{ display:inline; margin-right:8px; color:var(--dim); }}
.use {{ display:inline-block; min-width:230px; }}
.warn {{ font-size:12px; color:var(--warn); background:#fffbeb; border:1px solid #f3d18a;
  border-radius:3px; padding:1px 6px; white-space:nowrap; }}
.bad {{ font-size:12px; color:#9f1239; background:#fff1f2; border:1px solid #f0abbc;
  border-radius:3px; padding:1px 6px; white-space:nowrap; }}
.folds {{ margin-left:10px; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap;
  font-family:ui-monospace,Menlo,monospace; }}
.big {{ font-size:21px; font-weight:700; }}
.delta {{ color:var(--good); }}
.flag {{ text-align:center; color:var(--good); font-size:19px; font-weight:700; }}
tfoot td {{ border-top:2px solid var(--ink); font-weight:700; background:var(--paper); }}
.note {{ background:#fffbeb; border:1px solid #f3d18a; border-radius:4px;
  padding:16px 18px; margin:28px 0 0; font-size:15.5px; }}
.note b {{ color:var(--warn); }}
.how {{ color:var(--mute); font-size:15px; margin:22px 0 10px; }}
.more {{ margin-top:30px; font-size:15px; color:var(--mute); }}
.more a {{ color:var(--good); }}
.stamp {{ color:var(--dim); font-size:13px; margin-top:34px; }}
@media (max-width:820px) {{ .use {{ min-width:0; }} td,th {{ padding:10px 8px; font-size:15px; }} }}
</style>
</head>
<body>
<div class="wrap">

<p class="eyebrow">TCR–pMHC binding · {len(rows)} ชุดข้อมูล · 5-fold CV · 3 seed</p>
<h1>แต่ละชุดได้เท่าไหร่ และใช้โมเดลอะไร</h1>
<p class="sub">ทำซ้ำ KavrakiLab/STAG_public แล้วรวมโมเดลห้าสายเข้าด้วยกัน</p>

<div class="tiles">
  <div class="tile"><div class="k">เฉลี่ย {len(rows)} ชุด</div><div class="v">{mean8:.3f}</div>
    <div class="n">ดีกว่าโมเดลเดี่ยวที่ดีสุด {gain:+.3f}</div></div>
  <div class="tile"><div class="k">เฉลี่ย {len(trust)} ชุดที่ label เชื่อได้</div><div class="v">{mean5:.3f}</div>
    <div class="n">ตัด A*03 / A*11 ตามเกณฑ์ของเปเปอร์เอง</div></div>
  <div class="tile"><div class="k">ผ่านเกณฑ์ {BAR:.2f}</div><div class="v">{passed} / {len(rows)}</div>
    <div class="n">เกณฑ์ที่พี่โอ๊ตตั้งไว้</div></div>
</div>

<p class="how">ตารางเรียงจากคะแนนมากไปน้อย · <b>โมเดลที่ใช้</b> คือสูตรที่ถูกเลือกจาก fold อื่น
ไม่เคยเลือกจาก fold ที่กำลังให้คะแนน · ถ้าบาง fold เลือกคนละสูตร จะแสดงทุกสูตรพร้อมจำนวน fold</p>

<table>
<thead><tr>
  <th>ชุดข้อมูล</th><th>โมเดลที่ใช้</th><th>AUC</th><th>ผ่าน</th>
  <th>ถ้าใช้โมเดลเดียว</th><th>ได้</th><th>ต่าง</th>
</tr></thead>
<tbody>
{chr(10).join(trs)}
</tbody>
<tfoot><tr>
  <td>เฉลี่ยทั้ง {len(rows)} ชุด</td><td></td><td class="num big">{mean8:.3f}</td><td></td>
  <td></td><td class="num">{sum(r['single_auc'] for r in rows)/len(rows):.3f}</td>
  <td class="num delta">{gain:+.3f}</td>
</tr></tfoot>
</table>

<div class="note">
<b>อ่านตัวเลขพวกนี้ยังไง</b><br>
AUC ทุกตัวในตาราง — และในเปเปอร์ต้นฉบับ — มีการเลือก epoch บน fold ที่กำลังวัดปนอยู่ราว <b>+0.03</b>
(วัดได้จริง: +0.029 ที่ k=5, +0.050 ที่ k=10) เทียบกันเองยุติธรรม แต่ค่าสัมบูรณ์สูงเกินจริง<br>
KLGGALQAK ไม่มี TCR ที่ตีพิมพ์ในสามฐานข้อมูลให้ตรวจ label เลยแม้แต่คู่เดียว (หลังตัดชุด 10x ที่ซ้ำกับข้อมูลต้นทาง)
ส่วน AVFDRKSDAK กับ IVTDFSVIK มีอย่างละ 36 และ 37 คู่ — คะแนนของสามชุดนี้บอกความไม่รู้ของข้อมูล
มากกว่าความสามารถของโมเดล
</div>

<p class="more">รายละเอียดทั้งหมด — ทุกสมาชิก ทุกสูตร ที่มาของข้อมูล การตรวจสอบย้อน และสิ่งที่ลองแล้วไม่เวิร์ก —
อยู่ใน <a href="report_full.html">report_full.html</a></p>

<p class="stamp">อัปเดต {datetime.datetime.now():%Y-%m-%d %H:%M} ·
สร้างจาก runs/results/fusion_all.json ด้วย local/build_summary.py</p>

</div>
</body>
</html>
"""
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({len(doc):,} bytes) · {len(rows)} datasets · "
          f"mean {mean8:.3f} · {passed}/{len(rows)} over {BAR}")


if __name__ == "__main__":
    main()
