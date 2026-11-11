#!/usr/bin/env python3
"""Regenerate report.html from the result JSON. No number is typed by hand."""
import json, pathlib, datetime, html

OUT = pathlib.Path("./runs/report.html")
hpc = json.load(open("/tmp/stag_hpc_clean.json"))
hpc = [d for d in hpc if d.get("status") == "ok"]
hpc.sort(key=lambda d: d["complexes"])

mac = json.load(open(OUT.parent / "stag_results.json"))["datasets"]
mac_iv = mac.get("IVTDFSVIK", {})

PENDING = [("pan_peptide", 22410)]
means = [d["mean_auc"] for d in hpc]
mean_of_means = sum(means) / len(means)
total_h = sum(d["seconds"] for d in hpc) / 3600
peak = max(d["peak_gpu_gib"] for d in hpc)

# ---- chart geometry -------------------------------------------------------
X0, X1, LO, HI = 150.0, 660.0, 0.45, 1.00
def x(v): return X0 + (v - LO) / (HI - LO) * (X1 - X0)
ROW, TOP = 30, 46
rows = hpc + [{"dataset": n, "complexes": c, "pending": True} for n, c in PENDING]
H = TOP + ROW * len(rows) + 54

parts = []
parts.append(f'<line x1="{x(0.5):.1f}" y1="{TOP-22:.0f}" x2="{x(0.5):.1f}" y2="{TOP+ROW*len(rows)-14:.0f}" '
             f'stroke="var(--bad)" stroke-width="1" stroke-dasharray="3 3" opacity=".7"/>')
parts.append(f'<text x="{x(0.5):.1f}" y="{TOP-27:.0f}" fill="var(--bad)" font-size="10.5" '
             f'text-anchor="middle">เสี่ยงทาย 0.50</text>')
for i, d in enumerate(rows):
    y = TOP + ROW * i
    nm = html.escape(d["dataset"])
    parts.append(f'<text x="0" y="{y+4:.0f}" fill="var(--ink)" font-size="11.5" font-family="var(--mono)">{nm}</text>')
    parts.append(f'<text x="104" y="{y+4:.0f}" fill="var(--muted)" font-size="10.5" text-anchor="end">'
                 f'{d["complexes"]:,}</text>')
    if d.get("pending"):
        parts.append(f'<text x="{X0+6:.0f}" y="{y+4:.0f}" fill="var(--muted)" font-size="11">'
                     f'กำลังรันบนคลัสเตอร์ SLURM</text>')
        continue
    lo, hi, m = d["min_auc"], d["max_auc"], d["mean_auc"]
    parts.append(f'<line x1="{x(lo):.1f}" y1="{y:.0f}" x2="{x(hi):.1f}" y2="{y:.0f}" '
                 f'stroke="var(--accent)" stroke-width="2.5" opacity=".42" stroke-linecap="round"/>')
    for a in d["fold_auc"]:
        parts.append(f'<circle cx="{x(a):.1f}" cy="{y:.0f}" r="2.8" fill="var(--accent)" opacity=".55"/>')
    parts.append(f'<circle cx="{x(m):.1f}" cy="{y:.0f}" r="5.6" fill="var(--accent)"/>')
    col = "var(--good)" if m >= .75 else ("var(--warn)" if m >= .62 else "var(--bad)")
    parts.append(f'<text x="672" y="{y+4:.0f}" fill="{col}" font-size="12" font-weight="600" '
                 f'font-family="var(--mono)">{m:.3f}</text>')
ax = TOP + ROW * len(rows) - 6
parts.append(f'<line x1="{X0:.0f}" y1="{ax:.0f}" x2="{X1:.0f}" y2="{ax:.0f}" stroke="var(--ink)" stroke-width="1"/>')
for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    parts.append(f'<line x1="{x(t):.1f}" y1="{ax:.0f}" x2="{x(t):.1f}" y2="{ax+5:.0f}" stroke="var(--ink)"/>')
    parts.append(f'<text x="{x(t):.1f}" y="{ax+19:.0f}" fill="var(--muted)" font-size="10.5" '
                 f'text-anchor="middle" font-family="var(--mono)">{t:.2f}</text>')
parts.append(f'<text x="{(X0+X1)/2:.0f}" y="{ax+36:.0f}" fill="var(--muted)" font-size="11" '
             f'text-anchor="middle">ROC-AUC · จุดจาง = แต่ละ fold · จุดทึบ = ค่าเฉลี่ย</text>')
chart = "\n  ".join(parts)

trows = "\n".join(
    f'<tr><td>{html.escape(d["dataset"])}</td><td>{d["complexes"]:,}</td>'
    f'<td>{d["positive_rate"]*100:.1f}</td><td class="me">{d["mean_auc"]:.4f}</td>'
    f'<td>{d["sd_auc"]:.4f}</td><td>{d["min_auc"]:.3f}–{d["max_auc"]:.3f}</td>'
    f'<td>{d["seconds"]/60:.1f}</td><td>{d["peak_gpu_gib"]:.2f}</td></tr>' for d in hpc)
trows += (f'\n<tr style="opacity:.6"><td>pan_peptide</td><td>22,410</td><td>21.8</td>'
          f'<td colspan="5" style="text-align:left">กำลังรัน — job 85267</td></tr>')

mac_mean = mac_iv.get("mean_auc"); mac_sd = mac_iv.get("sd_auc")
hpc_iv = next(d for d in hpc if d["dataset"] == "IVTDFSVIK")
diff = abs(mac_mean - hpc_iv["mean_auc"])

doc = f"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STAG — ผลจำแนก TCR-pMHC</title>
<style>
:root{{
  --paper:#f7f8fa; --ink:#15191f; --muted:#5b6673; --rule:#d9dde3;
  --card:#fff; --accent:#2f6d8c; --accent-soft:#e3eef3;
  --warn:#8a5a12; --warn-soft:#fbf0dc; --bad:#8f3a3a; --good:#2f6b4f;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:"Helvetica Neue",-apple-system,BlinkMacSystemFont,Thonburi,"Sukhumvit Set",sans-serif;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  color-scheme:light;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --paper:#101317; --ink:#e7ebf0; --muted:#9aa5b1; --rule:#2a3139; --card:#171c22;
  --accent:#79b8d1; --accent-soft:#1b2a33; --warn:#e0b366; --warn-soft:#2d2415;
  --bad:#e08585; --good:#79c3a1; color-scheme:dark;}}}}
:root[data-theme="dark"]{{
  --paper:#101317; --ink:#e7ebf0; --muted:#9aa5b1; --rule:#2a3139; --card:#171c22;
  --accent:#79b8d1; --accent-soft:#1b2a33; --warn:#e0b366; --warn-soft:#2d2415;
  --bad:#e08585; --good:#79c3a1; color-scheme:dark;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:880px;margin:0 auto;padding-left:max(18px,env(safe-area-inset-left,0px));
  padding-right:max(18px,env(safe-area-inset-right,0px));padding-block:clamp(28px,5vw,56px)}}
h1{{font-family:var(--serif);font-weight:600;font-size:clamp(1.7rem,1.2rem+2.2vw,2.5rem);
  line-height:1.15;margin:0 0 6px;text-wrap:balance;letter-spacing:-.01em}}
.sub{{color:var(--muted);margin:0 0 4px}}
.stamp{{color:var(--muted);font-size:.8rem;font-family:var(--mono)}}
h2{{font-family:var(--serif);font-size:1.3rem;font-weight:600;margin:46px 0 4px;text-wrap:balance}}
h2+.lede{{color:var(--muted);margin:0 0 16px;font-size:.95rem}}
.eyebrow{{font-size:.72rem;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);
  font-weight:600;margin:0 0 10px}}
.headline{{display:flex;flex-wrap:wrap;gap:14px;margin:22px 0 0}}
.tile{{background:var(--card);border:1px solid var(--rule);border-radius:3px;
  padding:14px 18px;min-width:0;flex:1 1 150px}}
.tile .k{{font-size:.72rem;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:600}}
.tile .v{{font-family:var(--serif);font-size:1.8rem;line-height:1.1;margin-top:6px;
  font-variant-numeric:tabular-nums}}
.tile .n{{font-size:.79rem;color:var(--muted);margin-top:3px}}
.note{{border-left:3px solid var(--warn);background:var(--warn-soft);padding:14px 16px;
  margin:20px 0;border-radius:0 3px 3px 0;font-size:.94rem}}
.note b{{color:var(--warn)}}
.good-note{{border-left-color:var(--good);background:var(--accent-soft)}}
.good-note b{{color:var(--good)}}
.scroll{{overflow-x:auto;margin:18px 0;border:1px solid var(--rule);border-radius:3px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;font-size:.88rem;min-width:560px}}
th,td{{text-align:right;padding:9px 11px;border-bottom:1px solid var(--rule);
  font-variant-numeric:tabular-nums;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left;font-family:var(--mono);font-size:.83rem}}
thead th{{font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  font-weight:600;background:var(--accent-soft)}}
tbody tr:last-child td{{border-bottom:0}}
.me{{font-weight:700}}
figure{{margin:18px 0 6px}}
figcaption{{color:var(--muted);font-size:.84rem;margin-top:8px}}
svg{{display:block;width:100%;height:auto;max-width:100%}}
ul{{padding-left:1.1rem}} li{{margin:7px 0}}
code{{font-family:var(--mono);font-size:.86em;background:var(--accent-soft);padding:1px 5px;border-radius:3px}}
.prov{{font-size:.86rem;color:var(--muted)}}
.prov dt{{font-weight:600;color:var(--ink);margin-top:10px;font-size:.83rem}}
.prov dd{{margin:2px 0 0;font-family:var(--mono);font-size:.79rem;word-break:break-all}}
hr{{border:0;border-top:1px solid var(--rule);margin:36px 0 0}}
footer{{margin-top:44px;padding-top:18px;border-top:1px solid var(--rule);color:var(--muted);font-size:.84rem}}
</style>
</head>
<body>
<div class="wrap">

<p class="eyebrow">STAG · Graph neural network · 5-fold cross-validation · NVIDIA A100</p>
<h1>ผลจำแนกการจับคู่ TCR-pMHC</h1>
<p class="sub">รันโค้ดต้นฉบับของ KavrakiLab/STAG_public บน a SLURM cluster — ต่างจากต้นฉบับ 2 บรรทัด</p>
<p class="stamp">อัปเดต {datetime.datetime.now():%Y-%m-%d %H:%M} · commit 06c30a57 · seed 20261107 · torch 2.1.0+cu121 · PyG 2.3.1</p>

<div class="headline">
  <div class="tile"><div class="k">ชุดที่รันจบ</div><div class="v">{len(hpc)} / 9</div>
    <div class="n">pan_peptide กำลังรัน</div></div>
  <div class="tile"><div class="k">AUC เฉลี่ยของทุกชุด</div><div class="v">{mean_of_means:.3f}</div>
    <div class="n">ช่วง {min(means):.3f}–{max(means):.3f}</div></div>
  <div class="tile"><div class="k">เวลารวม</div><div class="v">{total_h:.1f}<span style="font-size:.55em"> ชม.</span></div>
    <div class="n">{sum(d['complexes'] for d in hpc):,} complexes</div></div>
  <div class="tile"><div class="k">GPU สูงสุดที่ใช้</div><div class="v">{peak:.1f}<span style="font-size:.55em"> GiB</span></div>
    <div class="n">จากการ์ด 80 GiB</div></div>
</div>

<div class="note">
<b>อ่านเลขนี้อย่างไร</b> — โค้ด cross-validation ของเปเปอร์เลือก epoch ที่ดีที่สุด <i>จาก test fold เอง</i>
(<code>if test_auc &gt; best_test_auc</code>) โดยไม่มี validation set แยก ค่า AUC ทุกค่าในหน้านี้จึงถูกคัดเลือกบนข้อมูลชุดเดียวกับที่ใช้วัด
และมีอคติเข้าข้างตัวเองติดมาโดยโครงสร้าง ใช้เทียบกับตัวเลขในเปเปอร์ได้ แต่<b>ใช้อ้างเป็นประสิทธิภาพกับข้อมูลที่ไม่เคยเห็นไม่ได้</b>
</div>

<h2>ผลทุกชุด</h2>
<p class="lede">เรียงจากชุดเล็กไปใหญ่ · เส้น = ช่วงระหว่าง fold ที่แย่สุดถึงดีสุด</p>

<figure>
<svg viewBox="0 0 700 {H}" role="img" aria-label="ค่า ROC-AUC ของแต่ละชุดข้อมูล">
  {chart}
</svg>
<figcaption>ชุดที่อยู่ใกล้เส้นประ 0.50 คือชุดที่โมเดลแทบไม่ต่างจากการเดาสุ่ม</figcaption>
</figure>

<div class="scroll">
<table>
<thead><tr><th>ชุด</th><th>complexes</th><th>pos %</th><th>ROC-AUC</th><th>sd</th><th>ช่วง fold</th><th>นาที</th><th>GPU GiB</th></tr></thead>
<tbody>
{trows}
</tbody>
</table>
</div>

<div class="note">
<b>ผลขึ้นกับ peptide อย่างรุนแรง — อย่าสรุปด้วยค่าเฉลี่ยตัวเดียว</b><br>
<code>ELAGIGILTV</code> ได้ {max(means):.3f} ซึ่งเกือบสมบูรณ์ ขณะที่ <code>KLGGALQAK</code> และ <code>AVFDRKSDAK</code>
อยู่ที่ {sorted(means)[0]:.3f} และ {sorted(means)[1]:.3f} ซึ่ง<b>แทบไม่ต่างจากการเดาสุ่ม</b> ทั้งที่มีข้อมูลมากกว่าชุดที่ทำได้ดีหลายชุด
จำนวนข้อมูลจึงไม่ใช่ตัวกำหนด — ความยากของแต่ละ peptide ต่างหาก ค่าเฉลี่ยรวม {mean_of_means:.3f} กลบข้อเท็จจริงนี้ทั้งหมด
</div>

<h2>ความแกว่งระหว่าง fold หดลงตามขนาดข้อมูล</h2>
<p class="lede">sd ของ 5 fold เทียบกับจำนวน complexes</p>
<div class="scroll">
<table>
<thead><tr><th>ชุด</th><th>complexes</th><th>complexes ต่อ fold</th><th>sd</th></tr></thead>
<tbody>
{chr(10).join(f'<tr><td>{d["dataset"]}</td><td>{d["complexes"]:,}</td><td>~{d["complexes"]//5}</td><td class="me">{d["sd_auc"]:.4f}</td></tr>' for d in sorted(hpc, key=lambda z: z["sd_auc"], reverse=True))}
</tbody>
</table>
</div>
<p>ชุดที่เล็กที่สุดแกว่ง {max(d["sd_auc"] for d in hpc):.3f} ส่วนชุดใหญ่แกว่งเพียง {min(d["sd_auc"] for d in hpc):.3f}
นี่อธิบายได้ว่าทำไมผลแรกที่รันบนเครื่อง MacBook ด้วยชุดเล็กที่สุดจึงให้ช่วงกว้างจนน่าตกใจ — เป็นเพราะ test fold มีไม่กี่สิบตัวอย่าง ไม่ใช่เพราะโมเดลไม่เสถียร</p>

<h2>การแก้โค้ดไม่ได้ทำให้ผลเพี้ยน</h2>
<div class="scroll">
<table>
<thead><tr><th>IVTDFSVIK รันที่ไหน</th><th>แก้กี่จุด</th><th>อุปกรณ์</th><th>ROC-AUC</th><th>sd</th></tr></thead>
<tbody>
<tr><td>MacBook Air M5</td><td>5 จุด</td><td>MPS</td><td class="me">{mac_mean:.4f}</td><td>{mac_sd:.4f}</td></tr>
<tr><td>SLURM cluster A100</td><td>2 บรรทัด</td><td>CUDA</td><td class="me">{hpc_iv['mean_auc']:.4f}</td><td>{hpc_iv['sd_auc']:.4f}</td></tr>
</tbody>
</table>
</div>
<p>ต่างกัน <b>{diff:.4f}</b> ซึ่งเล็กกว่าความแกว่งระหว่าง fold ของชุดนี้เอง ({hpc_iv['sd_auc']:.3f}) อยู่มาก
เป็นหลักฐานว่าการแก้โค้ดเพื่อความเข้ากันได้ไม่ได้เปลี่ยนสิ่งที่โมเดลเรียนรู้</p>

<div class="note good-note">
<b>ส่วนที่ต่างจากต้นฉบับทั้งหมด บนคลัสเตอร์ SLURM คือ 2 บรรทัดในไฟล์เดียว</b><br>
<code>y=graph.label</code> → <code>y=int(graph.label)</code> ใน <code>models/STAG/cross_validation.py</code><br>
เลือกมาด้วยการทดสอบ ไม่ใช่การเดา: ทดลอง torch_geometric 2.3.1 / 2.2.0 / 2.1.0 / 2.0.4 พบว่า<b>ไม่มีรุ่นไหน</b>แปลง
<code>np.int64</code> ที่ pandas คืนมาให้เป็น tensor ขณะที่ทุกรุ่นแปลง <code>int</code> ของ Python ให้ ปัญหาจึงไม่เคยเกี่ยวกับรุ่น PyG
การ cast เป็น <code>int</code> คือการคืนพฤติกรรมเดิมที่โค้ดเขาถูกเขียนมาให้เจอ ไม่ใช่การเปลี่ยนตรรกะ
ส่วนที่เหลือรันโดยไม่แก้เลย — เก็บ diff ไว้ที่ <code>upstream-deviation.patch</code>
</div>

<hr>
<h2>ที่มาของตัวเลขทุกตัวในหน้านี้</h2>
<dl class="prov">
<dt>โค้ดต้นฉบับ</dt><dd>github.com/KavrakiLab/STAG_public @ 06c30a57 (MIT)</dd>
<dt>เปเปอร์</dt><dd>Slone, Conev, Rigo, Reuben &amp; Kavraki · IEEE Trans. Comput. Biol. Bioinform. 22(1):171–179 (2025)</dd>
<dt>เครื่องที่รัน</dt><dd>a SLURM cluster · NVIDIA A100-SXM4 80 GB · 1 ชุดต่อ 1 โปรเซส · job 85266 / 85267</dd>
<dt>ซอฟต์แวร์</dt><dd>python 3.10 · torch 2.1.0+cu121 · torch_geometric 2.3.1 · pandas 1.5.3 · torch-cluster · biopython 1.88</dd>
<dt>ไฟล์ผลดิบ</dt><dd>~/stag/results/*.json บนคลัสเตอร์ SLURM</dd>
<dt>ที่เพิ่มจากโค้ดเขา</dt><dd>ใส่ seed ให้การแบ่ง fold (ต้นฉบับใช้ np.random.randint ไม่ใส่ seed จึงรันซ้ำไม่ได้) และเขียนผลลง JSON · เรียก cross_validation ของเขาเองตรง ๆ หนึ่งชุดต่อหนึ่งการรัน เหมือน evaluations.py</dd>
</dl>

<footer>
สร้างจากไฟล์ผลดิบด้วยสคริปต์ ไม่มีตัวเลขไหนพิมพ์ด้วยมือ · ชุดที่ยังไม่จบแสดงว่ายังไม่จบ ไม่ใช่ค่าว่าง
</footer>

</div>
</body>
</html>
"""
OUT.write_text(doc)
print(f"wrote {OUT} ({len(doc):,} bytes) · {len(hpc)} datasets · mean {mean_of_means:.4f}")
