# make_fig1_drawio.py — writes figures/fig1_pipeline.drawio (Fig. 1 of the paper).
# Open the .drawio file in draw.io (desktop or app.diagrams.net) to edit it, then
# File > Export as > PDF (crop, transparent background off) to figures/fig1_pipeline.pdf.
# Run: python figures/make_fig1_drawio.py
from pathlib import Path
from xml.sax.saxutils import escape

FONT = "fontFamily=Helvetica;fontSize=15;fontColor=#1A1A1A;"
cells, eid = [], [100]


def nid():
    eid[0] += 1
    return f"n{eid[0]}"


def box(x, y, w, h, html, style, cid=None):
    cid = cid or nid()
    cells.append(f'<mxCell id="{cid}" value="{escape(html, {chr(34): "&quot;"})}" style="{style}{FONT}html=1;whiteSpace=wrap;" '
                 f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return cid


def edge(src, dst, label="", style="", pts=None):
    cid = nid()
    geo = '<mxGeometry relative="1" as="geometry">'
    if pts:
        geo += "<Array as=\"points\">" + "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts) + "</Array>"
    geo += "</mxGeometry>"
    cells.append(f'<mxCell id="{cid}" value="{escape(label)}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;'
                 f'endArrow=block;endFill=1;strokeWidth=1.6;strokeColor=#404040;{FONT}fontStyle=1;html=1;{style}" '
                 f'edge="1" parent="1" source="{src}" target="{dst}">{geo}</mxCell>')


# palette: one light fill + darker border per stage
STAGE = {1: ("#EEF3FB", "#5B7DB1"), 2: ("#EEF6EE", "#5E9A5E"), 3: ("#FFF8E5", "#C9A227"),
         4: ("#F3EEF8", "#8062A8"), 5: ("#F4F4F4", "#7A7A7A")}
PANEL = "rounded=1;arcSize=4;strokeWidth=1.6;verticalAlign=top;"
INNER = "rounded=1;arcSize=8;fillColor=#FFFFFF;strokeWidth=1.2;"
TOP, PH = 66, 350                      # panel top and height
COLS = [(10, 205), (235, 285), (540, 265), (825, 190), (1035, 205)]
TITLES = ["Data ingestion<br>and indexing", "Hybrid retrieval", "Confidence<br>estimation",
          "Abstention<br>decision", "Answer or<br>abstain"]

panels = {}
for k, ((x, w), title) in enumerate(zip(COLS, TITLES), 1):
    fill, line = STAGE[k]
    box(x, 8, 34, 34, f"<b>{k}</b>", f"ellipse;fillColor={line};strokeColor=none;fontColor=#FFFFFF;fontSize=17;")
    box(x + 40, 2, w - 40, 48, f"<b>{title}</b>", "text;align=left;verticalAlign=middle;fontSize=16;strokeColor=none;fillColor=none;")
    panels[k] = box(x, TOP, w, PH, "", f"{PANEL}fillColor={fill};strokeColor={line};")

# 1. ingestion
x, w = COLS[0]; line = STAGE[1][1]; cx = x + 22; iw = w - 44
nvd = box(cx, TOP + 22, iw, 82, "NVD 2023 feed<br>30,932 CVEs", f"shape=cylinder3;boundedLbl=1;size=10;fillColor=#FFFFFF;strokeColor={line};strokeWidth=1.2;")
pre = box(cx, TOP + 136, iw, 70, "Filter and tag<br>(CVE, severity, CWE)", f"{INNER}strokeColor={line};")
idx = box(cx, TOP + 238, iw, 82, "BM25 index<br>+ FAISS dense index<br>(MiniLM-L6)", f"{INNER}strokeColor={line};")
edge(nvd, pre); edge(pre, idx)

# 2. hybrid retrieval
x, w = COLS[1]; line = STAGE[2][1]
q = box(x + 62, TOP + 18, w - 124, 40, "<b>Query <i>q</i></b>", "rounded=1;arcSize=50;fillColor=#FFE9D6;strokeColor=#D9822B;strokeWidth=1.2;")
bm = box(x + 14, TOP + 96, 122, 66, "BM25<br>(lexical)", f"{INNER}strokeColor={line};")
de = box(x + w - 136, TOP + 96, 122, 66, "Dense<br>(cosine)", f"{INNER}strokeColor={line};")
rrf = box(x + 40, TOP + 196, w - 80, 50, "Reciprocal rank fusion<br><i>k</i> = 60", f"{INNER}strokeColor={line};")
top = box(x + 55, TOP + 276, w - 110, 56, "Top-5 fused records", f"shape=document;boundedLbl=1;size=0.18;fillColor=#FFFFFF;strokeColor={line};strokeWidth=1.2;")
edge(q, bm, "", "exitX=0.25;exitY=1;entryX=0.5;entryY=0;")
edge(q, de, "", "exitX=0.75;exitY=1;entryX=0.5;entryY=0;")
edge(bm, rrf, "", "entryX=0.25;entryY=0;"); edge(de, rrf, "", "entryX=0.75;entryY=0;")
edge(rrf, top)
edge(idx, panels[2], "", f"exitX=1;exitY=0.5;entryX=0;entryY=0.5;")  # indexes feed retrieval

# 3. confidence estimation
x, w = COLS[2]; line = STAGE[3][1]; cx = x + 16; iw = w - 32
sig = box(cx, TOP + 18, iw, 104,
          "<b>Signals</b><br><i>s</i><sub>1</sub>, <i>s</i><sub>2</sub>: dense similarity<br>of fused ranks 1 and 2<br>"
          "<i>m</i> = (<i>s</i><sub>1</sub> &minus; <i>s</i><sub>2</sub>) / <i>s</i><sub>1</sub>",
          f"{INNER}strokeColor={line};")
sco = box(cx, TOP + 150, iw, 92,
          "<b>Confidence score</b><br><i>C</i> = &alpha;<i>s</i><sub>1</sub> + (1 &minus; &alpha;)<i>m</i><br>"
          "&alpha; = 0.6,&nbsp; 0 &le; <i>C</i> &le; 1", f"{INNER}strokeColor={line};")
note = box(cx, TOP + 266, iw, 62, "Top record found only<br>by BM25: <i>s</i><sub>1</sub> = 0, so <i>C</i> = 0",
           f"rounded=1;arcSize=8;dashed=1;fillColor=#FFFFFF;strokeColor={line};strokeWidth=1.2;fontSize=14;")
edge(sig, sco)
edge(top, sig, "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;", pts=[(525, TOP + 304), (525, TOP + 70)])

# 4. abstention decision
x, w = COLS[3]; line = STAGE[4][1]
dec = box(x + 22, TOP + 120, w - 44, 120, "<i>C</i> &ge; &theta; ?<br>&theta; = 0.20", f"rhombus;fillColor=#FFFFFF;strokeColor={line};strokeWidth=1.4;")
box(x + 14, TOP + 262, w - 28, 70, "Higher &theta;: fewer but<br>more reliable answers",
    f"rounded=1;arcSize=8;dashed=1;fillColor=#FFFFFF;strokeColor={line};strokeWidth=1.2;fontSize=14;")
edge(sco, dec, "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")

# 5. answer or abstain
x, w = COLS[4]; cx = x + 14; iw = w - 28
ans = box(cx, TOP + 20, iw, 146,
          "<b>Answer</b><br>Llama-3.2 (3B) answers<br>from the top-5 records,<br>citing CVE IDs",
          "rounded=1;arcSize=8;fillColor=#E6F2E8;strokeColor=#2E7D32;strokeWidth=1.4;")
abst = box(cx, TOP + 184, iw, 146,
           "<b>Abstain</b><br>Return &ldquo;Insufficient<br>context&rdquo;; the LLM<br>is not called",
           "rounded=1;arcSize=8;fillColor=#FBE7E7;strokeColor=#C62828;strokeWidth=1.4;")
edge(dec, ans, "Yes", "exitX=0.5;exitY=0;entryX=0;entryY=0.5;fontColor=#1B5E20;labelBackgroundColor=#F3EEF8;")
edge(dec, abst, "No", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;fontColor=#8E1B1B;labelBackgroundColor=#F3EEF8;")

xml = ('<mxfile host="drawio"><diagram id="fig1" name="Fig. 1 pipeline">'
       '<mxGraphModel dx="1250" dy="430" grid="1" gridSize="5" guides="1" page="0" math="0" shadow="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')
out = Path(__file__).with_name("fig1_pipeline.drawio")
out.write_text(xml)
print("wrote", out)
