"""Professional PDF report builder (ReportLab): cover, TOC, figures, tables, references."""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, ListFlowable, ListItem, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

NAVY = colors.HexColor("#1F4E79")
LIGHT = colors.HexColor("#EAF1F8")
GREY = colors.HexColor("#6C757D")
ACCENT = colors.HexColor("#E07B39")

BODY, BOLD, ITAL, BOLDITAL, MONO = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique", "Courier"


def _register_fonts() -> None:
    """Use matplotlib's bundled DejaVu fonts (full Unicode) when available."""
    global BODY, BOLD, ITAL, BOLDITAL, MONO
    try:
        import matplotlib

        d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        files = {"DejaVu": "DejaVuSans.ttf", "DejaVu-Bold": "DejaVuSans-Bold.ttf", "DejaVu-Italic": "DejaVuSans-Oblique.ttf",
                 "DejaVu-BoldItalic": "DejaVuSans-BoldOblique.ttf", "DejaVuMono": "DejaVuSansMono.ttf"}
        if all((d / f).exists() for f in files.values()):
            for name, f in files.items():
                if name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(name, str(d / f)))
            pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Italic",
                                          boldItalic="DejaVu-BoldItalic")
            BODY, BOLD, ITAL, BOLDITAL, MONO = "DejaVu", "DejaVu-Bold", "DejaVu-Italic", "DejaVu-BoldItalic", "DejaVuMono"
    except Exception:
        pass


def _styles() -> dict[str, ParagraphStyle]:
    return {
        "body": ParagraphStyle("body", fontName=BODY, fontSize=9.5, leading=14.5, alignment=TA_JUSTIFY, spaceAfter=7),
        "bullet": ParagraphStyle("bullet", fontName=BODY, fontSize=9.5, leading=14, alignment=TA_LEFT),
        "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=17, leading=21, textColor=NAVY, spaceBefore=6, spaceAfter=10),
        "h1x": ParagraphStyle("h1x", fontName=BOLD, fontSize=17, leading=21, textColor=NAVY, spaceBefore=6, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=12.5, leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=5),
        "h3": ParagraphStyle("h3", fontName=BOLD, fontSize=10.5, leading=14, textColor=colors.HexColor("#333333"), spaceBefore=8, spaceAfter=3),
        "caption": ParagraphStyle("caption", fontName=ITAL, fontSize=8.3, leading=11, textColor=GREY, alignment=TA_CENTER, spaceAfter=10),
        "title": ParagraphStyle("title", fontName=BOLD, fontSize=27, leading=33, textColor=NAVY, alignment=TA_LEFT),
        "subtitle": ParagraphStyle("subtitle", fontName=BODY, fontSize=13, leading=18, textColor=GREY, alignment=TA_LEFT),
        "meta": ParagraphStyle("meta", fontName=BODY, fontSize=9, leading=13, textColor=colors.HexColor("#444444")),
        "cell": ParagraphStyle("cell", fontName=BODY, fontSize=8, leading=10.5, alignment=TA_LEFT),
        "cellh": ParagraphStyle("cellh", fontName=BOLD, fontSize=8, leading=10.5, textColor=colors.white, alignment=TA_LEFT),
        "ref": ParagraphStyle("ref", fontName=BODY, fontSize=8.3, leading=11.5, leftIndent=18, firstLineIndent=-18, spaceAfter=4),
        "box": ParagraphStyle("box", fontName=BODY, fontSize=9.5, leading=14, alignment=TA_LEFT, textColor=colors.HexColor("#222222")),
        "toc1": ParagraphStyle("toc1", fontName=BOLD, fontSize=10.5, leading=17, leftIndent=0),
        "toc2": ParagraphStyle("toc2", fontName=BODY, fontSize=9.5, leading=14, leftIndent=16),
    }


# ----------------------------------------------------------------------
def md_inline(s: str) -> str:
    s = re.sub(r"\{\{[^}]*\}\}", "", s)
    s = escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", rf'<font face="{MONO}">\1</font>', s)
    s = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", r'<font color="#1F4E79">[\1]</font>', s)
    return s


class _Doc(BaseDocTemplate):
    def __init__(self, path: str, short_title: str, **kw):
        super().__init__(path, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2.2 * cm,
                         bottomMargin=2.0 * cm, **kw)
        self.short_title = short_title
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame], onPage=self._cover),
            PageTemplate(id="body", frames=[frame], onPage=self._page),
        ])

    def _cover(self, canv, doc):
        canv.saveState()
        canv.setFillColor(NAVY)
        canv.rect(0, A4[1] - 1.6 * cm, A4[0], 1.6 * cm, stroke=0, fill=1)
        canv.setFillColor(ACCENT)
        canv.rect(0, A4[1] - 1.75 * cm, A4[0], 0.15 * cm, stroke=0, fill=1)
        canv.restoreState()

    def _page(self, canv, doc):
        canv.saveState()
        canv.setStrokeColor(colors.HexColor("#CCCCCC"))
        canv.line(self.leftMargin, 1.6 * cm, A4[0] - self.rightMargin, 1.6 * cm)
        canv.setFont(BODY, 7.5)
        canv.setFillColor(GREY)
        canv.drawString(self.leftMargin, 1.15 * cm, self.short_title[:90])
        canv.drawRightString(A4[0] - self.rightMargin, 1.15 * cm, f"Page {doc.page}")
        canv.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            name = flowable.style.name
            if name in ("h1", "h2"):
                text = re.sub(r"<[^>]+>", "", flowable.getPlainText())
                level = 0 if name == "h1" else 1
                key = f"h{self.seq.nextf('toc')}"
                self.canv.bookmarkPage(key)
                self.notify("TOCEntry", (level, text, self.page, key))


def _table(columns: list, rows: list, st: dict, width: float) -> Table:
    cols = [str(c) for c in columns]
    data = [[Paragraph(escape(c), st["cellh"]) for c in cols]]
    for r in rows:
        data.append([Paragraph(md_inline(str(c)) if len(str(c)) < 400 else escape(str(c)[:400]), st["cell"]) for c in r])
    lens = [max([len(cols[i])] + [min(len(str(r[i])), 60) for r in rows if i < len(r)]) for i in range(len(cols))]
    tot = sum(max(l, 6) for l in lens)
    widths = [max(width * max(l, 6) / tot, 1.6 * cm) for l in lens]
    scale = width / sum(widths)
    widths = [w * scale for w in widths]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D3DE")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _fig_num(cid: str) -> str:
    m = re.search(r"(\d+)", cid)
    return m.group(1) if m else cid


def _blocks(md: str, st: dict, width: float, charts: dict, tables: dict, used: set) -> list:
    """Convert light markdown + {{FIGURE:id}}/{{TABLE:id}} tokens to flowables."""
    out: list = []
    para: list[str] = []
    bullets: list[str] = []
    numbered: list[str] = []

    def flush():
        nonlocal para, bullets, numbered
        if para:
            out.append(Paragraph(md_inline(" ".join(para)), st["body"]))
            para = []
        if bullets:
            out.append(ListFlowable([ListItem(Paragraph(md_inline(b), st["bullet"]), leftIndent=14) for b in bullets],
                                    bulletType="bullet", start="\u2022", leftIndent=14, bulletFontSize=8))
            out.append(Spacer(1, 6))
            bullets = []
        if numbered:
            out.append(ListFlowable([ListItem(Paragraph(md_inline(b), st["bullet"]), leftIndent=16) for b in numbered],
                                    bulletType="1", leftIndent=16, bulletFontName=BODY, bulletFontSize=9))
            out.append(Spacer(1, 6))
            numbered = []

    for raw in md.splitlines():
        line = raw.rstrip()
        tok = re.fullmatch(r"\s*\{\{(FIGURE|TABLE):(\w+)\}\}\s*", line)
        if tok:
            flush()
            kind, cid = tok.group(1), tok.group(2)
            if kind == "FIGURE" and cid in charts and cid not in used:
                ch = charts[cid]
                used.add(cid)
                try:
                    from PIL import Image as PILImage  # reportlab dependency

                    with PILImage.open(ch.path) as im:
                        w, h = im.size
                except Exception:
                    w, h = 8, 4.5
                dw = min(width, 15.5 * cm)
                dh = dw * h / w
                if dh > 11 * cm:
                    dh, dw = 11 * cm, 11 * cm * w / h
                cap = Paragraph(f"<b>Figure {_fig_num(cid)}.</b> {md_inline(ch.caption or ch.title)}", st["caption"])
                out.append(KeepTogether([Image(ch.path, width=dw, height=dh), Spacer(1, 3), cap]))
            elif kind == "TABLE" and cid in tables and ("t:" + cid) not in used:
                tb = tables[cid]
                used.add("t:" + cid)
                cap = Paragraph(f"<b>Table {_fig_num(cid)}.</b> {md_inline(tb.caption or tb.title)}", st["caption"])
                out.append(Spacer(1, 4))
                out.append(_table(tb.columns, tb.rows, st, width))
                out.append(Spacer(1, 3))
                out.append(cap)
            continue
        if not line.strip():
            flush()
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            lvl = min(len(m.group(1)), 3)
            out.append(Paragraph(md_inline(m.group(2)), st["h2" if lvl <= 2 else "h3"]))
            continue
        m = re.match(r"^\s*[-*\u2022]\s+(.*)", line)
        if m:
            if para or numbered:
                flush()
            bullets.append(m.group(1))
            continue
        m = re.match(r"^\s*\d+[.)]\s+(.*)", line)
        if m:
            if para or bullets:
                flush()
            numbered.append(m.group(1))
            continue
        if bullets or numbered:
            flush()
        para.append(line.strip())
    flush()
    return out


def build_pdf(path: str | Path, *, title: str, subtitle: str, topic: str, exec_summary: str, takeaways: list[str],
              sections: list[dict], charts: dict, tables: dict, references: list[dict], appendix_tables: list[dict],
              meta: dict) -> str:
    _register_fonts()
    st = _styles()
    doc = _Doc(str(path), short_title=title, title=title, author="Deep Research Agent", subject=topic)
    W = doc.width
    story: list = []

    # cover -----------------------------------------------------------
    story += [Spacer(1, 5.5 * cm), Paragraph(escape(title), st["title"]), Spacer(1, 10), Paragraph(escape(subtitle), st["subtitle"]),
              Spacer(1, 1.3 * cm)]
    meta_rows = [[Paragraph(f"<b>{escape(k)}</b>", st["meta"]), Paragraph(escape(str(v)), st["meta"])] for k, v in meta.items()]
    if meta_rows:
        t = Table(meta_rows, colWidths=[4.2 * cm, W - 4.2 * cm])
        t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1, ACCENT), ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                               ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(t)
    story += [Spacer(1, 1.2 * cm), Paragraph("Generated by a multi-agent AI research pipeline. Findings should be verified against the cited "
                                             "primary sources before being relied upon.", st["meta"]), NextPageTemplate("body"), PageBreak()]

    # TOC -------------------------------------------------------------
    toc = TableOfContents()
    toc.levelStyles = [st["toc1"], st["toc2"]]
    toc.dotsMinLevel = 0
    story += [Paragraph("Contents", st["h1x"]), toc, PageBreak()]

    # executive summary ----------------------------------------------
    story.append(Paragraph("Executive Summary", st["h1"]))
    for para in [p for p in exec_summary.split("\n\n") if p.strip()]:
        story.append(Paragraph(md_inline(para.strip()), st["body"]))
    if takeaways:
        items = [Paragraph("<b>Key takeaways</b>", st["box"]), Spacer(1, 4)] + [
            Paragraph("\u2022 " + md_inline(t), st["box"]) for t in takeaways]
        box = Table([[items]], colWidths=[W])
        box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("LINEBEFORE", (0, 0), (0, -1), 3, NAVY),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                 ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        story += [Spacer(1, 6), box]
    story.append(PageBreak())

    # body sections ---------------------------------------------------
    used: set = set()
    for i, sec in enumerate(sections, 1):
        story.append(Paragraph(f"{i}. {escape(sec['title'])}", st["h1"]))
        story += _blocks(sec["content"], st, W, charts, tables, used)
        story.append(Spacer(1, 10))

    # leftover charts / tables not placed by the writer ---------------
    leftovers = [c for c in charts.values() if c.id not in used]
    if leftovers:
        story.append(Paragraph("Additional Figures", st["h2"]))
        for c in leftovers:
            story += _blocks("{{FIGURE:%s}}" % c.id, st, W, charts, tables, used)

    # references ------------------------------------------------------
    story += [PageBreak(), Paragraph("References", st["h1"])]
    for ref in references:
        link = escape(ref["url"], {'"': "&quot;"})
        authors = f"{escape(ref['authors'])}. " if ref.get("authors") else ""
        year = f"({escape(ref['year'])}). " if ref.get("year") else ""
        story.append(Paragraph(f"[{ref['n']}] {authors}{year}<b>{escape(ref['title'])}</b>. {escape(ref['origin'])}. "
                               f'<link href="{link}" color="#1F4E79">{escape(ref["url"][:110])}</link>', st["ref"]))

    # appendices ------------------------------------------------------
    for k, ap in enumerate(appendix_tables):
        title, _, note = ap["title"].partition(" (")
        story += [Spacer(1, 12), Paragraph(f"Appendix {chr(65 + k)}. {escape(title)}", st["h2"])]
        if note:
            story.append(Paragraph(escape("(" + note), st["caption"]))
        story.append(_table(ap["columns"], ap["rows"], st, W))

    doc.multiBuild(story)
    return str(path)
