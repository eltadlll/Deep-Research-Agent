from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from ..tools.pdf_report import build_pdf
from .base import BaseAgent


class ReportBuilderAgent(BaseAgent):
    name = "ReportBuilder"
    role = "You assemble the final PDF (and Markdown) report with figures, tables and references."

    def _refs(self, order: list[int]) -> list[dict]:
        refs = []
        for n, sid in enumerate(order, 1):
            x = self.state.get(sid)
            if x:
                refs.append({"n": n, "title": x.title, "authors": x.authors, "year": x.year, "origin": x.origin or x.domain,
                             "url": x.url or x.pdf_url or x.download_url})
        return refs

    def _appendices(self, cited: list[int]) -> list[dict]:
        s = self.state
        cmap = {int(k): v for k, v in s.stats.get("citation_map", {}).items()}
        rows = []
        for x in sorted([x for x in s.sources if x.selected], key=lambda z: -z.score):
            rows.append([str(cmap.get(x.id, "-")), x.title[:70], {"paper": "Paper", "web": "Web", "wiki": "Wiki", "dataset": "Dataset"}.get(x.kind, x.kind),
                         x.year or "-", f"{x.score:.1f}", x.status.replace("_", " ")])
        aps = [{"title": "All sources reviewed (relevance score 0-10; [#] = citation number in this report)",
                "columns": ["Ref", "Title", "Type", "Year", "Score", "Status"], "rows": rows}]
        ds = [[d.get("name", ""), f"{d.get('rows', '-'):,}" if isinstance(d.get("rows"), int) else "-", str(d.get("cols", "-")),
               d.get("coverage", "-")[:40], (d.get("description") or "")[:110]] for d in s.datasets if d.get("rows")]
        if ds:
            aps.append({"title": "Datasets analysed", "columns": ["File", "Rows", "Cols", "Coverage", "Description"], "rows": ds})
        f = s.findings.get("findings", [])
        if f:
            aps.append({"title": "Research questions and confidence",
                        "columns": ["Question", "Confidence", "Evidence (refs)"],
                        "rows": [[x.get("question", "")[:110], x.get("confidence", "-"),
                                  ", ".join(str(cmap[i]) for i in x.get("evidence_ids", []) if i in cmap) or "-"] for x in f]})
        return aps

    def _markdown(self, refs: list[dict]) -> str:
        s = self.state
        md = [f"# {s.title}", f"*{s.subtitle}*", "", "## Executive Summary", s.exec_summary, ""]
        md += ["**Key takeaways**"] + [f"- {t}" for t in s.takeaways] + [""]
        charts = {c.id: c for c in s.charts}
        tables = {t.id: t for t in s.tables}
        for i, sec in enumerate(s.sections, 1):
            md.append(f"## {i}. {sec['title']}")
            body = sec["content"]

            def fig(m):
                c = charts.get(m.group(1))
                return f"![{c.caption or c.title}](charts/{Path(c.path).name})\n*Figure: {c.caption or c.title}*" if c else ""

            def tab(m):
                t = tables.get(m.group(1))
                if not t:
                    return ""
                head = "| " + " | ".join(map(str, t.columns)) + " |\n|" + "---|" * len(t.columns)
                return head + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in t.rows) + f"\n*Table: {t.caption or t.title}*"

            body = re.sub(r"\{\{FIGURE:(\w+)\}\}", fig, body)
            body = re.sub(r"\{\{TABLE:(\w+)\}\}", tab, body)
            md += [body, ""]
        md += ["## References"] + [f"[{r['n']}] {r['authors'] + '. ' if r['authors'] else ''}{'(' + r['year'] + '). ' if r['year'] else ''}{r['title']}. {r['url']}" for r in refs]
        return "\n".join(md)

    def run(self, cited_order: list[int]) -> Path:
        s = self.state
        refs = self._refs(cited_order)
        docs = [x for x in s.sources if x.selected and x.kind != "dataset" and x.status == "summarized"]
        meta = {
            "Topic": s.topic, "Date": f"{dt.date.today():%d %B %Y}",
            "Evidence base": f"{len(docs)} documents read, {sum(1 for d in s.datasets if d.get('rows'))} datasets analysed, {len(refs)} sources cited",
            "Charts / tables": f"{len(s.charts)} / {len(s.tables)}",
            "Research depth": f"{self.cfg.depth} ({s.stats.get('rounds', 1)} round(s))",
        }
        out = s.run_dir / "report.pdf"
        build_pdf(out, title=s.title, subtitle=s.subtitle, topic=s.topic, exec_summary=s.exec_summary, takeaways=s.takeaways,
                  sections=s.sections, charts={c.id: c for c in s.charts}, tables={t.id: t for t in s.tables},
                  references=refs, appendix_tables=self._appendices(cited_order), meta=meta)
        (s.run_dir / "report.md").write_text(self._markdown(refs), encoding="utf-8")
        (s.run_dir / "sources.json").write_text(json.dumps(
            [{"ref": r["n"], **{k: r[k] for k in ("title", "authors", "year", "origin", "url")}} for r in refs], indent=2), encoding="utf-8")
        for t in s.tables:
            import csv

            with open(s.run_dir / "tables" / f"{t.id}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(t.columns)
                w.writerows(t.rows)
        self.log(f"report written: {out}")
        return out
