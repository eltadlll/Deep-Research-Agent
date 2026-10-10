from __future__ import annotations

import datetime as dt
import re

from .base import BaseAgent
from .synthesizer import usable_sources


class WriterAgent(BaseAgent):
    name = "Writer"
    role = "You write clear, rigorous, well-structured analytical prose for expert readers, citing sources as [id]."

    # -- context -----------------------------------------------------------
    def _context(self) -> dict:
        s = self.state
        srcs = usable_sources(s)
        src_block = "\n".join(
            f"[{x.id}] {x.title[:100]} ({x.year or 'n.d.'}, {x.domain}): {x.summary[:380]} "
            + ("KEY NUMBERS: " + "; ".join(f"{n.get('claim')}: {n.get('value')}" for n in x.key_numbers[:3]) if x.key_numbers else "")
            for x in srcs)
        f = s.findings
        find_block = "\n".join(
            f"* Q: {x.get('question')}\n  A: {x.get('answer')} [evidence: {', '.join(map(str, x.get('evidence_ids', [])))}; confidence: {x.get('confidence')}; caveats: {x.get('caveats', '')[:160]}]"
            for x in f.get("findings", []))
        extra = ""
        if f.get("contradictions"):
            extra += "\nCONTRADICTIONS: " + " | ".join(f"{c.get('issue')} (sources {c.get('sources')}): {c.get('note')}" for c in f["contradictions"][:4])
        if f.get("gaps"):
            extra += "\nGAPS: " + " | ".join(map(str, f["gaps"][:5]))
        ana = []
        for a in s.analyses:
            refs = ([f"FIGURE id {a['chart_id']}"] if a.get("chart_id") else []) + ([f"TABLE id {a['table_id']}"] if a.get("table_id") else [])
            ana.append(f"({a['id']}) dataset '{a['dataset']}' [data source id {a['source_id']}] - {a['title']} "
                       f"[{', '.join(refs) or 'no figure'}]: " + " ".join(a["insights"][:3]))
        return {"srcs": srcs, "src_block": src_block, "find_block": find_block + extra, "ana_block": "\n".join(ana)}

    def _prompt(self, title: str, goal: str, ctx: dict, words: str, facts: str = "", prior: str = "") -> str:
        s = self.state
        src_ids = ", ".join(str(x.id) for x in ctx["srcs"]) or "none"
        data_ids = ", ".join(str(d["source_id"]) for d in s.datasets)
        figs = ", ".join(c.id for c in s.charts)
        tabs = ", ".join(t.id for t in s.tables)
        fig_list = "\n".join(f"{c.id}: {c.caption or c.title}" for c in s.charts)
        tab_list = "\n".join(f"{t.id}: {t.title}" for t in s.tables)
        return f"""TOPIC: {s.topic}
SECTION TO WRITE: "{title}" - purpose: {goal}
Target length: {words} words. Sections already written: {prior or 'none'}.

SOURCE_IDS: {src_ids}
DATASET_SOURCE_IDS: {data_ids or 'none'}
FIG_IDS: {figs or 'none'}
TABLE_IDS: {tabs or 'none'}

RESEARCH FINDINGS:
{ctx['find_block'] or 'n/a'}

SOURCES:
{ctx['src_block'] or 'n/a'}

DATA ANALYSIS RESULTS (computed by code - these numbers are reliable):
{ctx['ana_block'] or 'n/a'}

AVAILABLE FIGURES:
{fig_list or 'none'}
AVAILABLE TABLES:
{tab_list or 'none'}
{('FACTS: ' + facts) if facts else ''}

RULES:
- Write in markdown: paragraphs, optional "### " sub-headings, "- " bullets, **bold** for key terms. Do NOT repeat the section title as a heading.
- Cite with numeric brackets like [3] or [2, 5]. Cite ONLY ids listed in SOURCE_IDS or DATASET_SOURCE_IDS. Every factual claim needs a citation.
- Use numbers ONLY if they appear in the sources/analysis above. Never invent statistics, quotes, or sources.
- Be balanced: state confidence, caveats and disagreements. Separate evidence from interpretation.
- To place a figure or table, put `{{{{FIGURE:<id>}}}}` or `{{{{TABLE:<id>}}}}` on its own line right after the paragraph that discusses it, and refer to it in text as "Figure N"/"Table N" (N = digits of the id). Use each id at most once overall.
Write the section now (body text only)."""

    @staticmethod
    def _clean(text: str, title: str) -> str:
        text = text.strip()
        text = re.sub(r"^```\w*\s*|\s*```$", "", text)
        lines = text.splitlines()
        while lines and (re.match(r"^#{1,3}\s", lines[0]) or lines[0].strip().strip("*").lower() == title.lower()):
            lines.pop(0)
        return "\n".join(lines).strip()

    # -- main ----------------------------------------------------------------
    def _section_specs(self) -> list[tuple[str, str, str, str]]:
        s = self.state
        a = s.stats
        specs = [("Introduction", "Frame the topic, why it matters, the scope, and the questions the report answers.", "250-350", ""),
                 ("Methodology and Evidence Base", "Explain transparently how evidence was gathered and analysed, and its limits.", "200-300",
                  f"Queries run: {len(s.queries_run)}; candidate sources found: {len(s.sources)}; documents read: "
                  f"{sum(1 for x in s.sources if x.status == 'summarized' and x.kind != 'dataset')}; datasets analysed: "
                  f"{sum(1 for d in s.datasets if d.get('rows'))}; charts produced: {len(s.charts)}; "
                  f"search engines/APIs: DuckDuckGo, arXiv, OpenAlex, Wikipedia, data.gov, Zenodo, World Bank; analysis run in Python/pandas; "
                  f"date: {dt.date.today():%d %B %Y}; research rounds: {a.get('rounds', 1)}.")]
        for o in s.plan.get("outline", []):
            if isinstance(o, dict) and o.get("section"):
                specs.append((str(o["section"]), str(o.get("goal", "")), "350-500", ""))
        if s.analyses:
            specs.append(("Data Analysis", "Present and interpret the quantitative analysis of the downloaded datasets, referencing the figures and tables.", "400-550",
                          "Place the figures and tables here."))
        specs += [("Discussion and Limitations", "Synthesize across sources, address contradictions, gaps, biases, and reliability.", "300-450", ""),
                  ("Conclusions and Recommendations", "Give clear conclusions and prioritized, actionable recommendations grounded in the evidence.", "250-400", "")]
        return specs

    def run(self) -> None:
        s = self.state
        ctx = self._context()
        s.sections = []
        for title, goal, words, facts in self._section_specs():
            prior = ", ".join(x["title"] for x in s.sections)
            try:
                text = self.llm.complete(self._prompt(title, goal, ctx, words, facts, prior), system=self.system, temperature=0.4, task="writer")
            except Exception as e:  # noqa: BLE001
                text = f"_This section could not be generated automatically ({str(e)[:80]})._"
            s.sections.append({"title": title, "content": self._clean(text, title)})
            self.log(f"wrote '{title}' ({len(s.sections[-1]['content'].split())} words)")

    def revise(self, issues: list[dict]) -> int:
        s = self.state
        ctx = self._context()
        by_sec: dict[str, list[dict]] = {}
        for i in issues:
            by_sec.setdefault(i["section"], []).append(i)
        n = 0
        for sec in s.sections:
            if sec["title"] not in by_sec:
                continue
            probs = "\n".join(f"- {i['problem']} Fix: {i.get('fix', '')}" for i in by_sec[sec["title"]])
            prompt = f"""TOPIC: {s.topic}
Revise the report section "{sec['title']}" to resolve the reviewer's issues. Keep what is good, keep {{{{FIGURE:..}}}}/{{{{TABLE:..}}}} lines unchanged,
keep citations as [id] using only these ids: {', '.join(str(x.id) for x in ctx['srcs'])}. Remove or soften claims the evidence does not support.

ISSUES:
{probs}

EVIDENCE:
{ctx['src_block'][:9000]}

CURRENT TEXT:
{sec['content']}

Return the revised section body only (markdown, no title heading)."""
            try:
                new = self._clean(self.llm.complete(prompt, system=self.system, temperature=0.3, task="writer"), sec["title"])
                if len(new.split()) > 40:
                    sec["content"] = new
                    n += 1
            except Exception:  # noqa: BLE001
                continue
        self.log(f"revised {n} sections")
        return n
