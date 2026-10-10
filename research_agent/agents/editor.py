from __future__ import annotations

import re

from .base import BaseAgent

CIT = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class EditorAgent(BaseAgent):
    name = "Editor"
    role = "You write the executive summary, key takeaways and title, and make the report consistent."

    def _summary(self) -> None:
        s = self.state
        body = "\n\n".join(f"## {x['title']}\n{x['content'][:1100]}" for x in s.sections)
        prompt = f"""TOPIC: {s.topic}
DRAFT REPORT (condensed):
{body[:14000]}

Write: a professional title, a one-line subtitle, an executive summary (170-230 words, 2 paragraphs, keep any [id] citations that appear in the
draft, add no new facts) and 4-6 key takeaways (<= 25 words each, may keep [id] citations).
Return JSON: {{"title": "...", "subtitle": "...", "executive_summary": "...", "key_takeaways": ["..."]}}"""
        res = self.llm.complete_json(prompt, system=self.system, task="editor", default={})
        if not isinstance(res, dict):
            res = {}
        s.title = str(res.get("title") or s.title or f"Research Report: {s.topic}")
        s.subtitle = str(res.get("subtitle") or "Multi-agent research report")
        s.exec_summary = str(res.get("executive_summary") or (s.sections[-1]["content"][:900] if s.sections else ""))
        s.takeaways = [str(x) for x in res.get("key_takeaways", [])][:6]

    def _renumber(self) -> list[int]:
        """Renumber citations by order of first appearance; drop invalid ids. Returns ordered source ids actually cited."""
        s = self.state
        valid = {x.id for x in s.sources}
        order: list[int] = []

        def scan(text: str) -> None:
            for m in CIT.findall(text):
                for n in re.split(r"\s*,\s*", m):
                    n = int(n)
                    if n in valid and n not in order:
                        order.append(n)

        scan(s.exec_summary)
        for t in s.takeaways:
            scan(t)
        for sec in s.sections:
            scan(sec["content"])
        mapping = {old: new for new, old in enumerate(order, 1)}

        def sub(text: str) -> str:
            def rep(m):
                ids = sorted({mapping[int(n)] for n in re.split(r"\s*,\s*", m.group(1)) if int(n) in mapping})
                return "[" + ", ".join(map(str, ids)) + "]" if ids else ""

            out = CIT.sub(rep, text)
            return re.sub(r"[ ]+([.,;:])", r"\1", re.sub(r" {2,}", " ", out))

        s.exec_summary = sub(s.exec_summary)
        s.takeaways = [sub(t) for t in s.takeaways]
        for sec in s.sections:
            sec["content"] = sub(sec["content"])
        s.stats["citation_map"] = {str(old): new for old, new in mapping.items()}
        return order

    def run(self) -> list[int]:
        self._summary()
        order = self._renumber()
        self.log(f"title: '{self.state.title}'; {len(order)} distinct sources cited")
        return order
