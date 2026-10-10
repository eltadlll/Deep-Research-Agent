from __future__ import annotations

from ..state import Source
from .base import BaseAgent


class SummarizerAgent(BaseAgent):
    name = "Summarizer"
    role = "You read source excerpts and extract faithful summaries, key points and key numbers."

    def _excerpt(self, s: Source) -> str:
        text, k = s.text, self.cfg.max_chunks_per_source
        size = 9000
        if len(text) <= size * k:
            return text
        # first chunk + evenly spaced chunks from the rest
        parts = [text[:size]]
        step = (len(text) - size) // max(1, k - 1) if k > 1 else 0
        for i in range(1, k):
            start = size + (i - 1) * step
            parts.append("...\n" + text[start : start + size])
        return "\n".join(parts)

    def _one(self, s: Source) -> None:
        qs = "; ".join(self.state.plan.get("research_questions", [])[:6])
        prompt = f"""TOPIC: {self.state.topic}
Research questions: {qs}

SOURCE [{s.id}] "{s.title}" ({s.kind}, {s.year or 'n.d.'}, {s.domain}){' - ABSTRACT ONLY' if s.status == 'abstract_only' else ''}
EXCERPT:
{self._excerpt(s)}

Summarise ONLY what this excerpt actually says. Return JSON:
{{"summary": "3-5 sentences", "key_points": ["up to 6 specific findings"], "key_numbers": [{{"claim": "...", "value": "..."}}] (max 5; only numbers explicitly in the text),
 "relevance": <0-10 relevance to the topic>, "quality_notes": "methodology, bias, date, limitations in <= 20 words"}}"""
        res = self.llm.complete_json(prompt, system=self.system, task="summarize", default={})
        if not isinstance(res, dict) or not res.get("summary"):
            s.summary = s.snippet[:400] or s.title
            s.relevance = 4
            s.status = "summarized"
            return
        s.summary = str(res.get("summary", ""))[:900]
        s.key_points = [str(x)[:300] for x in res.get("key_points", [])][:6]
        s.key_numbers = [x for x in res.get("key_numbers", []) if isinstance(x, dict)][:5]
        try:
            s.relevance = float(res.get("relevance", 5))
        except (TypeError, ValueError):
            s.relevance = 5
        s.quality_notes = str(res.get("quality_notes", ""))[:200]
        s.status = "summarized"

    def run(self, sources: list[Source]) -> None:
        todo = [s for s in sources if s.status in ("read", "abstract_only") and s.kind != "dataset"]
        for i, s in enumerate(todo, 1):
            try:
                self._one(s)
            except Exception as e:  # noqa: BLE001
                s.summary, s.relevance, s.status = s.snippet[:400], 3, "summarized"
                self.log(f"summary fallback for [{s.id}]: {str(e)[:80]}")
        low = sum(1 for s in todo if s.relevance < 4)
        self.log(f"summarised {len(todo)} sources ({low} judged low-relevance)")
