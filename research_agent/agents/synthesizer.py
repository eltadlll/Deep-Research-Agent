from __future__ import annotations

from .base import BaseAgent


def usable_sources(state, min_rel: float = 4.0):
    return [s for s in state.sources if s.kind != "dataset" and s.status == "summarized" and s.relevance >= min_rel]


class SynthesisAgent(BaseAgent):
    name = "Synthesizer"
    role = "You cross-read all source summaries, answer each research question, and note agreements, contradictions and gaps."

    def run(self) -> dict:
        s = self.state
        srcs = usable_sources(s)
        if not srcs:
            s.findings = {"findings": [], "agreements": [], "contradictions": [], "gaps": ["No usable sources were retrieved."], "themes": []}
            self.log("no usable sources to synthesise")
            return s.findings
        qs = "\n".join(f"Q{i}: {q}" for i, q in enumerate(s.plan.get("research_questions", []), 1))
        blocks = "\n".join(
            f"[{x.id}] {x.title[:110]} ({x.year or 'n.d.'}): {x.summary[:420]} KEY: {' | '.join(x.key_points[:3])[:300]} "
            f"NUMBERS: {'; '.join(str(n.get('claim', '')) + '=' + str(n.get('value', '')) for n in x.key_numbers[:3])[:200]}"
            for x in srcs)
        prompt = f"""TOPIC: {s.topic}
{qs}

SOURCE SUMMARIES:
{blocks}

For EACH question, write an evidence-based answer (3-5 sentences) using ONLY the summaries above. Cite supporting source ids.
Also list agreements across sources, contradictions (with ids), and gaps in the evidence.
Return JSON:
{{"findings": [{{"question": "...", "answer": "...", "evidence_ids": [ints], "confidence": "high|medium|low", "caveats": "..."}}],
 "agreements": ["..."], "contradictions": [{{"issue": "...", "sources": [ints], "note": "..."}}], "gaps": ["..."], "themes": ["..."]}}"""
        res = self.llm.complete_json(prompt, system=self.system, task="synthesize", default={})
        if not isinstance(res, dict):
            res = {}
        valid = {x.id for x in srcs}
        for f in res.get("findings", []):
            ids = []
            for i in f.get("evidence_ids", []):
                try:
                    if int(i) in valid:
                        ids.append(int(i))
                except (TypeError, ValueError):
                    pass
            f["evidence_ids"] = ids
        res.setdefault("findings", [])
        for k in ("agreements", "contradictions", "gaps", "themes"):
            res.setdefault(k, [])
        s.findings = res
        self.log(f"{len(res['findings'])} findings, {len(res['contradictions'])} contradictions, {len(res['gaps'])} gaps")
        return res
