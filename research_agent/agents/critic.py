from __future__ import annotations

import re

from .base import BaseAgent
from .synthesizer import usable_sources

CIT = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class CriticAgent(BaseAgent):
    name = "Critic"
    role = "You are a skeptical reviewer: you check evidence sufficiency, unsupported claims, bias, and missing perspectives."

    def review_evidence(self, round_no: int) -> dict:
        s = self.state
        srcs = usable_sources(s)
        f = s.findings
        findings = "\n".join(f"- {x.get('question')} -> confidence {x.get('confidence')}, {len(x.get('evidence_ids', []))} sources. Caveats: {x.get('caveats', '')[:150]}"
                             for x in f.get("findings", []))
        prompt = f"""TOPIC: {s.topic}
ROUND: {round_no}
Usable sources: {len(srcs)} ({sum(1 for x in srcs if x.kind == 'paper')} papers). Datasets analysable: {len(s.datasets)}.
FINDINGS:
{findings}
KNOWN GAPS: {'; '.join(map(str, f.get('gaps', [])))[:600]}
CONTRADICTIONS: {len(f.get('contradictions', []))}

Is the evidence sufficient for a credible, well-rounded report? If not, propose targeted follow-up searches that would close the
biggest gaps (max 3 web queries, max 2 academic queries). Return JSON:
{{"verdict": "enough|needs_more", "gaps": ["..."], "followup_queries": ["..."], "followup_academic": ["..."]}}"""
        res = self.llm.complete_json(prompt, system=self.system, task="critic_evidence", default={"verdict": "enough"})
        if not isinstance(res, dict):
            res = {"verdict": "enough"}
        res.setdefault("followup_queries", [])
        res.setdefault("followup_academic", [])
        if len(srcs) < 3 and not res["followup_queries"]:
            res["verdict"], res["followup_queries"] = "needs_more", [s.topic + " overview", s.topic + " report pdf"]
        s.critique["evidence"] = res
        self.log(f"evidence verdict: {res.get('verdict')} ({len(res['followup_queries'])} follow-up queries)")
        return res

    def review_draft(self) -> dict:
        s = self.state
        valid = {x.id for x in s.sources}
        issues: list[dict] = []
        # programmatic checks first (cheap, objective)
        for sec in s.sections:
            cites = {int(n) for m in CIT.findall(sec["content"]) for n in re.split(r"\s*,\s*", m)}
            bad = cites - valid
            if len(sec["content"].split()) > 120 and not (cites & valid):
                issues.append({"section": sec["title"], "problem": "No citations in a substantive section.", "fix": "Cite supporting sources using only the provided ids."})
            if bad:
                issues.append({"section": sec["title"], "problem": f"Cites non-existent source ids {sorted(bad)}.", "fix": "Remove or replace those citations."})
        # LLM check of claims against evidence
        ev = "\n".join(f"[{x.id}] {x.summary[:260]}" for x in usable_sources(s)[:25])
        draft = "\n\n".join(f"## {sec['title']}\n{sec['content'][:1800]}" for sec in s.sections)
        prompt = f"""TOPIC: {s.topic}
EVIDENCE (source summaries):
{ev}

DRAFT REPORT:
{draft}

Review the draft as a skeptical editor. Flag ONLY real problems: claims not supported by the evidence above, overstated certainty,
numbers not present in evidence, missing caveats, or sections that don't address their purpose. Max 5 issues.
Return JSON: {{"overall_quality": <1-10>, "issues": [{{"section": "<exact section title>", "problem": "...", "fix": "..."}}]}}"""
        res = self.llm.complete_json(prompt, system=self.system, task="critic_draft", default={"issues": []})
        if not isinstance(res, dict):
            res = {"issues": []}
        titles = {sec["title"] for sec in s.sections}
        llm_issues = [i for i in res.get("issues", []) if isinstance(i, dict) and i.get("section") in titles]
        res["issues"] = issues + llm_issues
        s.critique["draft"] = res
        self.log(f"draft review: quality {res.get('overall_quality', '?')}/10, {len(res['issues'])} issues")
        return res
