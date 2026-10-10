from __future__ import annotations

from collections import Counter

from ..state import Source
from .base import BaseAgent

GOOD = ("arxiv.org", "nature.com", "science.org", "sciencedirect.com", "springer.com", "ieee.org", "acm.org", "who.int",
        "worldbank.org", "oecd.org", "imf.org", "un.org", "europa.eu", "nih.gov", "ncbi.nlm.nih.gov", "wikipedia.org",
        "mckinsey.com", "iea.org", "irena.org", "ourworldindata.org", "data.gov", "zenodo.org", "openalex.org",
        "brookings.edu", "pewresearch.org", "bloomberg.com", "reuters.com", "ft.com", "economist.com", "plos.org")
BAD = ("pinterest.", "facebook.", "instagram.", "tiktok.", "quora.", "reddit.", "youtube.", "twitter.", "x.com", "medium.com/@",
       "linkedin.", "scribd.", "slideshare.", "coursehero.")


def heuristic_score(s: Source) -> float:
    d = s.domain.lower()
    score = 5.0
    if d.endswith((".gov", ".edu", ".ac.uk", ".int")) or ".gov." in d or ".edu." in d or ".ac." in d:
        score += 2.5
    if any(g in d for g in GOOD):
        score += 2.0
    if any(b in d or b in s.url.lower() for b in BAD):
        score -= 4.0
    if s.kind == "paper":
        score += 1.0
    if s.pdf_url:
        score += 0.5
    if s.year.isdigit():
        y = int(s.year)
        score += 0.6 if y >= 2020 else (0.2 if y >= 2015 else -0.4)
    return max(0.0, min(10.0, score))


class CuratorAgent(BaseAgent):
    name = "SourceCurator"
    role = "You judge candidate sources for relevance and credibility and pick the best, diverse set."

    def _score_batch(self, batch: list[Source]) -> None:
        lines = "\n".join(
            f"[id={s.id}] ({s.kind}, {s.domain}, {s.year or 'n.d.'}) {s.title[:140]} :: {s.snippet[:220]}" for s in batch)
        prompt = f"""TOPIC: {self.state.topic}
Research questions: {'; '.join(self.state.plan.get('research_questions', [])[:6])}

Score each candidate 0-10 for how useful and credible it would be for a deep research report on the topic
(10 = authoritative and directly relevant; 0 = irrelevant / spam / marketing). Datasets should score high only if they
plausibly contain quantitative data on the topic.

CANDIDATES:
{lines}

Return JSON: {{"scores": [{{"id": <int>, "score": <0-10>, "reason": "<8 words max>"}}]}}"""
        res = self.llm.complete_json(prompt, system=self.system, task="curator", default={"scores": []})
        by_id = {s.id: s for s in batch}
        for item in (res.get("scores", []) if isinstance(res, dict) else []):
            try:
                s = by_id.get(int(item.get("id")))
                if s:
                    s.llm_score = float(item.get("score", 5))
                    s.reason = str(item.get("reason", ""))[:100]
            except (TypeError, ValueError):
                continue

    def run(self, candidates: list[Source], n_docs: int, n_datasets: int) -> list[Source]:
        cands = [s for s in candidates if s.status == "found" and not s.selected]
        for s in cands:
            s.heuristic = heuristic_score(s)
        # pre-filter obviously bad ones and cap LLM load
        pool = [s for s in cands if s.heuristic >= 2.0]
        pool.sort(key=lambda s: s.heuristic, reverse=True)
        pool = pool[:45]
        for i in range(0, len(pool), 15):
            self._score_batch(pool[i : i + 15])
        for s in pool:
            llm = s.llm_score if s.llm_score else s.heuristic
            s.score = round(0.7 * llm + 0.3 * s.heuristic, 2)

        docs = sorted([s for s in pool if s.kind != "dataset"], key=lambda s: s.score, reverse=True)
        data = sorted([s for s in pool if s.kind == "dataset"], key=lambda s: s.score, reverse=True)

        chosen: list[Source] = []
        per_domain: Counter = Counter()
        for s in docs:
            if len(chosen) >= n_docs:
                break
            if s.score < 4.5 or per_domain[s.domain] >= 3:
                continue
            chosen.append(s)
            per_domain[s.domain] += 1
        # make sure academic papers are represented
        if sum(1 for s in chosen if s.kind == "paper") < min(3, n_docs // 3):
            extra = [s for s in docs if s.kind == "paper" and s not in chosen and s.score >= 4.5]
            for s in extra[: min(3, n_docs // 3) - sum(1 for c in chosen if c.kind == "paper")]:
                if len(chosen) >= n_docs and chosen:
                    # replace the weakest non-paper
                    weakest = min((c for c in chosen if c.kind != "paper"), key=lambda c: c.score, default=None)
                    if weakest:
                        chosen.remove(weakest)
                        weakest.selected = False
                chosen.append(s)
        chosen_data = [s for s in data if s.score >= 4.5][: max(0, n_datasets)]
        for s in chosen + chosen_data:
            s.selected = True
        self.log(f"scored {len(pool)} candidates -> selected {len(chosen)} documents + {len(chosen_data)} datasets")
        return chosen + chosen_data
