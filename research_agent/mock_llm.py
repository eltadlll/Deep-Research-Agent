"""Deterministic fake LLM used for offline demos and tests (LLM_PROVIDER=mock)."""
from __future__ import annotations

import json
import re


def _topic(prompt: str) -> str:
    m = re.search(r"^TOPIC:\s*(.+)$", prompt, re.M)
    return m.group(1).strip() if m else "the topic"


def mock_response(task: str, prompt: str) -> str:
    t = _topic(prompt)
    if task == "planner":
        return json.dumps({
            "title": f"{t.title()}: A Research Report",
            "scope": f"Evidence review of {t}.",
            "research_questions": [
                f"What is the current state of {t}?",
                f"What are the main drivers and barriers for {t}?",
                f"What does quantitative data say about trends in {t}?",
            ],
            "web_queries": [t, f"{t} overview statistics"],
            "academic_queries": [t],
            "dataset_queries": [t],
            "worldbank_indicators": [{"code": "SP.POP.TOTL", "name": "Population, total"}],
            "countries": ["USA", "CHN", "IND", "DEU", "BRA", "ETH"],
            "outline": [
                {"section": "Current State and Background", "goal": "Describe where things stand."},
                {"section": "Drivers and Barriers", "goal": "Explain what pushes and holds back progress."},
                {"section": "Quantitative Trends", "goal": "Summarise what the data show."},
            ],
            "keywords": t.split()[:5],
        })
    if task == "curator":
        ids = [int(i) for i in re.findall(r"\[id=(\d+)\]", prompt)]
        return json.dumps({"scores": [{"id": i, "score": 6 + (i % 4), "reason": "mock relevance"} for i in ids]})
    if task == "summarize":
        return json.dumps({
            "summary": "This source discusses the topic in some depth and reports several measurable trends.",
            "key_points": ["Adoption is growing steadily.", "Costs are falling.", "Policy support matters."],
            "key_numbers": [{"claim": "Example growth figure", "value": "12% per year"}],
            "relevance": 7,
            "quality_notes": "mock",
        })
    if task == "synthesize":
        qs = re.findall(r"^Q(\d+):\s*(.+)$", prompt, re.M)
        ids = [int(i) for i in re.findall(r"^\[(\d+)\]", prompt, re.M)] or [1]
        findings = [{
            "question": q, "answer": f"Evidence indicates a consistent picture for: {q}",
            "evidence_ids": ids[:3], "confidence": ["high", "medium", "low"][int(n) % 3],
            "caveats": "Based on a limited sample of sources.",
        } for n, q in qs]
        return json.dumps({"findings": findings, "agreements": ["Sources agree on the overall direction."],
                           "contradictions": [], "gaps": ["More long-term data would help."], "themes": ["growth", "cost"]})
    if task == "analyst":
        num = re.search(r"NUMERIC:\s*(.*)", prompt)
        cat = re.search(r"CATEGORICAL:\s*(.*)", prompt)
        tim = re.search(r"TIME:\s*(.*)", prompt)
        nums = [c.strip() for c in (num.group(1).split(",") if num else []) if c.strip()]
        cats = [c.strip() for c in (cat.group(1).split(",") if cat else []) if c.strip()]
        times = [c.strip() for c in (tim.group(1).split(",") if tim else []) if c.strip()]
        specs = []
        if times and nums:
            specs.append({"type": "trend", "x": times[0], "y": nums[0], "group": cats[0] if cats else None,
                          "title": f"{nums[0]} over time", "why": "temporal trend"})
        if nums:
            specs.append({"type": "distribution", "column": nums[0], "title": f"Distribution of {nums[0]}", "why": "spread"})
        if cats and nums:
            specs.append({"type": "groupby_bar", "category": cats[0], "value": nums[0], "agg": "mean", "top": 10,
                          "title": f"Mean {nums[0]} by {cats[0]}", "why": "comparison"})
        specs.append({"type": "summary_table", "title": "Summary statistics", "why": "overview"})
        return json.dumps({"description": "Mock dataset description.", "analyses": specs})
    if task == "caption":
        ids = re.findall(r"^CHART (\w+):", prompt, re.M)
        return json.dumps({"captions": {i: f"Chart {i}: illustrates a key pattern in the data." for i in ids}})
    if task == "critic_evidence":
        if "ROUND: 1" in prompt:
            return json.dumps({"verdict": "needs_more", "gaps": ["Need more recent evidence."],
                               "followup_queries": [f"{t} latest developments"], "followup_academic": [f"{t} review"]})
        return json.dumps({"verdict": "enough", "gaps": [], "followup_queries": [], "followup_academic": []})
    if task == "critic_draft":
        return json.dumps({"overall_quality": 8, "issues": []})
    if task == "writer":
        ids = re.search(r"SOURCE_IDS:\s*([\d, ]+)", prompt)
        idl = [i.strip() for i in ids.group(1).split(",") if i.strip()] if ids else []
        figs = re.search(r"FIG_IDS:\s*([\w, ]+)", prompt)
        fl = [i.strip() for i in figs.group(1).split(",") if i.strip()] if figs else []
        c1 = f"[{idl[0]}]" if idl else ""
        c2 = f"[{idl[1]}, {idl[-1]}]" if len(idl) > 1 else c1
        body = (
            f"The evidence reviewed points to a clear and consistent picture {c1}. "
            f"Several independent sources report steady progress alongside persistent constraints {c2}.\n\n"
            f"- Growth continues across most indicators {c1}\n- Cost and policy are the main levers {c2}\n\n"
            f"A second paragraph elaborates on nuance, trade-offs and uncertainty in the underlying data {c1}."
        )
        if fl and "Data Analysis" in prompt:
            body += "\n\n" + "\n\n".join("{{FIGURE:%s}}" % f for f in fl[:2])
        return body
    if task == "editor":
        return json.dumps({
            "title": f"{t.title()}: Evidence, Data and Outlook",
            "subtitle": "An automated multi-agent research report",
            "executive_summary": "This report synthesises web sources, academic papers and datasets. Findings point to steady growth with notable constraints, and it highlights where further evidence is needed.",
            "key_takeaways": ["Growth is steady across indicators.", "Costs and policy are the main levers.", "Data coverage remains uneven."],
        })
    return "{}"
