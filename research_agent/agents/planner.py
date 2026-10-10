from __future__ import annotations

from .base import BaseAgent


class PlannerAgent(BaseAgent):
    name = "Planner"
    role = "You turn a user's idea into a rigorous research plan: key questions, search queries, datasets to look for, and a report outline."

    def run(self) -> dict:
        s = self.state
        prompt = f"""TOPIC: {s.topic}

Create a research plan. Return JSON with exactly these keys:
- "title": a professional report title
- "scope": 1-2 sentences defining scope
- "research_questions": 4-7 specific, answerable questions
- "web_queries": 3-5 concise web search queries (include one aimed at statistics/reports)
- "academic_queries": 2-3 queries suited for arXiv / OpenAlex (academic wording)
- "dataset_queries": 2-3 queries to find downloadable open datasets (CSV/XLSX)
- "worldbank_indicators": 0-3 objects {{"code": "<World Bank indicator code>", "name": "..."}} ONLY if the topic is country-level/economic/social/environmental and you are confident the code exists (e.g. NY.GDP.MKTP.CD, SP.POP.TOTL, EG.ELC.ACCS.ZS, EN.GHG.CO2.PC.CE.AR5); otherwise []
- "countries": up to 8 ISO3 codes relevant to the topic for World Bank data, or [] for all
- "outline": {self.cfg.max_sections - 2} to {self.cfg.max_sections} objects {{"section": "...", "goal": "..."}} for the report's core body sections
- "keywords": 5 keywords"""
        plan = self.llm.complete_json(prompt, system=self.system, task="planner", default={})
        if not isinstance(plan, dict) or not plan.get("research_questions"):
            plan = {
                "title": f"Research Report: {s.topic}", "scope": s.topic,
                "research_questions": [f"What is {s.topic}?", f"What are the key trends in {s.topic}?", f"What are the risks and opportunities of {s.topic}?"],
                "web_queries": [s.topic, f"{s.topic} statistics report"], "academic_queries": [s.topic],
                "dataset_queries": [s.topic], "worldbank_indicators": [], "countries": [],
                "outline": [{"section": "Background", "goal": "Context"}, {"section": "Key Findings", "goal": "Main evidence"}, {"section": "Implications", "goal": "So what"}],
                "keywords": s.topic.split()[:5],
            }
        for k in ("web_queries", "academic_queries", "dataset_queries", "worldbank_indicators", "countries", "outline"):
            plan.setdefault(k, [])
        plan["outline"] = plan["outline"][: self.cfg.max_sections]
        s.plan = plan
        s.title = plan.get("title", "")
        self.log(f"{len(plan['research_questions'])} research questions, {len(plan['web_queries'])} web / "
                 f"{len(plan['academic_queries'])} academic / {len(plan['dataset_queries'])} dataset queries, "
                 f"{len(plan['outline'])} outline sections")
        return plan
