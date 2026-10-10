"""Orchestrator: wires the 13 agents into an iterative research pipeline."""
from __future__ import annotations

import datetime as dt
import math
import time
import traceback
from pathlib import Path
from typing import Callable

from .agents import (AnalystAgent, CriticAgent, CuratorAgent, DownloaderAgent, EditorAgent, PlannerAgent, ReaderAgent,
                     ReportBuilderAgent, SearchAgent, SummarizerAgent, SynthesisAgent, VisualizationAgent, WriterAgent)
from .config import Config
from .llm import LLM
from .state import ResearchState, slugify
from .tools import get_toolkit


class ResearchOrchestrator:
    def __init__(self, topic: str, cfg: Config | None = None, on_event: Callable[[dict], None] | None = None,
                 llm: LLM | None = None, toolkit=None):
        self.cfg = cfg or Config()
        self.topic = topic.strip()
        if not self.topic:
            raise ValueError("topic must not be empty")
        run_dir = Path(self.cfg.runs_dir) / f"{dt.datetime.now():%Y%m%d-%H%M%S}-{slugify(self.topic)}"
        self.state = ResearchState(self.topic, run_dir, on_event=on_event, verbose=self.cfg.verbose)
        self.llm = llm or LLM(self.cfg)
        self.tk = toolkit or get_toolkit(self.cfg)
        a = lambda cls: cls(self.llm, self.cfg, self.state, self.tk)  # noqa: E731
        self.planner, self.searcher, self.curator = a(PlannerAgent), a(SearchAgent), a(CuratorAgent)
        self.downloader, self.reader, self.summarizer = a(DownloaderAgent), a(ReaderAgent), a(SummarizerAgent)
        self.synth, self.analyst, self.viz = a(SynthesisAgent), a(AnalystAgent), a(VisualizationAgent)
        self.critic, self.writer, self.editor, self.builder = a(CriticAgent), a(WriterAgent), a(EditorAgent), a(ReportBuilderAgent)

    # ------------------------------------------------------------------
    def _gather(self, web_q: list[str], acad_q: list[str], data_q: list[str], doc_budget: int, include_datasets: bool = True) -> None:
        s = self.state
        s.set_stage("search")
        cands = self.searcher.run(web_q, acad_q, data_q, include_datasets=include_datasets)
        s.set_stage("curate")
        n_data = max(0, self.cfg.max_datasets - len(s.datasets)) if include_datasets else 0
        picked = self.curator.run(cands, n_docs=doc_budget, n_datasets=n_data)
        s.set_stage("download")
        self.downloader.run(picked)
        s.set_stage("read")
        self.reader.run(picked)
        s.set_stage("summarize")
        self.summarizer.run(picked)
        s.save()

    def run(self) -> Path:
        s, cfg = self.state, self.cfg
        t0 = time.time()
        s.log("Orchestrator", f"topic: {self.topic} | depth={cfg.depth} | llm={cfg.llm_provider}"
                              f"{'/' + cfg.gemini_model if cfg.llm_provider == 'gemini' else ''} | out={s.run_dir}")
        try:
            s.set_stage("plan")
            plan = self.planner.run()
            s.save()

            rounds = 0
            web_q, acad_q, data_q = plan["web_queries"], plan["academic_queries"], plan["dataset_queries"]
            include_ds = True
            if plan.get("worldbank_indicators"):
                s.set_stage("worldbank")
                self.downloader.fetch_worldbank()
            budget = cfg.max_sources
            for r in range(1, cfg.research_rounds + 1):
                rounds = r
                self._gather(web_q, acad_q, data_q, doc_budget=budget, include_datasets=include_ds)
                if r == cfg.research_rounds:
                    break
                s.set_stage("review-evidence")
                self.synth.run()
                verdict = self.critic.review_evidence(r)
                if verdict.get("verdict") != "needs_more" or not (verdict["followup_queries"] or verdict["followup_academic"]):
                    break
                web_q, acad_q, data_q = verdict["followup_queries"][:3], verdict["followup_academic"][:2], []
                include_ds = False
                budget = max(2, math.ceil(cfg.max_sources / 3))
            s.stats["rounds"] = rounds

            s.set_stage("synthesize")
            self.synth.run()
            s.set_stage("analyze-data")
            chart_reqs = self.analyst.run() if s.datasets else []
            s.set_stage("visualize")
            self.viz.run(chart_reqs)
            s.save()

            s.set_stage("write")
            self.writer.run()
            s.set_stage("review-draft")
            review = self.critic.review_draft()
            if review.get("issues"):
                self.writer.revise(review["issues"])
            s.set_stage("edit")
            cited = self.editor.run()
            s.set_stage("build-pdf")
            out = self.builder.run(cited)
            s.stats.update(seconds=round(time.time() - t0), llm_calls=self.llm.calls, llm_failures=self.llm.failures,
                           sources_found=len(s.sources), sources_cited=len(cited))
            s.set_stage("done")
            s.save()
            s.log("Orchestrator", f"finished in {s.stats['seconds']}s, {self.llm.calls} LLM calls -> {out}")
            return out
        except Exception as e:  # noqa: BLE001
            s.log("Orchestrator", f"FAILED: {e}")
            s.stats["error"] = f"{e}\n{traceback.format_exc()[-1500:]}"
            s.save()
            raise
