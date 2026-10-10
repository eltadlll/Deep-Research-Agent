from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..state import Source
from .base import BaseAgent


class SearchAgent(BaseAgent):
    name = "WebSearcher"
    role = "You find candidate web pages, PDFs, academic papers and datasets across many engines."

    def run(self, web_queries: list[str], academic_queries: list[str], dataset_queries: list[str],
            include_datasets: bool = True) -> list[Source]:
        n = self.cfg.search_results_per_query
        tk = self.tk
        jobs: list[tuple[str, callable]] = []
        for q in web_queries:
            jobs.append((f"web:{q}", lambda q=q: tk.web_search(q, n)))
            jobs.append((f"pdf:{q}", lambda q=q: tk.web_search(q, max(4, n // 2), filetype="pdf")))
        if web_queries:
            jobs.append((f"wiki:{web_queries[0]}", lambda: tk.wikipedia(web_queries[0], 2)))
        for q in academic_queries:
            jobs.append((f"arxiv:{q}", lambda q=q: tk.arxiv(q, 5)))
            jobs.append((f"openalex:{q}", lambda q=q: tk.openalex(q, 6)))
        if include_datasets:
            for q in dataset_queries:
                jobs.append((f"data.gov:{q}", lambda q=q: tk.ckan(q, 5)))
                jobs.append((f"zenodo:{q}", lambda q=q: tk.zenodo(q, 4)))
                jobs.append((f"csv:{q}", lambda q=q: tk.web_search(q, 5, filetype="csv")))
        self.state.queries_run += [j[0] for j in jobs]

        def run_job(job):
            label, fn = job
            try:
                return label, fn(), None
            except Exception as e:  # noqa: BLE001
                return label, [], str(e)[:120]

        new: list[Source] = []
        with ThreadPoolExecutor(max_workers=6) as ex:
            for label, items, err in ex.map(run_job, jobs):
                if err:
                    self.log(f"{label} failed: {err}")
                    continue
                added = 0
                for it in items:
                    src = self.state.add_source(**it)
                    if src:
                        new.append(src)
                        added += 1
                self.log(f"{label} -> {len(items)} results, {added} new")
        self.log(f"{len(new)} new candidate sources ({sum(1 for s in new if s.kind == 'dataset')} datasets)")
        return new
