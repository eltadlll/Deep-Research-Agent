from __future__ import annotations

from collections import Counter

from ..state import ChartInfo
from ..tools import charts as C
from .base import BaseAgent


class VisualizationAgent(BaseAgent):
    name = "Visualizer"
    role = "You turn analysis results into clear, honest charts with concise captions."

    def _render(self, req: dict, path: str) -> str | None:
        k = req["kind"]
        if k == "line":
            return C.line_chart(req["df"], req["x"], req["y"], req.get("group"), req["title"], path)
        if k == "bar":
            return C.bar_chart(req["labels"], req["values"], req["title"], path, xlabel=req.get("xlabel", ""))
        if k == "hist":
            return C.hist_chart(req["series"], req["title"], path, xlabel=req.get("xlabel", ""))
        if k == "scatter":
            return C.scatter_chart(req["df"], req["x"], req["y"], req["title"], path)
        if k == "heatmap":
            return C.heatmap_chart(req["corr"], req["title"], path)
        return None

    def _meta_requests(self) -> list[dict]:
        s = self.state
        reqs = []
        docs = [x for x in s.sources if x.selected and x.status in ("summarized", "read", "abstract_only", "downloaded")]
        if len(docs) >= 3:
            kinds = Counter({"dataset": "Dataset", "paper": "Academic paper", "wiki": "Encyclopedia", "web": "Web page / report"}.get(x.kind, x.kind) for x in docs)
            reqs.append({"kind": "bar", "labels": list(kinds.keys()), "values": list(kinds.values()), "title": "Evidence base by source type",
                         "xlabel": "number of sources", "dataset": "", "analysis_id": "meta", "meta_note": "Composition of the retrieved evidence."})
        years = Counter(int(x.year) for x in docs if x.year.isdigit() and 1990 <= int(x.year) <= 2100)
        if sum(years.values()) >= 4:
            ys = sorted(years)
            reqs.append({"kind": "bar", "labels": [str(y) for y in ys], "values": [years[y] for y in ys], "title": "Publication year of sources",
                         "xlabel": "number of sources", "dataset": "", "analysis_id": "meta", "meta_note": "Recency of the evidence base."})
        return reqs

    def run(self, requests: list[dict]) -> None:
        s = self.state
        all_reqs = requests + self._meta_requests()
        for i, req in enumerate(all_reqs, 1):
            cid = f"fig{i}"
            path = str(s.run_dir / "charts" / f"{cid}.png")
            try:
                if self._render(req, path):
                    s.charts.append(ChartInfo(id=cid, path=path, title=req["title"], dataset=req.get("dataset", "")))
                    for a in s.analyses:
                        if a["id"] == req.get("analysis_id"):
                            a["chart_id"] = cid
            except Exception as e:  # noqa: BLE001
                self.log(f"chart '{req.get('title')}' failed: {str(e)[:100]}")
        # captions in a single LLM call
        if s.charts:
            lines = []
            for c in s.charts:
                a = next((a for a in s.analyses if a.get("chart_id") == c.id), None)
                facts = " ".join(a["insights"][:2]) if a else "Descriptive chart of the evidence base."
                lines.append(f"CHART {c.id}: {c.title} | FACTS: {facts[:400]}")
            prompt = ("Write a one-sentence, factual caption (max 28 words) for each chart, using only the given facts.\n"
                      + "\n".join(lines) + '\nReturn JSON: {"captions": {"<chart id>": "<caption>"}}')
            res = self.llm.complete_json(prompt, system=self.system, task="caption", default={})
            caps = res.get("captions", {}) if isinstance(res, dict) else {}
            for c in s.charts:
                c.caption = str(caps.get(c.id) or c.title)
        self.log(f"rendered {len(s.charts)} charts")
