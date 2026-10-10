"""Offline tests: run with  python -m pytest -q   (or python tests/test_pipeline.py)."""
import http.server
import shutil
import sys
import threading
import tempfile
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from research_agent.agents.curator import heuristic_score  # noqa: E402
from research_agent.config import Config  # noqa: E402
from research_agent.llm import parse_json_loose  # noqa: E402
from research_agent.orchestrator import ResearchOrchestrator  # noqa: E402
from research_agent.state import Source  # noqa: E402
from research_agent.tools.data_tools import clean_df, load_table, profile_df  # noqa: E402
from research_agent.tools.doc_tools import extract_pdf_text, html_to_text  # noqa: E402
from research_agent.tools.toolkit import Toolkit  # noqa: E402


def test_json_repair():
    assert parse_json_loose('```json\n{"a": 1,}\n```') == {"a": 1}
    assert parse_json_loose('Sure! Here: [1, 2, 3] done') == [1, 2, 3]


def test_heuristics():
    good = Source(id=1, title="x", url="https://www.who.int/report.pdf", kind="paper", year="2023", pdf_url="x")
    bad = Source(id=2, title="x", url="https://www.pinterest.com/pin/1")
    assert heuristic_score(good) > heuristic_score(bad)


def test_data_tools(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"year": range(2000, 2020), "country": ["A", "B"] * 10, "gdp": [f"{1000 + i * 10:,}" for i in range(20)]}).to_csv(p, index=False)
    df = clean_df(load_table(str(p)))
    prof = profile_df(df)
    kinds = {c["name"]: c["kind"] for c in prof["columns"]}
    assert kinds["year"] == "time" and kinds["gdp"] == "numeric" and kinds["country"] == "categorical"


def test_downloads_from_local_server(tmp_path):
    """Real Toolkit.download() against a local HTTP server: PDF, CSV and HTML sniffing + size limit."""
    from research_agent.tools.demo import _make_pdf

    site = tmp_path / "site"
    site.mkdir()
    _make_pdf(site / "paper.pdf", "Local paper")
    (site / "data.csv").write_text("year,v\n" + "\n".join(f"{y},{y*2}" for y in range(2000, 2030)))
    (site / "page.html").write_text("<html><body><article>" + "<p>Some long paragraph of text for testing extraction.</p>" * 20 + "</article></body></html>")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(http.server.SimpleHTTPRequestHandler, directory=str(site)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        tk = Toolkit(Config())
        out = tmp_path / "out"
        pdf = tk.download(f"{base}/paper.pdf", out)
        assert pdf["kind"] == "pdf" and len(extract_pdf_text(pdf["path"])[0]) > 300
        csv_ = tk.download(f"{base}/data.csv", out)
        assert csv_["kind"] == "csv" and len(load_table(csv_["path"])) == 30
        html = tk.download(f"{base}/page.html", out)
        assert html["kind"] == "html" and len(html_to_text(Path(html["path"]).read_text())[0]) > 300
        try:
            tk.download(f"{base}/paper.pdf", out, max_mb=0.0001)
            raise AssertionError("size limit not enforced")
        except ValueError:
            pass
    finally:
        srv.shutdown()


def test_full_offline_pipeline(tmp_path):
    cfg = Config().apply_depth("quick")
    cfg.offline_demo, cfg.llm_provider, cfg.verbose, cfg.runs_dir = True, "mock", False, tmp_path
    pdf = ResearchOrchestrator("urban heat islands and tree cover", cfg).run()
    assert pdf.exists() and pdf.stat().st_size > 20_000
    run = pdf.parent
    assert (run / "report.md").exists() and (run / "state.json").exists()
    assert list((run / "charts").glob("*.png")) and list((run / "datasets").glob("*.csv"))
    from pypdf import PdfReader

    text = " ".join(p.extract_text() for p in PdfReader(str(pdf)).pages)
    assert "Executive Summary" in text and "References" in text


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
