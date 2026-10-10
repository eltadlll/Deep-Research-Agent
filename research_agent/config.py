from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional
    pass

# depth presets -> overrides applied on top of Config defaults
PRESETS = {
    "quick": dict(max_sources=6, max_datasets=1, research_rounds=1, max_chunks_per_source=1, max_sections=4),
    "standard": dict(max_sources=12, max_datasets=3, research_rounds=2, max_chunks_per_source=2, max_sections=5),
    "deep": dict(max_sources=20, max_datasets=5, research_rounds=3, max_chunks_per_source=3, max_sections=6),
}


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass
class Config:
    # LLM
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "gemini").lower())
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.0-flash"))
    requests_per_minute: int = field(default_factory=lambda: int(_env("LLM_RPM", "10")))
    max_prompt_chars: int = 60000

    # research budget
    depth: str = "standard"
    max_sources: int = 12
    max_datasets: int = 3
    research_rounds: int = 2
    max_chunks_per_source: int = 2
    max_sections: int = 5
    search_results_per_query: int = 8

    # network / files
    max_file_mb: float = field(default_factory=lambda: float(_env("MAX_FILE_MB", "25")))
    request_timeout: int = field(default_factory=lambda: int(_env("REQUEST_TIMEOUT", "30")))
    contact_email: str = field(default_factory=lambda: _env("CONTACT_EMAIL"))
    user_agent: str = "Mozilla/5.0 (compatible; DeepResearchAgent/1.0; +research-bot)"
    download_workers: int = 4

    # output
    runs_dir: Path = field(default_factory=lambda: Path(_env("RUNS_DIR", "runs")))
    verbose: bool = True

    # offline demo (fake search + fake downloads, use with LLM_PROVIDER=mock)
    offline_demo: bool = False

    def apply_depth(self, depth: str) -> "Config":
        depth = (depth or "standard").lower()
        if depth not in PRESETS:
            raise ValueError(f"depth must be one of {list(PRESETS)}")
        self.depth = depth
        for k, v in PRESETS[depth].items():
            setattr(self, k, v)
        return self
