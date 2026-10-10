from __future__ import annotations

from ..state import Source
from ..tools.doc_tools import extract_pdf_text
from .base import BaseAgent

MAX_TEXT = 90_000


class ReaderAgent(BaseAgent):
    name = "DocumentReader"
    role = "You extract clean text from PDFs and web pages."

    def run(self, sources: list[Source]) -> None:
        read = abstract = failed = 0
        for s in sources:
            if s.kind == "dataset" or not s.selected:
                continue
            if s.status == "downloaded":
                if s.file_kind == "pdf":
                    text, pages = extract_pdf_text(s.local_path)
                    s.pages = pages
                    s.text = text[:MAX_TEXT]
                if len(s.text) >= 300:
                    s.status = "read"
                    read += 1
                    continue
                s.error = "no extractable text (scanned PDF?)"
                s.status = "failed"
            if s.status == "failed" and len(s.snippet) >= 150:
                s.text, s.status = s.snippet, "abstract_only"
                abstract += 1
            elif s.status == "failed":
                failed += 1
        self.log(f"read {read} full documents, {abstract} abstract-only fallbacks, {failed} unusable")
