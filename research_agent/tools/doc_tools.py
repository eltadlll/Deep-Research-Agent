"""File sniffing + text extraction from PDFs and HTML."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path


def sniff_file(path: str, content_type: str = "") -> str:
    """Return one of: pdf, html, csv, xlsx, xls, json, zip, parquet, text, unknown."""
    p = Path(path)
    try:
        head = p.read_bytes()[:4096]
    except Exception:
        return "unknown"
    ext = p.suffix.lower().lstrip(".")
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PAR1"):
        return "parquet"
    if head.startswith(b"PK\x03\x04"):
        try:
            names = zipfile.ZipFile(path).namelist()
            if any(n.startswith("xl/") for n in names):
                return "xlsx"
        except Exception:
            pass
        return "zip"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "xls"
    low = head.lstrip().lower()
    if low.startswith((b"<!doctype html", b"<html", b"<head", b"<body")) or b"<html" in low[:500]:
        return "html"
    if "json" in content_type or low.startswith((b"{", b"[")):
        return "json"
    if ext in ("csv", "tsv", "txt") or "csv" in content_type or "text/plain" in content_type:
        try:
            head.decode("utf-8")
        except UnicodeDecodeError:
            pass
        return "csv" if ext in ("csv", "tsv") or "csv" in content_type else "text"
    if ext in ("xlsx", "xls", "json", "parquet", "zip", "pdf"):
        return ext
    if head.count(b",") > 5 and b"\n" in head:
        return "csv"
    return "unknown"


def html_to_text(html: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg", "iframe"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text("\n")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    # drop very short boilerplate lines but keep headings-ish lines
    lines = [ln for ln in lines if len(ln) > 25 or (ln and ln[0].isupper() and len(ln) > 8)]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, title


def _clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(path: str, max_pages: int = 80) -> tuple[str, int]:
    """Return (text, total_pages)."""
    text, pages = "", 0
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = len(reader.pages)
        parts = []
        for pg in reader.pages[:max_pages]:
            try:
                parts.append(pg.extract_text() or "")
            except Exception:
                continue
        text = "\n".join(parts)
    except Exception:
        pass
    if len(text.strip()) < 200:
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                pages = pages or len(pdf.pages)
                text = "\n".join((pg.extract_text() or "") for pg in pdf.pages[:max_pages])
        except Exception:
            pass
    return _clean_pdf_text(text), pages


def extract_pdf_tables(path: str, max_pages: int = 15, max_tables: int = 3) -> list[list[list[str]]]:
    out: list[list[list[str]]] = []
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for pg in pdf.pages[:max_pages]:
                for tb in pg.extract_tables() or []:
                    if tb and len(tb) > 2 and len(tb[0]) > 1:
                        out.append([[(c or "").strip() for c in row] for row in tb])
                    if len(out) >= max_tables:
                        return out
    except Exception:
        pass
    return out
