"""Document intelligence: extract text + structure from PDF, DOCX and CSV files and split into
citation-ready chunks (each chunk keeps source file, page/section/row metadata)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import pdfplumber
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from ..config import settings
from ..text_utils import normalise_ws

_HEADING_RE = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][^.!?]{2,80}$")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str            # file name
    doc_type: str          # pdf | docx | csv | txt
    location: str          # "page 2", "section 4", "row 17"
    section: str = ""
    customer_id: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ raw extraction
def extract_pdf(path: Path) -> list[tuple[str, str, str]]:
    """Return [(section_title, text, location)] blocks. Headings detected by numbering pattern."""
    blocks, cur_title, buf, cur_loc = [], "", [], "page 1"
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            for line in (page.extract_text() or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                if _HEADING_RE.match(line):
                    if buf:
                        blocks.append((cur_title, " ".join(buf), cur_loc))
                    cur_title, buf, cur_loc = line, [], f"page {pno}"
                else:
                    buf.append(line)
            if not cur_title and buf and pno == 1:
                pass
    if buf:
        blocks.append((cur_title, " ".join(buf), cur_loc))
    return blocks


def extract_docx(path: Path) -> list[tuple[str, str, str]]:
    doc = Document(path)
    blocks, title, buf = [], "", []
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            p = Paragraph(child, doc)
            txt = p.text.strip()
            if not txt:
                continue
            if p.style.name.lower().startswith(("heading", "title")):
                if buf:
                    blocks.append((title, " ".join(buf), f"section {title.split()[0].rstrip('.') if title else '0'}"))
                title, buf = txt, []
            else:
                buf.append(txt)
        elif child.tag.endswith("}tbl"):
            tbl = Table(child, doc)
            rows = [[c.text.strip() for c in r.cells] for r in tbl.rows]
            if rows:
                hdr = rows[0]
                for r in rows[1:]:
                    buf.append("; ".join(f"{h}: {v}" for h, v in zip(hdr, r)) + ".")
    if buf:
        blocks.append((title, " ".join(buf), f"section {title.split()[0].rstrip('.') if title else '0'}"))
    return blocks


def extract_csv(path: Path) -> list[dict]:
    """Each CSV row becomes a record; text columns are rendered as 'column: value' sentences."""
    df = pd.read_csv(path).fillna("")
    return df.to_dict("records")


# ------------------------------------------------------------------ chunking
def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Sentence-aware splitter: pack sentences up to `size` chars, carrying `overlap` chars between chunks."""
    sents = re.split(r"(?<=[.!?])\s+", normalise_ws(text))
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > size:
            chunks.append(cur.strip())
            cur = cur[-overlap:].split(" ", 1)[-1] + " " if overlap else ""
        cur += s + " "
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def load_file(path: Path) -> list[Chunk]:
    path = Path(path)
    ext = path.suffix.lower()
    name = path.name
    size, ov = settings.chunk_chars, settings.chunk_overlap
    out: list[Chunk] = []
    if ext in (".pdf", ".docx"):
        blocks = extract_pdf(path) if ext == ".pdf" else extract_docx(path)
        for bi, (title, text, loc) in enumerate(blocks):
            for ci, piece in enumerate(split_text(text, size, ov)):
                # prefix the section title so the chunk is self-describing for retrieval
                full = f"{title}. {piece}" if title else piece
                out.append(Chunk(f"{name}::{bi}.{ci}", full, name, ext[1:], loc, section=title))
    elif ext == ".csv":
        for i, rec in enumerate(extract_csv(path), 1):
            cid = str(rec.get("customer_id", "") or "")
            text = " | ".join(f"{k}: {v}" for k, v in rec.items() if str(v) != "")
            out.append(Chunk(f"{name}::row{i}", text, name, "csv", f"row {i}", section=name.rsplit('.', 1)[0],
                             customer_id=cid, meta={k: str(v) for k, v in rec.items()}))
    elif ext in (".txt", ".md"):
        for ci, piece in enumerate(split_text(path.read_text(errors="ignore"), size, ov)):
            out.append(Chunk(f"{name}::{ci}", piece, name, "txt", f"part {ci + 1}"))
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return out


def load_directory(directory: Path | None = None) -> list[Chunk]:
    directory = Path(directory or settings.documents_dir)
    chunks: list[Chunk] = []
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() in {".pdf", ".docx", ".csv", ".txt", ".md"}:
            chunks.extend(load_file(p))
    return chunks
