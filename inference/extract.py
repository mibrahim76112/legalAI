"""Uploaded file bytes -> plain contract text (PDF, DOCX, TXT)."""

import io
import re
import zipfile
from xml.etree import ElementTree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _pdf(data):
    from pypdf import PdfReader
    pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages]
    # pypdf emits tabs between words in many layouts; the model was trained on
    # ordinary spacing, so collapse runs of spaces and tabs within a line
    text = "\n".join(re.sub(r"[ \t]+", " ", pg) for pg in pages)
    return _unwrap(text)


_ENDS_SENTENCE = re.compile(r"[.;:!?][”’\"')]*$")
_SECTION_START = re.compile(r"^(\d+(\.\d+)*\.?|§\s*\d+|[A-Z]\.|\([a-z0-9]{1,4}\))\s")
_CONTINUES = re.compile(r"^[,;:)a-z]")


def _unwrap(text):
    """Rejoin the visual line wraps a PDF keeps, so sentences reach the model
    whole: the model was trained on paragraph text, and a mid-sentence break
    makes it quote one fragment per line."""
    lines = [l.strip() for l in text.split("\n")]
    out = []
    for line in lines:
        if out and line and out[-1]:
            prev = out[-1]
            heading = len(prev) < 60 and (_SECTION_START.match(prev) or prev.isupper())
            joins = _CONTINUES.match(line) or not (
                _ENDS_SENTENCE.search(prev) or heading or _SECTION_START.match(line))
            if joins:
                out[-1] = prev + ("" if line[0] in ",;:)" else " ") + line
                continue
        out.append(line)
    return "\n".join(out)


def _docx(data):
    root = ElementTree.fromstring(zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml"))
    paras = []
    for p in root.iter(f"{W}p"):
        parts = []
        for el in p.iter():
            if el.tag == f"{W}t":
                parts.append(el.text or "")
            elif el.tag == f"{W}tab":
                parts.append("\t")
            elif el.tag in (f"{W}br", f"{W}cr"):
                parts.append("\n")
        paras.append("".join(parts))
    return "\n".join(paras)


def extract_text(filename, data):
    name = filename.lower()
    if name.endswith(".pdf"):
        text = _pdf(data)
    elif name.endswith(".docx"):
        text = _docx(data)
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError("Unsupported file type; use PDF, DOCX or TXT")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError("No text found in the file (a scanned PDF needs OCR first)")
    return text
