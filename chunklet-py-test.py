#!/usr/bin/env python3
"""
Streaming file chunker using chunklet-py.

Reads PDF, DOCX, TXT, MD files with bounded memory, splits content into
~800-token chunks with 12% overlap, and POSTs each chunk to an HTTP endpoint.

Usage:
    python main.py <file_path> [endpoint_url]
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Generator

import requests
from chunklet import DocumentChunker


# ---------------------------------------------------------------------------
# CJK-aware token estimator (ported from Go estimateTokens)
# ---------------------------------------------------------------------------

def _is_cjk(cp: int) -> bool:
    """Return True if *cp* is a CJK character, kana, hangul, or CJK-adjacent punctuation."""
    return (
        (0x4E00 <= cp <= 0x9FFF)   # CJK Unified Ideographs
        or (0x3400 <= cp <= 0x4DBF)  # CJK Extension A
        or (0x20000 <= cp <= 0x2A6DF)  # CJK Extension B
        or (0xF900 <= cp <= 0xFAFF)  # CJK Compatibility Ideographs
        or (0x3040 <= cp <= 0x309F)  # Hiragana
        or (0x30A0 <= cp <= 0x30FF)  # Katakana
        or (0xAC00 <= cp <= 0xD7AF)  # Hangul Syllables
        or (0x3000 <= cp <= 0x303F)  # CJK Symbols/Punctuation
        or (0xFF00 <= cp <= 0xFFEF)  # Halfwidth/Fullwidth Forms
    )


def estimate_tokens(text: str) -> int:
    """
    Conservative upper-bound token estimate.

    Heuristic:
      - CJK characters: 1.5 tokens each (typically 1-2 in real tokenizers)
      - Non-CJK, non-space chars: 0.5 tokens each (English avg ~0.25, code ~0.4)
    """
    cjk = 0
    other = 0
    for ch in text:
        if _is_cjk(ord(ch)):
            cjk += 1
        elif not ch.isspace():
            other += 1
    est = int(cjk * 1.5 + other * 0.5)
    return est if est >= 1 else (1 if text else 0)

# ---------------------------------------------------------------------------
# Text cleaning — removes noise that wastes chunk space and hurts embeddings
# ---------------------------------------------------------------------------

# Invisible / control characters (except \n kept for paragraph structure)
_INVISIBLE_RE = re.compile(
    '[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f'
    '\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]'
)

# HTML / XML tags
_HTML_TAG_RE = re.compile(r'<[^>]*>')

# HTML numeric / named entities (after common ones are decoded)
_HTML_ENTITY_RE = re.compile(r'&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;')

# Markdown image ![alt](url) — removed entirely
_MD_IMAGE_RE = re.compile(r'!\[[^\]]*\]\([^)]*\)')

# Markdown link [text](url) — keep text only
_MD_LINK_RE = re.compile(r'\[([^\]]*)\]\([^)]*\)')

# Markdown reference link [text][ref] — keep text only
_MD_REF_LINK_RE = re.compile(r'\[([^\]]*)\]\[[^\]]*\]')

# Markdown reference definition [ref]: url
_MD_REF_DEF_RE = re.compile(r'^\[[^\]]+\]:\s*\S.*$', re.MULTILINE)

# Markdown heading markers
_MD_HEADING_RE = re.compile(r'^#{1,6}\s+', re.MULTILINE)

# Markdown bold / italic (order: bold first, then italic)
_MD_BOLD_RE = re.compile(r'\*\*([^*]+)\*\*')
_MD_ITALIC_RE = re.compile(r'\*([^*]+)\*')

# Markdown blockquote
_MD_BLOCKQUOTE_RE = re.compile(r'^>\s?', re.MULTILINE)

# Markdown horizontal rules
_MD_HR_RE = re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE)

# Markdown code fences (remove markers, keep content)
_MD_CODE_FENCE_RE = re.compile(r'^```\w*$', re.MULTILINE)

# Punctuation repetition: 3+ → 1 for most (！！！→！, 。。。→。)
_PUNCT_REPEAT_RE = re.compile(r'([!！?？。，,;；:：])\1{2,}')

# English period repetition: 4+ → 1 (preserve ... as ellipsis)
_DOT_REPEAT_RE = re.compile(r'\.{4,}')

# base64-like strings (40+ chars of base64 alphabet)
_BASE64_RE = re.compile(r'[A-Za-z0-9+/]{40,}={0,2}')

# UUID
_UUID_RE = re.compile(
    r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-'
    r'[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
)

# Long hex strings (32+ hex digits)
_LONG_HEX_RE = re.compile(r'\b[0-9a-fA-F]{32,}\b')

# Long digit-only sequences (20+ digits, e.g. raw IDs / timestamps / serials)
_LONG_DIGIT_RE = re.compile(r'\b\d{20,}\b')

_HTML_ENTITIES = {
    '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
    '&apos;': "'", '&#39;': "'", '&nbsp;': ' ', '&ensp;': ' ',
    '&emsp;': ' ', '&ndash;': '-', '&mdash;': '—',
    '&lsquo;': "'", '&rsquo;': "'", '&ldquo;': '"', '&rdquo;': '"',
    '&hellip;': '…', '&copy;': '©', '&reg;': '®',
    '&trade;': '™',
}


def remove_invisible_chars(text: str) -> str:
    """Remove invisible/control characters.  \\t and \\r become spaces."""
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    return _INVISIBLE_RE.sub('', text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces → 1, multiple newlines → max 2."""
    text = re.sub(r'[ \t ]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^ +', '', text, flags=re.MULTILINE)
    text = re.sub(r' +$', '', text, flags=re.MULTILINE)
    return text.strip()


def remove_html_markdown_noise(text: str) -> str:
    """Strip HTML tags, decode entities, and remove Markdown formatting cruft."""
    text = _MD_IMAGE_RE.sub('', text)
    text = _MD_LINK_RE.sub(r'\1', text)
    text = _MD_REF_LINK_RE.sub(r'\1', text)
    text = _MD_REF_DEF_RE.sub('', text)
    text = _MD_HEADING_RE.sub('', text)
    text = _MD_BOLD_RE.sub(r'\1', text)
    text = _MD_ITALIC_RE.sub(r'\1', text)
    text = _MD_BLOCKQUOTE_RE.sub('', text)
    text = _MD_HR_RE.sub('', text)
    text = _MD_CODE_FENCE_RE.sub('', text)
    text = _HTML_TAG_RE.sub('', text)
    for entity, char in _HTML_ENTITIES.items():
        text = text.replace(entity, char)
    text = _HTML_ENTITY_RE.sub('', text)
    return text


def remove_duplicate_lines(text: str) -> str:
    """Remove boilerplate lines that repeat across the document.

    Footers, headers, and navigation bars repeat verbatim across many pages.
    A line is boilerplate when it appears ≥ 3 times AND is either short
    (< 100 chars) or dominates > 30 % of non-empty lines.
    """
    lines = text.split('\n')
    if len(lines) < 5:
        return text

    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        counts[stripped] = counts.get(stripped, 0) + 1

    total = sum(1 for l in lines if l.strip())
    if total == 0:
        return text

    boilerplate: set[str] = set()
    for line, cnt in counts.items():
        if cnt >= 3 and (len(line) < 100 or cnt > total * 0.3):
            boilerplate.add(line)

    if not boilerplate:
        return text

    return '\n'.join(l for l in lines if l.strip() not in boilerplate)


def normalize_punctuation(text: str) -> str:
    """Collapse repeated punctuation: !!! → !, 。。。→ 。, .... → ."""
    text = _PUNCT_REPEAT_RE.sub(r'\1', text)
    text = _DOT_REPEAT_RE.sub('.', text)
    return text


def remove_long_meaningless_segments(text: str) -> str:
    """Drop base64, UUIDs, long hex / digit strings."""
    text = _BASE64_RE.sub(' ', text)
    text = _UUID_RE.sub(' ', text)
    text = _LONG_HEX_RE.sub(' ', text)
    text = _LONG_DIGIT_RE.sub(' ', text)
    return text


def clean_text(text: str) -> str:
    """Apply per-fragment cleaning passes to *text*.

    Order: remove noise → normalize punctuation → drop junk segments →
    normalize whitespace.
    """
    if not text:
        return text
    text = remove_invisible_chars(text)
    text = remove_html_markdown_noise(text)
    text = normalize_punctuation(text)
    text = remove_long_meaningless_segments(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".docx": "docx", ".md": "md", ".txt": "txt"}


def detect_file_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]

    # MIME-based fallback for files with missing / ambiguous extension
    try:
        import mimetypes

        mime, _ = mimetypes.guess_type(file_path)
        if mime:
            if mime == "text/plain":
                return "txt"
            if mime == "application/pdf":
                return "pdf"
            if mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return "docx"
    except Exception:
        pass

    raise ValueError(
        f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


# ---------------------------------------------------------------------------
# Streaming file readers (generator-based, bounded memory)
# ---------------------------------------------------------------------------

TEXT_READ_SIZE = 64 * 1024  # 64 KB per read for plain-text files
DOCX_FLUSH_SIZE = 64 * 1024  # flush paragraph buffer after this many chars


def stream_read_text(file_path: str) -> Generator[str, None, None]:
    """Yield text blocks from a plain-text or markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            block = f.read(TEXT_READ_SIZE)
            if not block:
                break
            yield block


def stream_read_pdf(file_path: str) -> Generator[str, None, None]:
    """Yield text page-by-page from a PDF file."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF support. Install with: pip install pdfplumber"
        )

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                yield text


def stream_read_docx(file_path: str) -> Generator[str, None, None]:
    """Yield paragraph groups from a .docx file, flushing periodically."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx is required for DOCX support. Install with: pip install python-docx"
        )

    doc = Document(file_path)
    buf: list[str] = []
    buf_size = 0

    for para in doc.paragraphs:
        text = para.text
        if not text:
            continue
        buf.append(text)
        buf_size += len(text)
        if buf_size >= DOCX_FLUSH_SIZE:
            yield "\n".join(buf)
            buf.clear()
            buf_size = 0

    if buf:
        yield "\n".join(buf)


def stream_read(file_path: str, file_type: str) -> Generator[str, None, None]:
    """Dispatch to the correct streaming reader based on file type."""
    if file_type in ("txt", "md"):
        yield from stream_read_text(file_path)
    elif file_type == "pdf":
        yield from stream_read_pdf(file_path)
    elif file_type == "docx":
        yield from stream_read_docx(file_path)
    else:
        raise ValueError(f"Unknown file type: {file_type}")


# ---------------------------------------------------------------------------
# Chunking with chunklet-py
# ---------------------------------------------------------------------------


def chunk_stream(
    file_path: str,
    file_type: str,
    max_tokens: int = 800,
    overlap_percent: int = 12,
) -> Generator[dict, None, None]:
    """
    Stream content from *file_path*, chunk it with chunklet-py, and yield dicts.

    Cross-batch semantic continuity is preserved by carrying the final chunk of
    each batch into the next batch as a prefix.
    """
    chunker = DocumentChunker(token_counter=estimate_tokens)
    accumulator = ""

    # Trigger chunklet when we have enough estimated tokens for at least 4 chunks.
    batch_token_threshold = max_tokens * 4
    # Safety valve: force-process if accumulator grows absurdly large.
    max_token_accumulator = max_tokens * 30

    chunk_index = 0

    for text_fragment in stream_read(file_path, file_type):
        # Clean fragment immediately so the accumulator never holds raw noise.
        fragment = clean_text(text_fragment)
        if not fragment:
            continue

        if accumulator:
            accumulator += "\n" + fragment
        else:
            accumulator = fragment

        acc_tokens = estimate_tokens(accumulator)
        if acc_tokens < batch_token_threshold and acc_tokens < max_token_accumulator:
            continue

        # Dedup boilerplate lines now that we have enough batch context.
        accumulator = remove_duplicate_lines(accumulator)
        if not accumulator.strip():
            accumulator = ""
            continue

        chunks = chunker.chunk_text(
            accumulator,
            max_tokens=max_tokens,
            overlap_percent=overlap_percent,
        )

        if not chunks:
            accumulator = ""
            continue

        if len(chunks) == 1:
            # A single chunk means we haven't accumulated enough meaningful text
            # to split.  Keep accumulating unless we hit the safety valve.
            if acc_tokens < max_token_accumulator:
                continue
            # Force-emit this single chunk.
            yield _to_dict(chunk_index, chunks[0].content)
            chunk_index += 1
            accumulator = ""
            continue

        # Emit all chunks except the last — it is carried forward as the
        # cross-batch overlap prefix for the next call.
        for chunk in chunks[:-1]:
            yield _to_dict(chunk_index, chunk.content)
            chunk_index += 1

        accumulator = chunks[-1].content

    # Process remaining text (already cleaned, just dedup).
    accumulator = remove_duplicate_lines(accumulator)
    if accumulator.strip():
        chunks = chunker.chunk_text(
            accumulator,
            max_tokens=max_tokens,
            overlap_percent=overlap_percent,
        )
        for chunk in chunks:
            yield _to_dict(chunk_index, chunk.content)
            chunk_index += 1


def _to_dict(index: int, content: str) -> dict:
    return {
        "index": index,
        "text": content,
        "size": len(content),
    }


# ---------------------------------------------------------------------------
# HTTP delivery
# ---------------------------------------------------------------------------

def send_chunk(chunk: dict, endpoint: str, timeout: int = 30) -> bool:
    """POST a single chunk to *endpoint*.  Returns True on success."""
    try:
        resp = requests.post(endpoint, json=chunk, timeout=timeout)
        resp.raise_for_status()
        return True
    except requests.exceptions.RequestException as exc:
        print(f"  [ERROR] Failed to send chunk {chunk['index']}: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_file(
    file_path: str,
    endpoint: str = "http://localhost:8080/api/v1/chunks",
    max_tokens: int = 800,
    overlap_percent: int = 12,
) -> tuple[int, int]:
    """
    Detect file type, stream content, chunk it, and POST each chunk.

    Returns (sent_count, fail_count).
    """
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_type = detect_file_type(file_path)

    print(f"File  : {file_path}")
    print(f"Type  : {file_type}")
    print(f"Chunk : max_tokens={max_tokens}, overlap={overlap_percent}%")
    print(f"Target: {endpoint}")
    print("-" * 60, flush=True)

    sent = 0
    failed = 0

    for chunk in chunk_stream(file_path, file_type, max_tokens, overlap_percent):
        status = "OK" if send_chunk(chunk, endpoint) else "FAIL"
        if status == "OK":
            sent += 1
        else:
            failed += 1
        print(
            f"  chunk {chunk['index']:04d}  "
            f"size={chunk['size']:5d} chars  "
            f"[{status}]",
            flush=True,
        )

    print("-" * 60)
    print(f"Done. Sent: {sent}, Failed: {failed}")
    return sent, failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <file_path> [endpoint_url]")
        print()
        print("Arguments:")
        print("  file_path      Path to a .txt, .md, .pdf, or .docx file")
        print("  endpoint_url   HTTP endpoint (default: http://localhost:8080/api/v1/chunks)")
        sys.exit(1)

    file_path = sys.argv[1]
    endpoint = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080/api/v1/chunks"

    try:
        sent, failed = process_file(file_path, endpoint)
        if failed > 0:
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
