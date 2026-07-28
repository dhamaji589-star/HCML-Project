"""Create a simple text PDF from the project conceptual Markdown notes."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a simple PDF from Markdown text.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def markdown_to_lines(text: str) -> list[str]:
    lines = []
    in_code = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if line.startswith("# "):
            lines.append(line[2:].upper())
            lines.append("")
        elif line.startswith("## "):
            lines.append(line[3:].upper())
            lines.append("")
        elif line.startswith("> "):
            lines.append("Quote: " + line[2:])
        elif in_code:
            lines.append("  " + line)
        else:
            lines.append(line)
    return lines


def wrap_lines(lines: list[str], width: int = 88) -> list[str]:
    wrapped = []
    for line in lines:
        if not line:
            wrapped.append("")
        elif line.startswith("  ") or line.startswith("- "):
            wrapped.extend(textwrap.wrap(line, width=width, subsequent_indent="  ") or [""])
        else:
            wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return wrapped


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(lines: list[str], lines_per_page: int = 45) -> bytes:
    pages = []
    page = []
    for line in lines:
        page.append(line)
        if len(page) >= lines_per_page:
            pages.append(page)
            page = []
    if page:
        pages.append(page)

    objects = ["<< /Type /Catalog /Pages 2 0 R >>"]
    page_refs = []
    page_streams = []
    for index, page_lines in enumerate(pages):
        page_obj_num = 4 + index * 2
        content_obj_num = page_obj_num + 1
        page_refs.append(f"{page_obj_num} 0 R")
        stream_lines = ["BT", "/F1 10 Tf", "50 790 Td", "14 TL"]
        for line_index, line in enumerate(page_lines):
            if line_index:
                stream_lines.append("T*")
            stream_lines.append(f"({escape_pdf_text(line)}) Tj")
        stream_lines.append("ET")
        page_streams.append((page_obj_num, content_obj_num, "\n".join(stream_lines)))

    objects.append(f"<< /Type /Pages /Kids [{' '.join(page_refs)}] /Count {len(page_refs)} >>")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for page_obj_num, content_obj_num, stream in page_streams:
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_obj_num} 0 R >>"
        )
        stream_length = len(stream.encode("latin-1", errors="replace"))
        objects.append(f"<< /Length {stream_length} >>\nstream\n{stream}\nendstream")

    pdf_parts = ["%PDF-1.4"]
    offsets = []
    for obj_num, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part.encode("latin-1", errors="replace")) + 1 for part in pdf_parts))
        pdf_parts.append(f"{obj_num} 0 obj")
        pdf_parts.append(obj)
        pdf_parts.append("endobj")

    xref_offset = sum(len(part.encode("latin-1", errors="replace")) + 1 for part in pdf_parts)
    pdf_parts.append("xref")
    pdf_parts.append(f"0 {len(objects) + 1}")
    pdf_parts.append("0000000000 65535 f ")
    for offset in offsets:
        pdf_parts.append(f"{offset:010d} 00000 n ")
    pdf_parts.append("trailer")
    pdf_parts.append(f"<< /Size {len(objects) + 1} /Root 1 0 R >>")
    pdf_parts.append("startxref")
    pdf_parts.append(str(xref_offset))
    pdf_parts.append("%%EOF")

    return ("\n".join(pdf_parts) + "\n").encode("latin-1", errors="replace")


def main() -> None:
    args = parse_args()
    lines = wrap_lines(markdown_to_lines(args.source.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(build_pdf(lines))
    print(f"PDF written: {args.output}")


if __name__ == "__main__":
    main()
