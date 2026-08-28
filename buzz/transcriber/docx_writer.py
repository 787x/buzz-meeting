"""Minimal standard-library DOCX writing primitives."""

from collections.abc import Sequence
from dataclasses import dataclass
import zipfile
from xml.sax.saxutils import escape

__all__ = [
    "PARAGRAPH_SPLIT_TIME",
    "DocxRun",
    "DocxWriter",
    "write_plain_docx",
]

# Same default gap (ms) used by the TXT exporter to split paragraphs.
PARAGRAPH_SPLIT_TIME = 2000

_CONTENT_TYPES_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" '
    'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.'
    'wordprocessingml.document.main+xml"/>'
    "</Types>"
)

_RELS_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="word/document.xml"/>'
    "</Relationships>"
)

_DOCUMENT_TEMPLATE = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    "<w:body>{body}<w:sectPr/></w:body>"
    "</w:document>"
)


@dataclass(frozen=True)
class DocxRun:
    text: str
    bold: bool = False


class DocxWriter:
    """Assemble a small document and write it as a minimal DOCX package."""

    def __init__(self) -> None:
        self._paragraphs: list[str] = []

    def add_title(self, text: str) -> None:
        self._paragraphs.append(_title_xml(text))

    def add_heading(self, text: str) -> None:
        self._paragraphs.append(_heading_xml(text))

    def add_paragraph(self, content: str | Sequence[DocxRun]) -> None:
        self._paragraphs.append(_paragraph_xml(_content_runs(content)))

    def add_bullet(self, content: str | Sequence[DocxRun]) -> None:
        runs = [DocxRun("• "), *_content_runs(content)]
        self._paragraphs.append(_paragraph_xml(runs))

    def write(self, out_path: str) -> None:
        document_xml = _DOCUMENT_TEMPLATE.format(body="".join(self._paragraphs))
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as docx:
            docx.writestr("[Content_Types].xml", _CONTENT_TYPES_XML)
            docx.writestr("_rels/.rels", _RELS_XML)
            docx.writestr("word/document.xml", document_xml)


def write_plain_docx(
    out_path: str,
    title: str,
    segments,
    include_timestamps: bool,
) -> None:
    """Write transcript segments using the generic DOCX writer."""
    from buzz.transcriber.file_transcriber import to_timestamp

    writer = DocxWriter()
    writer.add_title(title)

    if include_timestamps:
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            stamp = (
                f"[{to_timestamp(segment.start_time)} --> "
                f"{to_timestamp(segment.end_time)}]"
            )
            writer.add_paragraph([DocxRun(stamp + " ", bold=True), DocxRun(text)])
    else:
        current = []
        previous_end = None
        for segment in segments:
            if (
                previous_end is not None
                and (segment.start_time - previous_end) >= PARAGRAPH_SPLIT_TIME
                and current
            ):
                writer.add_paragraph(" ".join(current))
                current = []
            text = segment.text.strip()
            if text:
                current.append(text)
            previous_end = segment.end_time
        if current:
            writer.add_paragraph(" ".join(current))

    writer.write(out_path)


def _content_runs(content: str | Sequence[DocxRun]) -> list[DocxRun]:
    if isinstance(content, str):
        return [DocxRun(content)]
    return list(content)


def _run_xml(run: DocxRun) -> str:
    props = "<w:rPr><w:b/></w:rPr>" if run.bold else ""
    return f'<w:r>{props}<w:t xml:space="preserve">' f"{escape(run.text)}</w:t></w:r>"


def _paragraph_xml(runs: Sequence[DocxRun]) -> str:
    return "<w:p>" + "".join(_run_xml(run) for run in runs) + "</w:p>"


def _title_xml(text: str) -> str:
    return (
        '<w:p><w:pPr><w:spacing w:after="200"/></w:pPr>'
        '<w:r><w:rPr><w:b/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr>'
        f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
    )


def _heading_xml(text: str) -> str:
    return (
        '<w:p><w:pPr><w:spacing w:before="160" w:after="80"/></w:pPr>'
        '<w:r><w:rPr><w:b/><w:sz w:val="28"/><w:szCs w:val="28"/></w:rPr>'
        f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
    )
