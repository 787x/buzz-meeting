import ast
from dataclasses import dataclass
from pathlib import Path
import zipfile
from xml.etree import ElementTree

import pytest

from buzz.transcriber import docx_writer
from buzz.transcriber.docx_writer import DocxRun, DocxWriter, write_plain_docx


_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NAMESPACES = {"w": _WORD_NAMESPACE}
_XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


@dataclass(frozen=True)
class _Segment:
    text: str | None
    start_time: int
    end_time: int


def _document_root(path: Path) -> ElementTree.Element:
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
    return ElementTree.fromstring(document_xml)


def _paragraphs(path: Path) -> list[ElementTree.Element]:
    return _document_root(path).findall(".//w:body/w:p", _NAMESPACES)


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    return "".join(node.text or "" for node in paragraph.findall(".//w:t", _NAMESPACES))


def _run_details(paragraph: ElementTree.Element) -> list[tuple[str, bool]]:
    details = []
    for run in paragraph.findall("w:r", _NAMESPACES):
        text = run.find("w:t", _NAMESPACES)
        bold = run.find("w:rPr/w:b", _NAMESPACES) is not None
        details.append((text.text or "", bold))
    return details


def test_generic_writer_package_xml_escaping_unicode_and_order(tmp_path):
    out_path = tmp_path / "generic.docx"
    injected = "A & B < C > D </w:t><w:p>"
    unicode_text = "中文 café 😀 مرحبا"

    writer = DocxWriter()
    writer.add_title("Document title")
    writer.add_heading("Section heading")
    writer.add_paragraph(injected)
    writer.add_bullet(unicode_text)
    writer.write(str(out_path))

    with zipfile.ZipFile(out_path) as archive:
        assert set(archive.namelist()) == {
            "[Content_Types].xml",
            "_rels/.rels",
            "word/document.xml",
        }
        assert all(
            item.compress_type == zipfile.ZIP_DEFLATED for item in archive.infolist()
        )
        document_xml = archive.read("word/document.xml")

    ElementTree.fromstring(document_xml)
    paragraphs = _paragraphs(out_path)
    assert [_paragraph_text(item) for item in paragraphs] == [
        "Document title",
        "Section heading",
        injected,
        f"• {unicode_text}",
    ]
    assert _run_details(paragraphs[0]) == [("Document title", True)]
    assert _run_details(paragraphs[1]) == [("Section heading", True)]
    assert _run_details(paragraphs[3]) == [("• ", False), (unicode_text, False)]


def test_generic_writer_preserves_rich_runs(tmp_path):
    out_path = tmp_path / "runs.docx"
    writer = DocxWriter()
    writer.add_paragraph([DocxRun("Owner:", bold=True), DocxRun(" Alice")])
    writer.write(str(out_path))

    assert _run_details(_paragraphs(out_path)[0]) == [
        ("Owner:", True),
        (" Alice", False),
    ]


def test_generic_writer_preserves_newlines_and_whitespace(tmp_path):
    out_path = tmp_path / "whitespace.docx"
    text = "  line1\nline2   with  spaces  "
    writer = DocxWriter()
    writer.add_paragraph(text)
    writer.write(str(out_path))

    with zipfile.ZipFile(out_path) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")

    paragraph = _paragraphs(out_path)[0]
    text_node = paragraph.find(".//w:t", _NAMESPACES)
    assert text_node is not None
    assert text_node.text == text
    assert text_node.attrib[_XML_SPACE] == "preserve"
    assert "<w:br" not in document_xml


def test_generic_writer_instances_do_not_share_state(tmp_path):
    first_path = tmp_path / "first.docx"
    second_path = tmp_path / "second.docx"
    first = DocxWriter()
    second = DocxWriter()

    first.add_paragraph("first only")
    second.add_paragraph("second only")
    first.write(str(first_path))
    second.write(str(second_path))

    assert [_paragraph_text(item) for item in _paragraphs(first_path)] == ["first only"]
    assert [_paragraph_text(item) for item in _paragraphs(second_path)] == [
        "second only"
    ]


def test_generic_writer_propagates_missing_parent_error(tmp_path):
    writer = DocxWriter()
    writer.add_paragraph("content")

    with pytest.raises(FileNotFoundError):
        writer.write(str(tmp_path / "missing" / "document.docx"))

    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (1999, ["one two"]),
        (2000, ["one", "two"]),
        (2001, ["one", "two"]),
    ],
)
def test_plain_docx_paragraph_split_boundary(tmp_path, gap, expected):
    out_path = tmp_path / f"gap-{gap}.docx"
    segments = [
        _Segment(" one ", 0, 1000),
        _Segment(" two ", 1000 + gap, 4000),
    ]

    write_plain_docx(str(out_path), "Title", segments, False)

    assert [_paragraph_text(item) for item in _paragraphs(out_path)] == [
        "Title",
        *expected,
    ]


def test_plain_docx_empty_segment_updates_previous_end(tmp_path):
    out_path = tmp_path / "empty.docx"
    segments = [
        _Segment(" one ", 0, 100),
        _Segment("   ", 1000, 2000),
        _Segment(" two ", 3999, 4500),
        _Segment("", 8000, 8100),
    ]

    write_plain_docx(str(out_path), "Title", segments, False)

    assert [_paragraph_text(item) for item in _paragraphs(out_path)] == [
        "Title",
        "one two",
    ]


def test_plain_docx_timestamps_are_exact_and_bold(tmp_path):
    out_path = tmp_path / "timestamps.docx"
    segments = [
        _Segment(" hello ", 1234, 5678),
        _Segment(" ", 6000, 7000),
    ]

    write_plain_docx(str(out_path), "Title", segments, True)

    paragraphs = _paragraphs(out_path)
    assert [_paragraph_text(item) for item in paragraphs] == [
        "Title",
        "[00:00:01.234 --> 00:00:05.678] hello",
    ]
    assert _run_details(paragraphs[1]) == [
        ("[00:00:01.234 --> 00:00:05.678] ", True),
        ("hello", False),
    ]


def test_plain_docx_preserves_newline_text(tmp_path):
    out_path = tmp_path / "newline.docx"
    write_plain_docx(
        str(out_path),
        "Title",
        [_Segment("line1\nline2", 0, 1000)],
        False,
    )

    with zipfile.ZipFile(out_path) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")

    assert _paragraph_text(_paragraphs(out_path)[1]) == "line1\nline2"
    assert "<w:br" not in document_xml


def test_plain_docx_does_not_normalize_none_text(tmp_path):
    with pytest.raises(AttributeError):
        write_plain_docx(
            str(tmp_path / "none.docx"),
            "Title",
            [_Segment(None, 0, 1000)],
            False,
        )


def test_plain_docx_delegates_to_writer(monkeypatch):
    calls = []

    class _RecordingWriter:
        def add_title(self, text):
            calls.append(("title", text))

        def add_paragraph(self, content):
            calls.append(("paragraph", content))

        def write(self, out_path):
            calls.append(("write", out_path))

    monkeypatch.setattr(docx_writer, "DocxWriter", _RecordingWriter)

    write_plain_docx(
        "output.docx",
        "Title",
        [_Segment(" one ", 0, 1000), _Segment(" two ", 2000, 3000)],
        False,
    )

    assert calls == [
        ("title", "Title"),
        ("paragraph", "one two"),
        ("write", "output.docx"),
    ]


def test_docx_writer_public_api_and_import_isolation():
    assert docx_writer.__all__ == [
        "PARAGRAPH_SPLIT_TIME",
        "DocxRun",
        "DocxWriter",
        "write_plain_docx",
    ]

    module_path = Path(docx_writer.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")

    assert all(
        not name.startswith(
            (
                "PyQt",
                "buzz.db",
                "buzz.meeting",
                "buzz.plugins",
                "buzz.transcriber.file_transcriber",
            )
        )
        for name in top_level_imports
    )


def _assert_plugin_has_no_docx_package_implementation(source: str) -> None:
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules = {alias.name for alias in node.names}
            assert imported_modules.isdisjoint({"zipfile", "xml.sax.saxutils"})
        elif isinstance(node, ast.ImportFrom):
            imported_names = {alias.name for alias in node.names}
            assert node.module not in {"zipfile", "xml.sax.saxutils"}
            assert not (node.module == "xml.sax" and "saxutils" in imported_names)

    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    package_markers = (
        "[Content_Types].xml",
        "_rels/.rels",
        "word/document.xml",
        "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "http://schemas.openxmlformats.org/package/2006/relationships",
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
    )
    executable_strings = (
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    )
    assert not any(
        marker in value for value in executable_strings for marker in package_markers
    )


def test_plugin_contains_no_docx_package_implementation():
    plugin_path = (
        Path(__file__).parents[2] / "buzz" / "plugins" / "export_docx" / "plugin.py"
    )
    source = plugin_path.read_text(encoding="utf-8")

    _assert_plugin_has_no_docx_package_implementation(source)


@pytest.mark.parametrize(
    "duplicate_implementation",
    [
        "from zipfile import ZipFile\n",
        "from zipfile import ZipFile as WordZip\n",
        "import zipfile as zf\n",
        "from xml.sax.saxutils import escape as xml_escape\n",
        'PACKAGE_DOC = "word/document.xml"\n',
        (
            'DOCUMENT_TEMPLATE = "<w:document '
            'xmlns:w=\\"http://schemas.openxmlformats.org/'
            'wordprocessingml/2006/main\\">"\n'
        ),
        (
            "def build_word_archive(path):\n"
            "    from zipfile import ZipFile as Archive\n"
            "    with Archive(path, 'w') as package:\n"
            "        package.writestr('word/document.xml', '<w:document/>')\n"
        ),
    ],
    ids=[
        "ZipFile-from-import",
        "aliased-ZipFile-from-import",
        "aliased-zipfile-import",
        "XML-escape-import",
        "DOCX-package-part-constant",
        "WordprocessingML-template",
        "renamed-package-helper",
    ],
)
def test_plugin_duplication_guard_rejects_renamed_implementations(
    duplicate_implementation,
):
    with pytest.raises(AssertionError):
        _assert_plugin_has_no_docx_package_implementation(duplicate_implementation)


def test_buzz_spec_collects_docx_writer():
    spec_path = Path(__file__).parents[2] / "Buzz.spec"
    assert '"buzz.transcriber.docx_writer"' in spec_path.read_text(encoding="utf-8")
