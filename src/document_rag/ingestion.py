from pathlib import Path

from pypdf import PdfReader
from docx import Document


def load_pdf(file_path):
    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page_number": page_number,
                "text": text
            })

    return pages


def load_docx(file_path):
    document = Document(file_path)

    paragraphs = []

    for paragraph_number, paragraph in enumerate(
        document.paragraphs,
        start=1
    ):
        text = paragraph.text.strip()

        if text:
            paragraphs.append({
                "paragraph_number": paragraph_number,
                "text": text
            })

    return paragraphs


def load_document(file_path):

    file_path = Path(file_path)

    if file_path.suffix.lower() == ".pdf":
        return load_pdf(file_path)

    if file_path.suffix.lower() == ".docx":
        return load_docx(file_path)

    if file_path.suffix.lower() == ".txt":
        text = file_path.read_text(
            encoding="utf-8"
        )

        return [{
            "text": text
        }]

    raise ValueError(
        f"Unsupported file type: {file_path.suffix}"
    )