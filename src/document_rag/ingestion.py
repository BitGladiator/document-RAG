from pathlib import Path
import re

from pypdf import PdfReader
from docx import Document


def clean_text(text):


    text = re.sub(
        r"<(?:EOS|PAD)>",
        " ",
        text,
        flags=re.IGNORECASE
    )


    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def remove_repeated_blocks(text):


    words = text.split()

    if len(words) < 20:
        return text


    for block_size in range(
        min(150, len(words) // 2),
        20,
        -1
    ):

        for start in range(
            0,
            len(words) - (block_size * 2) + 1
        ):

            first = words[
                start:start + block_size
            ]

            second_start = start + block_size

            second = words[
                second_start:
                second_start + block_size
            ]

            normalized_first = " ".join(
                first
            ).lower()

            normalized_second = " ".join(
                second
            ).lower()

            if normalized_first == normalized_second:

                del words[
                    second_start:
                    second_start + block_size
                ]

                return " ".join(words)

    return text


def load_pdf(file_path):

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        text = clean_text(text)
        text = remove_repeated_blocks(text)

        if text:
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
            text = clean_text(text)
            text = remove_repeated_sentences(text)

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

        text = clean_text(text)
        text = remove_repeated_sentences(text)

        return [{
            "text": text
        }]

    raise ValueError(
        f"Unsupported file type: {file_path.suffix}"
    )