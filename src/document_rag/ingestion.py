from pathlib import Path
import re

from pypdf import PdfReader
from docx import Document


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_comparison(text: str) -> str:

    text = re.sub(
        r"<(?:EOS|PAD)>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    return normalize_whitespace(text).casefold()


def remove_repeated_eos_blocks(text: str) -> str:


    if not re.search(r"<EOS>", text, flags=re.IGNORECASE):
        return text

    parts = re.split(
        r"(<EOS>)",
        text,
        flags=re.IGNORECASE,
    )

    result = []
    current_block = []
    seen_blocks = set()

    for part in parts:

        if re.fullmatch(
            r"<EOS>",
            part,
            flags=re.IGNORECASE,
        ):
            current_block.append(part)

            block = "".join(current_block)
            normalized_block = normalize_for_comparison(block)

            if normalized_block:
                if normalized_block not in seen_blocks:
                    result.append(block)
                    seen_blocks.add(normalized_block)

            current_block = []

        else:
            current_block.append(part)


    if current_block:
        result.append("".join(current_block))

    return "".join(result)


def remove_repeated_lines(text: str) -> str:


    lines = text.splitlines()

    if len(lines) < 2:
        return text

    result = []
    seen = set()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        normalized = normalize_for_comparison(line)

        word_count = len(normalized.split())

        if word_count >= 8 and normalized in seen:
            continue

        result.append(line)
        seen.add(normalized)

    return "\n".join(result)


def clean_text(text: str) -> str:


    text = re.sub(
        r"<(?:EOS|PAD)>",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = normalize_whitespace(text)

    return text


def load_pdf(file_path: Path) -> list[dict]:


    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        raw_text = page.extract_text() or ""

        if not raw_text.strip():
            continue


        text = remove_repeated_eos_blocks(raw_text)

        text = remove_repeated_lines(text)

        text = clean_text(text)

        if text:
            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                }
            )

    return pages


def load_docx(file_path: Path) -> list[dict]:


    document = Document(file_path)

    paragraphs = []

    for paragraph_number, paragraph in enumerate(
        document.paragraphs,
        start=1,
    ):
        text = paragraph.text.strip()

        if not text:
            continue

        text = clean_text(text)

        if text:
            paragraphs.append(
                {
                    "paragraph_number": paragraph_number,
                    "text": text,
                }
            )

    return paragraphs


def load_txt(file_path: Path) -> list[dict]:


    text = file_path.read_text(
        encoding="utf-8",
    )

    text = clean_text(text)

    if not text:
        return []

    return [
        {
            "text": text,
        }
    ]


def load_document(file_path: Path) -> list[dict]:


    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".docx":
        return load_docx(file_path)

    if extension == ".txt":
        return load_txt(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )