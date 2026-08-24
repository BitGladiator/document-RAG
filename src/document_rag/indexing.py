from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .ingestion import load_document
from .chunking import chunk_text


CHROMA_PATH = "/workspace/data/chroma_data"
COLLECTION_NAME = "documents"

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


def index_document(file_path):

    file_path = Path(file_path)

    pages = load_document(file_path)

    documents = []
    metadatas = []
    ids = []

    chunk_index = 0

    for page in pages:

        text = page["text"]

        chunks = chunk_text(text)

        for chunk in chunks:

            documents.append(chunk)

            metadata = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix.lower(),
                "chunk_index": chunk_index,
            }

            if "page_number" in page:
                metadata["page_number"] = page["page_number"]

            if "paragraph_number" in page:
                metadata["paragraph_number"] = page[
                    "paragraph_number"
                ]

            metadatas.append(metadata)

            ids.append(
                f"{file_path.name}-{chunk_index}"
            )

            chunk_index += 1

    if not documents:
        return 0

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return len(documents)

def list_documents():
    documents = {}

    for metadata in collection.get(
        include=["metadatas"]
    )["metadatas"]:

        file_name = metadata.get("file_name")

        if file_name:
            documents[file_name] = {
                "file_name": file_name,
                "file_type": metadata.get("file_type"),
                "chunks": documents.get(
                    file_name,
                    {}
                ).get("chunks", 0) + 1
            }

    return list(documents.values())


def delete_document(file_name):

    collection.delete(
        where={
            "file_name": file_name
        }
    )

    file_path = Path(
        "/workspace/data/documents"
    ) / file_name

    if file_path.exists():
        file_path.unlink()

    return True