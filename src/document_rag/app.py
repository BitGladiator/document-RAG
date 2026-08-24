from pathlib import Path

from flask import Flask, jsonify, request, render_template

from .rag import ask_rag
from .indexing import (
    index_document,
    list_documents,
    delete_document
)


app = Flask(__name__)

DOCUMENTS_DIR = Path(
    "/workspace/data/documents"
)

DOCUMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():

    return jsonify({
        "status": "ok"
    })


@app.post("/ask")
def ask():

    data = request.get_json()

    if not data or "query" not in data:

        return jsonify({
            "error": "Missing 'query'"
        }), 400

    result = ask_rag(
        data["query"]
    )

    return jsonify(result)


@app.post("/upload")
def upload():

    if "file" not in request.files:

        return jsonify({
            "error": "No file provided"
        }), 400

    file = request.files["file"]

    if not file.filename:

        return jsonify({
            "error": "No filename provided"
        }), 400

    allowed_extensions = {
        ".pdf",
        ".docx",
        ".txt"
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        return jsonify({
            "error": "Unsupported file type"
        }), 400

    file_path = DOCUMENTS_DIR / file.filename

    file.save(file_path)

    chunks_indexed = index_document(
        file_path
    )

    return jsonify({
        "message": "Document uploaded successfully",
        "file_name": file.filename,
        "chunks_indexed": chunks_indexed
    })
    
@app.get("/documents")
def documents():

    return jsonify({
        "documents": list_documents()
    })


@app.delete("/documents/<file_name>")
def delete(file_name):

    delete_document(file_name)

    return jsonify({
        "message": "Document deleted",
        "file_name": file_name
    })