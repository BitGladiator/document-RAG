import json
from pathlib import Path
from flask import Flask, jsonify, request, render_template, Response, stream_with_context
from .rag import ask_rag, ask_rag_stream
from .memory import (
    add_interaction,
    get_history,
    clear_history
)
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

    if not data or "query" not in data or not str(data["query"]).strip():

        return jsonify({
            "error": "Missing 'query'"
        }), 400

    query = str(data["query"]).strip()
    file_path = data.get("file_path") or None

    def sse_event(event_type, payload):
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

    def generate():
        accumulated_answer = []
        try:
            sources, stream = ask_rag_stream(
                query=query,
                file_path=file_path
            )

            # Send retrieved sources immediately before LLM generation starts
            yield sse_event("sources", {"sources": sources})

            # Stream LLM tokens incrementally
            for token in stream:
                accumulated_answer.append(token)
                yield sse_event("token", {"token": token})

            full_answer = "".join(accumulated_answer)

            # Persist to history only upon successful completion
            add_interaction(
                query=query,
                answer=full_answer,
                sources=sources,
                file_path=file_path
            )

            yield sse_event("done", {
                "answer": full_answer,
                "sources": sources
            })

        except Exception as e:
            yield sse_event("error", {"error": str(e)})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.get("/history")
def history():

    return jsonify({
        "history": get_history()
    })


@app.delete("/history")
def delete_history():

    clear_history()

    return jsonify({
        "message": "Chat history cleared"
    })


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