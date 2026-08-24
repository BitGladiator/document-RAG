from flask import Flask, jsonify, request

from .rag import ask_rag


app = Flask(__name__)


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

    query = data["query"]

    result = ask_rag(query)

    return jsonify(result)