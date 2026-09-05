import os

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "/workspace/data/chroma_data"
COLLECTION_NAME = "documents"

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def retrieve(
    query,
    top_k=5,
    max_distance=2.0,
    file_path=None
):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    query_kwargs = {
    "query_embeddings": [query_embedding],
    "n_results": top_k,
    "include": [
        "documents",
        "metadatas",
        "distances"
    ]
    }

    if file_path:
        query_kwargs["where"] = {
          "file_path": file_path
        }

    results = collection.query(**query_kwargs)

    retrieved = []

    for i in range(len(results["documents"][0])):

       distance = results["distances"][0][i]

       if file_path or distance <= max_distance:

        retrieved.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": distance
        })

    return retrieved

def build_context(results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        metadata = result["metadata"]

        file_path = metadata.get(
            "file_path",
            "unknown"
        )

        chunk_index = metadata.get(
            "chunk_index",
            "unknown"
        )

        context_parts.append(
            f"""
[Source {i}]

File: {file_path}
Chunk: {chunk_index}

Content:
{result["text"]}
"""
        )

    return "\n".join(context_parts)


def generate_answer(query, context):

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

Rules:

1. Do not use outside knowledge.
2. If the answer cannot be found in the context, say:
   "I couldn't find the answer in the provided documents."
3. Do not invent facts.
4. Cite factual claims using [Source N].
5. Only cite sources that actually appear in the context.
6. Do not invent or guess source numbers.
7. Keep the answer clear and concise.

Context:
{context}

User question:
{query}

Answer:
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


def ask_rag(
    query,
    top_k=5,
    file_path=None
):

    results = retrieve(
    query=query,
    top_k=top_k,
    file_path=file_path
)

    if not results:
        return {
            "answer": (
                "I couldn't find the answer in the provided documents."
            ),
            "sources": []
        }

    context = build_context(results)

    answer = generate_answer(
        query=query,
        context=context
    )

    return {
        "answer": answer,
        "sources": [
            {
                "source_number": i,
                "metadata": result["metadata"],
                "distance": result["distance"],
                "text": result["text"]
            }
            for i, result in enumerate(results, start=1)
        ]
    }