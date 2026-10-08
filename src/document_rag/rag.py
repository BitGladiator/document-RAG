import json
import os
from pathlib import Path
import re
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


def generate_answer_stream(query, context):

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
        temperature=0,
        stream=True
    )

    for chunk in response:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content


def generate_answer(query, context):

    return "".join(generate_answer_stream(query, context))


def ask_rag_stream(
    query,
    top_k=5,
    file_path=None
):

    results = retrieve(
        query=query,
        top_k=top_k,
        file_path=file_path
    )

    sources = [
        {
            "source_number": i,
            "metadata": result["metadata"],
            "distance": result["distance"],
            "text": result["text"]
        }
        for i, result in enumerate(results, start=1)
    ]

    if not results:
        def empty_stream():
            yield "I couldn't find the answer in the provided documents."

        return sources, empty_stream()

    context = build_context(results)
    return sources, generate_answer_stream(query=query, context=context)


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


VERIFICATION_SYSTEM_PROMPT = """You are an expert document-grounded evidence verification auditor.
Your job is to verify whether factual statements in a generated answer are supported by the provided retrieved document evidence.

CRITICAL RULES:
1. THE RETRIEVED DOCUMENTS ARE THE ONLY AUTHORITY.
2. Answer ONLY: "Is this claim supported by the provided document evidence?"
3. NEVER use general world knowledge or outside facts. Even if a claim is a known true fact in reality, if the provided document evidence does NOT explicitly support it, you must classify it as UNSUPPORTED or PARTIALLY_SUPPORTED.
4. Break the generated answer into distinct, meaningful factual claims. Do NOT over-split into tiny fragments or individual words. Do NOT include greetings, pleasantries, or procedural statements like "Based on the documents,".
5. Use EXACTLY these three statuses:
   - "SUPPORTED": The provided document evidence directly and explicitly supports the claim.
   - "PARTIALLY_SUPPORTED": The evidence supports part of the claim, but not all of it, or only mentions related concepts without directly confirming the claim.
   - "UNSUPPORTED": The provided document evidence does not support the claim.
6. For each claim, cite "source_ids" as a list of integers corresponding to the SOURCE numbers (e.g., [1], [2]) that provide the supporting evidence. Cite ONLY source numbers that actually appear in DOCUMENT EVIDENCE. If UNSUPPORTED, source_ids must be [].
7. Provide an "explanation" that clearly explains the evidence relation using ONLY the text in the sources.
8. Assess the "overall" evidence as:
   - "strong": Most important claims are supported and there are no major unsupported claims.
   - "mixed": Some claims are supported but at least one meaningful claim is only partially supported or unsupported.
   - "weak": Important claims lack supporting evidence.
9. Provide a concise "summary" of the overall evidence evaluation.

Return ONLY a valid JSON object matching this schema:
{
  "overall": "strong" | "mixed" | "weak",
  "summary": "...",
  "claims": [
    {
      "claim": "...",
      "status": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED",
      "source_ids": [1],
      "explanation": "..."
    }
  ]
}
"""


def format_evidence_for_verification(sources):
    blocks = []
    for s in sources:
        s_num = s.get("source_number", "?")
        meta = s.get("metadata") or {}
        file_name = meta.get("file_name") or meta.get("file_path", "unknown")
        clean_name = Path(str(file_name)).name if file_name else "unknown"
        page_num = meta.get("page_number")
        page_str = f"Page {page_num}" if page_num is not None else "Page N/A"
        chunk_idx = meta.get("chunk_index", "N/A")
        text = s.get("text", "").strip()

        blocks.append(
            f"SOURCE {s_num}\n"
            f"Document: {clean_name}\n"
            f"Page: {page_str}\n"
            f"Chunk: {chunk_idx}\n"
            f"Text:\n{text}\n"
        )
    return "\n".join(blocks)


def _parse_verification_json(raw_text):
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_verification_data(data, valid_source_numbers):
    if not isinstance(data, dict):
        raise ValueError("Model output is not a JSON object")

    overall = str(data.get("overall", "")).lower().strip()
    if overall not in {"strong", "mixed", "weak"}:
        overall = None

    summary = str(data.get("summary", "")).strip()

    raw_claims = data.get("claims", [])
    if not isinstance(raw_claims, list):
        raw_claims = []

    validated_claims = []
    has_unsupported = False
    has_partially = False
    has_supported = False

    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        claim_text = str(item.get("claim", "")).strip()
        if not claim_text:
            continue

        raw_status = str(item.get("status", "")).upper().replace(" ", "_")
        if "PARTIAL" in raw_status:
            status = "PARTIALLY_SUPPORTED"
            has_partially = True
        elif "UNSUPPORT" in raw_status or "NOT_SUPPORT" in raw_status:
            status = "UNSUPPORTED"
            has_unsupported = True
        elif "SUPPORT" in raw_status:
            status = "SUPPORTED"
            has_supported = True
        else:
            status = "UNSUPPORTED"
            has_unsupported = True

        raw_source_ids = item.get("source_ids", [])
        if not isinstance(raw_source_ids, list):
            raw_source_ids = []

        clean_source_ids = []
        for sid in raw_source_ids:
            try:
                sid_int = int(sid)
                if sid_int in valid_source_numbers and sid_int not in clean_source_ids:
                    clean_source_ids.append(sid_int)
            except (ValueError, TypeError):
                continue

        explanation = str(item.get("explanation", "")).strip()

        validated_claims.append({
            "claim": claim_text,
            "status": status,
            "source_ids": clean_source_ids,
            "explanation": explanation
        })

    if not overall:
        if not validated_claims or (has_unsupported and not has_supported and not has_partially):
            overall = "weak"
        elif has_unsupported or has_partially:
            overall = "mixed"
        else:
            overall = "strong"

    if not summary:
        if overall == "strong":
            summary = "The answer is well supported by the retrieved documents."
        elif overall == "mixed":
            summary = "Some claims are supported, but others have partial or missing evidence in the retrieved documents."
        else:
            summary = "The answer lacks sufficient evidence in the retrieved documents."

    return {
        "overall": overall,
        "summary": summary,
        "claims": validated_claims
    }


def verify_answer(query, answer, sources):
    if not answer or not answer.strip():
        raise ValueError("Answer is empty")

    if not sources or not isinstance(sources, list):
        return {
            "overall": "weak",
            "summary": "No source evidence was available to verify this answer.",
            "claims": []
        }

    valid_source_numbers = set()
    for s in sources:
        try:
            num = s.get("source_number")
            if num is not None:
                valid_source_numbers.add(int(num))
        except (ValueError, TypeError):
            continue

    formatted_evidence = format_evidence_for_verification(sources)

    user_content = f"""QUESTION:
{query}

ANSWER:
{answer}

DOCUMENT EVIDENCE:
{formatted_evidence}
"""

    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    messages = [
        {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"}
        )
    except Exception:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0
        )

    raw_output = response.choices[0].message.content or ""
    parsed_json = _parse_verification_json(raw_output)
    return normalize_verification_data(parsed_json, valid_source_numbers)


def create_snippet(text: str, query: str, max_chars: int = 280) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned

    query_lower = query.lower()
    text_lower = cleaned.lower()

    # 1. Search for full query phrase
    pos = text_lower.find(query_lower)

    # 2. If not found, look for first occurrence of substantial query terms
    if pos == -1:
        words = [w for w in re.findall(r"\w+", query_lower) if len(w) >= 3]
        best_pos = -1
        for w in words:
            p = text_lower.find(w)
            if p != -1 and (best_pos == -1 or p < best_pos):
                best_pos = p
        pos = best_pos

    if pos == -1:
        snippet = cleaned[:max_chars]
        last_space = snippet.rfind(" ")
        if last_space > int(max_chars * 0.7):
            snippet = snippet[:last_space]
        return snippet + "..."

    half = max_chars // 2
    start = max(0, pos - half)
    end = min(len(cleaned), pos + half)

    # Align start to word boundary
    if start > 0:
        next_space = cleaned.find(" ", start)
        if next_space != -1 and next_space < pos:
            start = next_space + 1

    # Align end to word boundary
    if end < len(cleaned):
        prev_space = cleaned.rfind(" ", start, end)
        if prev_space != -1 and prev_space > pos:
            end = prev_space

    snippet = cleaned[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(cleaned):
        snippet = snippet + "..."

    return snippet


def search_documents(query: str, file_path: str = None, limit: int = 10) -> list[dict]:
    query_str = str(query).strip()
    if not query_str:
        return []

    try:
        if collection.count() == 0:
            return []
    except Exception:
        return []

    query_embedding = embedding_model.encode(query_str).tolist()

    candidate_k = min(max(limit * 3, 25), max(collection.count(), 1))

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": candidate_k,
        "include": ["documents", "metadatas", "distances"]
    }

    if file_path:
        clean_name = Path(str(file_path)).name
        if "/" in str(file_path) or "\\" in str(file_path):
            query_kwargs["where"] = {"file_path": str(file_path)}
        else:
            query_kwargs["where"] = {"file_name": clean_name}

    try:
        results = collection.query(**query_kwargs)
    except Exception:
        if file_path:
            try:
                query_kwargs["where"] = {"file_name": Path(str(file_path)).name}
                results = collection.query(**query_kwargs)
            except Exception:
                return []
        else:
            return []

    if not results or not results["documents"] or not results["documents"][0]:
        return []

    raw_docs = results["documents"][0]
    raw_metas = results["metadatas"][0]
    raw_dists = results["distances"][0]

    query_terms = [w.lower() for w in re.findall(r"\w+", query_str) if len(w) >= 2]
    query_phrase = query_str.lower()

    candidates = []
    for doc, meta, dist in zip(raw_docs, raw_metas, raw_dists):
        # Cosine distance bounded roughly in [0, 2]
        sim = max(0.0, 1.0 - (dist / 2.0))

        # Lexical relevance boosting
        doc_lower = doc.lower()
        lex_score = 0.0
        if query_phrase in doc_lower:
            lex_score += 0.35
        if query_terms:
            matched_terms = sum(1 for t in query_terms if t in doc_lower)
            lex_score += 0.25 * (matched_terms / len(query_terms))

        final_score = min(1.0, (0.7 * sim) + min(0.3, lex_score))

        candidates.append({
            "text": doc,
            "metadata": meta,
            "distance": dist,
            "score": round(final_score, 4)
        })

    # Sort candidates by combined score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate near-identical / overlapping chunks from the same document
    selected = []
    for cand in candidates:
        cand_meta = cand["metadata"] or {}
        cand_file = cand_meta.get("file_name") or cand_meta.get("file_path", "")
        cand_chunk_idx = cand_meta.get("chunk_index")
        cand_words = set(re.findall(r"\w+", cand["text"].lower()))

        is_dup = False
        for acc in selected:
            acc_meta = acc["metadata"] or {}
            acc_file = acc_meta.get("file_name") or acc_meta.get("file_path", "")
            if acc_file != cand_file:
                continue

            acc_chunk_idx = acc_meta.get("chunk_index")
            if cand_chunk_idx is not None and acc_chunk_idx is not None:
                if abs(int(cand_chunk_idx) - int(acc_chunk_idx)) <= 1:
                    acc_words = set(re.findall(r"\w+", acc["text"].lower()))
                    overlap = len(cand_words & acc_words) / max(len(cand_words | acc_words), 1)
                    if overlap > 0.4:
                        is_dup = True
                        break

            acc_words = set(re.findall(r"\w+", acc["text"].lower()))
            overlap = len(cand_words & acc_words) / max(len(cand_words | acc_words), 1)
            if overlap > 0.65:
                is_dup = True
                break

        if not is_dup:
            selected.append(cand)
            if len(selected) >= limit:
                break

    output = []
    for item in selected:
        meta = item["metadata"] or {}
        raw_file = meta.get("file_name") or meta.get("file_path", "unknown")
        file_name = Path(str(raw_file)).name
        snippet = create_snippet(item["text"], query_str)

        output.append({
            "file_name": file_name,
            "file_path": meta.get("file_path", file_name),
            "file_type": meta.get("file_type") or Path(file_name).suffix.lower(),
            "page_number": meta.get("page_number"),
            "chunk_index": meta.get("chunk_index"),
            "snippet": snippet,
            "text": item["text"],
            "score": item["score"]
        })

    return output