
from pathlib import Path
import json
import faiss
import numpy as np
from ollama import Client

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_FILE = PROJECT_ROOT / "data" / "faiss.index"
METADATA_FILE = PROJECT_ROOT / "data" / "vector_metadata.json"

EMBEDDING_MODEL = "nomic-embed-text"

client = Client(host="http://localhost:11434")


def generate_query_embedding(question: str):

    response = client.embed(
        model=EMBEDDING_MODEL,
        input=f"search_query: {question}"
    )

    return np.array(
        response.embeddings[0],
        dtype="float32"
    ).reshape(1, -1)


def semantic_search(question: str, top_k: int = 3):

    if not INDEX_FILE.exists():
        raise FileNotFoundError("FAISS index not found. Run embed_store.py first.")

    if not METADATA_FILE.exists():
        raise FileNotFoundError("Vector metadata file not found.")

    index = faiss.read_index(str(INDEX_FILE))

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not question.strip():
        raise ValueError("Please enter a question.")

    if not metadata:
        raise ValueError("No document metadata found.")

    top_k = min(top_k, index.ntotal, len(metadata))

    query_vector = generate_query_embedding(question)

    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        chunk = metadata[idx]

        results.append({
            "score": float(score),
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"]
        })

    return results


if __name__ == "__main__":

    print("====================================")
    print(" ENTERPRISE SEMANTIC SEARCH")
    print("====================================")

    while True:

        question = input("\nEnter your question (exit to stop): ")

        if question.lower().strip() == "exit":
            break

        try:
            results = semantic_search(question, top_k=3)

            print("\nRETRIEVED DOCUMENTS")
            print("------------------------------------")

            for position, result in enumerate(results, start=1):

                print(f"\nResult {position}")
                print("Similarity:", round(result["score"], 4))
                print("Document:", result["document_name"])
                print("Page:", result["page_number"])
                print("Content:", result["text"])

        except Exception as error:
            print("Search error:", error)
