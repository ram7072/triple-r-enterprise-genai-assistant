
from pathlib import Path
import json
import faiss
import numpy as np
from ollama import Client

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks.json"
INDEX_FILE = PROJECT_ROOT / "data" / "faiss.index"
METADATA_FILE = PROJECT_ROOT / "data" / "vector_metadata.json"

EMBEDDING_MODEL = "nomic-embed-text"

client = Client(host="http://localhost:11434")


def generate_embedding(text: str, task: str = "search_document"):
    response = client.embed(
        model=EMBEDDING_MODEL,
        input=f"{task}: {text}"
    )
    return response.embeddings[0]


def create_vector_store():

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}. Run ingest.py first."
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No document chunks found.")

    vectors = []

    for position, chunk in enumerate(chunks, start=1):
        print(f"Embedding {position}/{len(chunks)}")

        embedding = generate_embedding(chunk["text"])
        vectors.append(embedding)

    vector_array = np.array(vectors, dtype="float32")

    # Normalize vectors for cosine similarity using inner product
    faiss.normalize_L2(vector_array)

    dimension = vector_array.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(vector_array)

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)

    print("\nVector database created successfully!")
    print("Total vectors:", index.ntotal)
    print("Embedding dimensions:", dimension)
    print("FAISS index:", INDEX_FILE)
    print("Metadata:", METADATA_FILE)


if __name__ == "__main__":
    create_vector_store()
