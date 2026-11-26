
from pathlib import Path
import json
import re
from pypdf import PdfReader 

# -----------------------------------------
# 1. PROJECT CONFIGURATION
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENT_FOLDER = PROJECT_ROOT / "data" / "documents"

OUTPUT_FILE = PROJECT_ROOT / "data" / "chunks.json"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# -----------------------------------------
# 2. TEXT CLEANING
# -----------------------------------------

def clean_text(text: str) -> str:
    """
    Remove unnecessary whitespace from extracted text.
    """

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -----------------------------------------
# 3. TEXT CHUNKING
# -----------------------------------------

def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    chunks = []

    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        # Find a word boundary
        if end < len(text):

            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# -----------------------------------------
# 4. PDF INGESTION
# -----------------------------------------

def ingest_pdf(pdf_path: Path) -> list[dict]:

    print(f"\nProcessing PDF: {pdf_path.name}")

    reader = PdfReader(str(pdf_path))

    document_chunks = []

    chunk_number = 0

    for page_number, page in enumerate(reader.pages, start=1):

        extracted_text = page.extract_text() or ""

        extracted_text = clean_text(extracted_text)

        if not extracted_text:

            print(
                f"Warning: No extractable text on page {page_number}"
            )

            continue

        chunks = split_into_chunks(extracted_text)

        for chunk in chunks:

            chunk_number += 1

            document_chunks.append({

                "chunk_id": f"{pdf_path.stem}_{chunk_number}",

                "document_name": pdf_path.name,

                "page_number": page_number,

                "text": chunk

            })

    print(f"Pages processed: {len(reader.pages)}")

    print(f"Chunks created: {len(document_chunks)}")

    return document_chunks


# -----------------------------------------
# 5. MAIN INGESTION PROCESS
# -----------------------------------------

def main():

    print("====================================")
    print(" ENTERPRISE DOCUMENT INGESTION")
    print("====================================")

    DOCUMENT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = list(DOCUMENT_FOLDER.glob("*.pdf"))

    if not pdf_files:

        print("\nNo PDF documents found.")

        print(f"Please add PDFs to: {DOCUMENT_FOLDER}")

        return

    all_chunks = []

    for pdf_file in pdf_files:

        try:

            chunks = ingest_pdf(pdf_file)

            all_chunks.extend(chunks)

        except Exception as error:

            print(f"Failed to process {pdf_file.name}: {error}")

    # Save extracted chunks
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n====================================")
    print(" DOCUMENT INGESTION COMPLETED")
    print("====================================")

    print(f"Total PDFs: {len(pdf_files)}")

    print(f"Total chunks: {len(all_chunks)}")

    print(f"Output file: {OUTPUT_FILE}")

    # Display first chunk
    if all_chunks:

        print("\nFIRST CHUNK PREVIEW:")

        print("------------------------------------")

        print(all_chunks[0]["text"][:500])


if __name__ == "__main__":

    main()
