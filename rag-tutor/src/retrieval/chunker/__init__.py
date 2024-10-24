from typing import Callable, Sequence

from src.retrieval.chunker.code_chunker import chunk_code
from src.retrieval.chunker.md_chunker import chunk_doc as chunk_web_doc
from src.retrieval.models import Document, DocumentChunk
from src.retrieval.utils import length_fn


def chunk_doc(
    doc: Document, length_fn: Callable[[str], int], max_length: int = 256
) -> Sequence[DocumentChunk]:
    if not doc.content:
        return []
    if doc.source_type == "Source Code":
        chunks = chunk_code(doc, length_fn)
    else:
        chunks = chunk_web_doc(doc, length_fn, max_length)
    return chunks


def main():
    import json

    with open("data/cookbook_docs.jsonl", "r") as f:
        for line in f:
            doc = json.loads(line)
            doc = Document(**doc)
            chunks = chunk_doc(doc, length_fn=length_fn)
            for chunk in chunks:
                print(chunk.as_str)
                print()
                print("-" * 80)
            break
