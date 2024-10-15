import warnings
from typing import Callable, Sequence

from bs4 import BeautifulSoup, Tag
from src.retrieval.models import Document, DocumentChunk
from src.retrieval.utils import cleanup_text, html_to_md, length_fn, md_to_html
from tree_sitter_languages import get_parser

warnings.filterwarnings("ignore")


def split_on_headers(html_content: str, header_tags: Sequence[str]) -> Sequence[str]:
    soup = BeautifulSoup(html_content, "html.parser")
    chunks = []
    current_chunk = []
    container = soup.body if soup.body else soup
    for element in container.children:
        if isinstance(element, Tag) and element.name.lower() in header_tags:
            if current_chunk:
                chunks.append("".join(str(e) for e in current_chunk))
                current_chunk = []
        current_chunk.append(element)
    if current_chunk:
        chunks.append("".join(str(e) for e in current_chunk))
    return chunks


def is_pre_tag(node):
    if (
        node.type == "element"
        and node.children
        and node.children[0].type == "start_tag"
        and node.children[0].children[1].text.decode("utf-8") == "pre"
    ):
        return node.text
    return None


def chunk_node(node, length_callable: Callable[[str], int] = len, max_length=256):
    pre_content = is_pre_tag(node)
    if pre_content:
        return [pre_content]
    elif length_callable(node.text.decode("utf-8")) <= max_length:
        return [node.text]
    else:
        chunks = []
        for child in node.children:
            chunks.extend(chunk_node(child, length_callable, max_length))
        return chunks


def chunk_html(
    content: str, length_callable: Callable[[str], int] = len, max_length: int = 256
):
    parser = get_parser("html")
    tree = parser.parse(content.encode("utf-8"))
    chunks = chunk_node(
        tree.root_node, length_callable=length_callable, max_length=max_length
    )
    return list(map(lambda x: x.decode("utf-8"), chunks))


def merge_chunks_with_tolerance(
    chunks: Sequence[str],
    length_callable: Callable[[str], int],
    max_length: int = 256,
    length_tolerance: int = 64,
) -> Sequence[str]:
    if not chunks:
        return []
    merged_chunks = []
    current_chunk = chunks[0]
    for next_chunk in chunks[1:]:
        combined_length = length_callable(current_chunk + next_chunk)
        if combined_length <= max_length + length_tolerance:
            current_chunk += "\n\n" + next_chunk
        else:
            merged_chunks.append(current_chunk)
            current_chunk = next_chunk
    merged_chunks.append(current_chunk)
    return merged_chunks


def chunk_doc(
    doc: Document, length_fn: Callable[[str], int], max_length: int = 256
) -> Sequence[DocumentChunk]:
    document_dict = doc.model_dump(mode="json")
    html_content = md_to_html(document_dict.pop("content"))
    if not html_content:
        return []
    parent_chunks = split_on_headers(html_content, ["h1", "h2", "h3", "h4", "h5", "h6"])
    parent_chunks = merge_chunks_with_tolerance(
        parent_chunks, length_fn, max_length * 2, length_tolerance=max_length // 2
    )
    md_chunks = [
        chunk_html(md_chunk, length_fn, max_length) for md_chunk in parent_chunks
    ]
    md_chunks = [list(map(html_to_md, md_chunk)) for md_chunk in md_chunks]
    md_chunks = [
        merge_chunks_with_tolerance(
            md_chunk, length_fn, max_length, length_tolerance=max_length // 4
        )
        for md_chunk in md_chunks
    ]
    child_chunks = []
    for md_chunk, parent_chunk in zip(md_chunks, parent_chunks):
        parent_chunk = html_to_md(parent_chunk)
        parent_chunk = cleanup_text(parent_chunk)
        for chunk in md_chunk:
            cleaned_chunk = cleanup_text(chunk)
            if len(cleaned_chunk) > 10 and len(cleaned_chunk.splitlines()) > 2:
                child_chunks.append(
                    {
                        "embed_content": cleaned_chunk,
                        "content": parent_chunk,
                        "embed_tokens": length_fn(cleaned_chunk),
                        "num_tokens": length_fn(parent_chunk),
                    }
                )
    return [
        DocumentChunk(**{"document_id": doc.id, **document_dict, **chunk})
        for chunk in child_chunks
    ]


def main():
    import json

    with open("data/arxiv_articles.jsonl", "r") as f:
        for line in f:
            doc = json.loads(line)
            doc = Document(**doc)
            chunks = chunk_doc(doc, length_fn=length_fn)
            for chunk in chunks:
                print(chunk.as_str)
                print()
                print("-" * 80)
            break


if __name__ == "__main__":
    main()
