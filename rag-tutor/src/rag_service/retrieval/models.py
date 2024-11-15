from typing import Any, Literal, Optional, Sequence

from pydantic import BaseModel, computed_field

from src.rag_service.utils import generate_key

source_types = Literal["Webpage", "Paper", "Notebook", "Source Code"]


class Document(BaseModel):
    uri: str
    source_type: source_types
    content: str
    num_tokens: int
    links: Optional[Any]

    @property
    def as_str(self):
        content_str = (
            f"uri: {self.uri}\nSourceType: {self.source_type}"
            f"\n---\n\n"
            f"{self.content}"
        ).strip()

        return content_str

    @computed_field
    @property
    def id(self) -> str:
        return generate_key(self.as_str)


class DocumentChunk(BaseModel):
    document_id: str
    uri: str
    source_type: source_types
    embed_content: str
    embed_tokens: int
    content: str
    num_tokens: int
    links: Optional[Any] = None

    @property
    def as_str(self):
        content_str = (
            f"uri: {self.uri}\nSourceType: {self.source_type}"
            f"\n---\n\n"
            f"{self.content}"
        ).strip()

        return content_str

    @computed_field
    @property
    def id(self) -> str:
        return generate_key(self.as_str)


class RetrievalResults(BaseModel):
    query: str
    results: Sequence[DocumentChunk]

    @property
    def as_str(self):
        snippets_str = ""
        for idx, result in enumerate(self.results):
            snippets_str += f"\n<snippet idx={idx+1}>\n{result.as_str}\n</snippet>\n"

        content_str = (
            f"<query>\n{self.query}\b</query>\b<snippets>{snippets_str}</snippets>"
        )
        return content_str

    @computed_field
    @property
    def id(self) -> str:
        return generate_key(self.as_str)
