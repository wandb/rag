from typing import Any, Literal, Optional

from pydantic import BaseModel, computed_field

from src.retrieval.utils import generate_key

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
