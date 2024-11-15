import asyncio
from typing import Type

import instructor
import litellm
import weave
from litellm import acompletion
from litellm.caching import Cache, LiteLLMCacheType
from pydantic import BaseModel, Field

from src.rag_service.utils import CACHE_DIR, PROMPTS_DIR

disk_cache_dir = CACHE_DIR / "litellm"
litellm.cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(disk_cache_dir))
litellm.suppress_debug_info = True
client = instructor.from_litellm(acompletion)


class ParaphrasedQuery(BaseModel):
    paraphrased_queries: list[str] = Field(
        ...,
        title="Paraphrased Queries",
        description="A list of unique paraphrasing of the original question.",
        max_length=3,
    )


class DecomposedQuery(BaseModel):
    decomposed_queries: list[str] = Field(
        ...,
        title="Decomposed Queries",
        description="A list of unique decomposed queries.",
        max_length=3,
    )


class StepBackQuery(BaseModel):
    step_back_question: str = Field(
        ...,
        title="Step Back Queries",
        description="A more generic question that needs to be answered in order to answer the specific question.",
    )


@weave.op
async def enhance_query(
    query: str, instructions: str, model: str, response_model: Type[BaseModel]
) -> BaseModel:
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": query},
    ]

    completion = await client.chat.completions.create(
        messages=messages, response_model=response_model, model=model
    )
    return completion


@weave.op
async def paraphrase_query(query: str, model: str = "gpt-4o-mini") -> list[str]:

    instructions = (PROMPTS_DIR / "qe_paraphrase.md").read_text()
    completion = await enhance_query(query, instructions, model, ParaphrasedQuery)
    return completion.paraphrased_queries


@weave.op
async def decompose_query(query: str, model: str = "gpt-4o-mini") -> list[str]:

    instructions = (PROMPTS_DIR / "qe_decompose.md").read_text()

    completion = await enhance_query(query, instructions, model, DecomposedQuery)

    return completion.decomposed_queries


@weave.op
async def step_back_query(query: str, model: str = "gpt-4o-mini") -> list[str]:

    instructions = (PROMPTS_DIR / "qe_step_back.md").read_text()
    completion = await enhance_query(query, instructions, model, StepBackQuery)
    return completion.step_back_question


@weave.op
async def query_expansion(query: str) -> list[str]:
    paraphrased_queries = paraphrase_query(query)
    sub_questions = decompose_query(query)
    generic_question = step_back_query(query)
    paraphrased_queries, sub_questions, generic_question = await asyncio.gather(
        paraphrased_queries, sub_questions, generic_question
    )

    return [*paraphrased_queries, *sub_questions, generic_question]


if __name__ == "__main__":
    response = asyncio.run(query_expansion("What is contextual retrieval?"))
    print(response)
