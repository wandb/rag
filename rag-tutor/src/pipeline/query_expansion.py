import asyncio

import instructor
import litellm
import weave
from litellm import acompletion
from litellm.caching import Cache
from pydantic import BaseModel, Field

litellm.cache = Cache(type="disk", disk_cache_dir="data/cache/litellm")
client = instructor.from_litellm(acompletion)


@weave.op
async def paraphrase_query(query: str) -> list[str]:
    class PraphrasedQuery(BaseModel):
        paraphrased_queries: list[str] = Field(
            ...,
            title="Paraphrased Queries",
            description="A list of unique paraphrasing of the original question.",
            max_length=3,
        )

    SYSTEM = (
        "You are an expert at paraphrasing user questions into database queries. Your task is to perform query "
        "expansion on a given user question. This expanded query will be used to search a database of tutorials, "
        "cookbooks, and blogs about software and libraries for building LLM-powered RAG applications.\n"
        "Guidelines for query expansion:\n"
        "1. If there are multiple common ways of phrasing the user's question, include these variations.\n"
        "2. Include common synonyms for key words in the question.\n"
        "3. Do not try to rephrase or expand acronyms or words you are not familiar with.\n"
        "4. Ensure that the expanded queries maintain the original intent and meaning of the user's question.\n\n"
        "Examples\n:"
        "<example>\n"
        "User question: How do I implement vector search in a RAG system?\n"
        "<expanded_queries>\n"
        "1. How to implement vector search in a RAG system\n"
        "2. Implementing vector search for retrieval augmented generation\n"
        "3. Vector search techniques for RAG applications\n"
        "4. Best practices for vector search in LLM-powered RAG systems\n"
        "5. Integrating vector search in retrieval augmented generation pipelines\n"
        "</expanded_queries>"
        "</example>\n"
        "<example>\n"
        "User question: What are the best practices for prompt engineering?\n"
        "<expanded_queries>\n"
        "1. Best practices for prompt engineering.\n"
        "2. Effective prompt engineering techniques\n"
        "3. prompt engineering guidelines\n"
        "4. Optimizing prompts for LLM applications\n"
        "5. Tips for crafting prompts in RAG projects\n"
        "</expanded_queries>\n"
        "</example>"
        "Please provide 3-5 expanded queries based on the user's question. Ensure that each query is a unique and "
        "meaningful variation that could potentially yield different but relevant results from the database. If the "
        "original question is already well-formed and specific, you may provide fewer variations."
    )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": query},
    ]

    response = await client.chat.completions.create(
        messages=messages, response_model=PraphrasedQuery, model="gpt-4o-mini"
    )
    return response.paraphrased_queries


@weave.op
async def decompose_query(query: str) -> list[str]:
    class DecomposedQuery(BaseModel):
        decomposed_queries: list[str] = Field(
            ...,
            title="Decomposed Queries",
            description="A list of unique decomposed queries.",
            max_length=3,
        )

    SYSTEM = (
        "You are an expert at decomposing user questions into database queries. Your task is to perform query "
        "decomposition on a given user question. This involves breaking down the original question into distinct "
        "sub-questions that need to be answered in order to fully address the original query.\n"
        "To decompose this question:\n"
        "1. Carefully analyze the user question to identify its main components and any implicit sub-questions.\n"
        "2. Break down the question into smaller, more specific sub-questions that, when answered together, "
        "will provide a comprehensive answer to the original question.\n"
        "3. Ensure that each sub-question is distinct and focuses on a specific aspect of the original question.\n"
        "4. If there are any acronyms or terms in the question that you're not familiar with, do not try to rephrase "
        "or explain them. Include them in the sub-questions as they are.\n"
        "5. Keep the sub-questions concise and to the point.\n\n"
        "Now, please provide the decomposed sub-questions for the given user question."
    )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": query},
    ]

    response = await client.chat.completions.create(
        messages=messages, response_model=DecomposedQuery, model="gpt-4o-mini"
    )
    return response.decomposed_queries


@weave.op
async def step_back_query(query: str) -> list[str]:
    class StepBackQuery(BaseModel):
        step_back_question: str = Field(
            ...,
            title="Step Back Queries",
            description="A more generic question that needs to be answered in order to answer the specific question.",
        )

    SYSTEM = (
        "You are an expert at stepping back from specific user questions to identify the underlying intent and "
        "generate more generic questions that address the core principles and concepts. Your task is to take a "
        "specific question related to building LLM-powered RAG (Retrieval-Augmented Generation) applications and "
        "create a broader, more generic question that captures the fundamental concepts and intent behind the user's "
        "query.\n"
        "To generate a more generic question:\n"
        "1. Identify the key concepts and technologies mentioned in the user's question.\n"
        "2. Consider the underlying principles and broader topics that encompass these concepts.\n"
        "3. Formulate a question that addresses these broader concepts while remaining focused and technical.\n"
        "4. Ensure the generic question is concise and to the point.\n"
        "5. Do not rephrase or attempt to explain any acronyms or technical terms you're unfamiliar with.\n\n"
        "Your output should be a single, concise question that captures the essence of the user's query in a more "
        "general context. Write your generic question inside <generic_question> tags."
        "Here are two examples:\n\n"
        "Example 1:\n"
        "<user_question>How do I implement semantic search using FAISS and sentence transformers in a RAG "
        "pipeline?</user_question>\n"
        "<generic_question>What are the key components and techniques for implementing efficient vector similarity "
        "search in RAG systems?</generic_question>\n\n"
        "Example 2:\n"
        "<user_question>What's the best way to handle context window limitations when using GPT-3.5 for long document "
        "summarization in a RAG setup?</user_question>\n"
        "<generic_question>How can large language models effectively process and summarize long documents within "
        "context window constraints?</generic_question>\n\n"
        "Now, please provide your generic question based on the given user question:"
    )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": query},
    ]

    response = await client.chat.completions.create(
        messages=messages, response_model=StepBackQuery, model="gpt-4o-mini"
    )
    return response.step_back_question


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
