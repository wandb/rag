import asyncio
import json
import os

import instructor
from litellm import client, completion, speech
from pydantic import BaseModel, Field
from src.retrieval.docstore import HybridRetriever

retriever = HybridRetriever.load()
ins_client = instructor.from_litellm(completion)


SYSTEM_PROMPT = """You are an AI tutor specializing in Retrieval Augmented Generation (RAG) and Large Language Model (LLM) applications. 
Your role is to act as a conversational companion for AI engineers, providing guidance and explanations about concepts related to RAG. 
Your goal is to help users understand RAG concepts, gain insights and clarity, and develop intuition about the underlying principles their queries relate to.

When a user presents a query, follow these steps:

1. Use the `SearchRetrieve` function to research information relevant to the user's query and the underlying concepts
2. Once you receive the search results, carefully analyze and synthesize the most pertinent information. Focus on details that directly address the user's query and provide insights into RAG and LLM applications.

3. Craft a detailed response to the user's query. Your response should:
   a. Draw from the retrieved data to provide a detailed and accurate answer to the user's query
   b. Provide clear explanations of RAG concepts
   c. Offer insights into LLM applications
   d. Help the user understand the concepts in a friendly, conversational manner
   e. Use analogies or examples where appropriate to illustrate complex ideas
   f. Address any misconceptions or unclear points in the user's query

4. Present your response in a conversational tone, as if you're speaking directly to the user. Aim to make the explanation engaging and accessible, while maintaining technical accuracy.
5. If there are related concepts or topics that might be of interest to the user based on their query, briefly mention these and explain how they connect to the main topic.
6. Conclude your response by summarizing the key points and offering to clarify any remaining questions the user might have.
7. Be precise and concise and use paragraphs to structure your response. We will use this response to generate a speech output.

Begin with a brief acknowledgment of the user's query, then provide your detailed explanation. End with an invitation for further questions. For example:

<answer>
I understand you're asking about [brief restatement of the query].

[Your detailed explanation, insights, and examples here]

To sum up, [brief summary of key points]. Is there any part of this explanation you'd like me to elaborate on further?
</answer>

Remember, your goal is to help the user gain a deeper understanding of RAG and LLM concepts, so focus on providing clear, insightful, and practical information that an AI engineer would find valuable."""


class SearchRetrieve(BaseModel):
    queries: list[str] = Field(
        ...,
        title="Search Queries",
        description="A list of detailed and diverse natural language queries to search for relevant information from "
        "a knowledge engine",
        max_length=3,
    )
    limit: int = Field(
        5,
        title="Limit",
        description="The number of search results to retrieve for each query.",
    )


async def chat(query: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    response, raw_response = ins_client.chat.completions.create_with_completion(
        model="gpt-4o", messages=messages, response_model=SearchRetrieve
    )

    model_response = raw_response.choices[0].message

    model_response_dict = model_response.model_dump(mode="json")
    tool_calls = model_response_dict["tool_calls"]
    tool_call_id = tool_calls[0]["id"]
    tool_function_name = tool_calls[0]["function"]["name"]

    tasks = [
        retriever.invoke(query, limit=response.limit) for query in response.queries
    ]

    results = await asyncio.gather(*tasks)
    docs = [doc for result in results for doc in result]
    deduped_docs = {}
    for doc in docs:
        deduped_docs[doc.content] = doc
    docs = list(deduped_docs.values())
    context = json.dumps([doc.as_str for doc in docs])

    tool_message = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_function_name,
        "content": context,
    }

    messages.append(model_response)
    messages.append(tool_message)

    output = completion(
        model="gpt-4o",
        messages=messages,
    )

    answer = output.choices[0].message.content

    speech_file_path = "speech.mp3"
    response = speech(
        model="openai/tts-1",
        voice="alloy",
        input=answer,
    )
    response.stream_to_file(speech_file_path)
