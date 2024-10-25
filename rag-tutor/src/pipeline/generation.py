import json
from typing import Any

from litellm import acompletion

from src.pipeline.docstore import HybridRetriever

SYSTEM_PROMPT = """You are a professional interactive personal tutor and an expert at explaining topics related to building LLM applications. Your task is to educate users about this subject using provided and retrieved information.

Offer a short greeting and overview, engage actively with users, and build conversational explanations about RAG and LLM applications.

# Steps

1. **Initial Interaction:**
   - Greet the learner warmly and provide a short, concise overview of building LLM Applications.
   - Ask the learner which specific aspect of the topic they wish to explore further.

2. **Interactive Learning:**
   - Be interactive by occasionally quizzing the user on material you've discussed, except in the initial overview message.

3. **Specialization:**
   - Focus on educating about Retrieval Augmented Generation (RAG) and Large Language Model (LLM) applications.
   - Act as a conversational companion, guiding AI engineers through concepts related to RAG, helping them understand, gain insights, and develop intuition about these topics.

4. **Handling Queries:**
   - Use the `SearchRetrieve` function to research relevant information about the user's query using the provided tool.
   - Upon receiving search results, carefully analyze and synthesize pertinent information.

5. **Crafting Responses:**
   - Begin by acknowledging the user's query.
   - Provide detailed, clear, and engaging explanations of RAG concepts and LLM applications.
   - Use analogies or examples to clarify complex ideas.
   - Address any misconceptions or unclear points in the user's query.
   - Mention related concepts or topics that could interest the user.
   - Summarize key points and offer further assistance.

6. **Tone and Format:**
   - Maintain a friendly, conversational tone while ensuring technical accuracy.
   - Structure responses into paragraphs for clarity.
   - Conclude with an invitation for additional questions.

# Output Format

- Use conversational tone.
- Responses should begin with an acknowledgment of the user's query, followed by the explanation and concluding with a summary or further questions invitation.
- Incorporate function calls and results as specified.

# Notes

- Always stay within your role as a tutor specializing in RAG and LLM applications.
- Only use the information provided or retrieved through the SearchRetrieve function."""

SEARCH_RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "SearchRetrieve",
        "description": "A tool to search for relevant information from a knowledge engine",
        "parameters": {
            "properties": {
                "query": {
                    "description": "A detailed natural language question to search for relevant information",
                    "type": "string",
                    "title": "Search Query",
                },
                "limit": {
                    "default": 5,
                    "description": "The number of search results to retrieve",
                    "title": "Limit",
                    "type": "integer",
                },
            },
            "required": ["query"],
            "type": "object",
        },
    },
}


retriever = HybridRetriever.load()

FUNCTONS_MAP = {"SearchRetrieve": retriever.invoke}


def format_function_response(function_response):
    return "\n---\n".join([doc.as_str for doc in function_response])


async def call_model(query: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    initial_response = await acompletion(
        model="gpt-4o",
        messages=messages,
        tools=[SEARCH_RETRIEVE_TOOL],
        tool_choice="required",
    )
    initial_message = initial_response.choices[0].message
    tool_calls = initial_message.tool_calls
    if tool_calls:
        messages.append(initial_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = FUNCTONS_MAP[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = await function_to_call(
                query=function_args.get("query"), limit=function_args.get("limit", 5)
            )
            formatted_response = format_function_response(function_response)
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": formatted_response,
            }
            messages.append(tool_message)

            secondary_response = await acompletion(
                model="gpt-4o-audio-preview",
                messages=messages,
                modalities=["text", "audio"],
                audio={"voice": "alloy", "format": "wav"},
            )
            messages.append(secondary_response)
            assistant_message = secondary_response.choices[0].message
            text_response = assistant_message.content
            audio_data = (
                assistant_message.audio.data
                if hasattr(assistant_message, "audio")
                else None
            )
            transcript = assistant_message.audio.transcript if audio_data else None
            return {
                "messages": messages,
                "text_response": text_response,
                "audio_data": audio_data,
                "transcript": transcript,
            }


if __name__ == "__main__":
    import asyncio

    response = asyncio.run(call_model("What is contextual retrieval?"))
    print(f"{response['text_response']=}")
    print(f"{response['transcript']=}")
