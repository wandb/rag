import asyncio
import json

import litellm
import weave
from litellm import acompletion
from litellm.caching import Cache, LiteLLMCacheType

from src.rag_service.docstore import HybridRetriever
from src.rag_service.utils import CACHE_DIR, PROMPTS_DIR

disk_cache_dir = CACHE_DIR / "litellm"
litellm.cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=str(disk_cache_dir))
litellm.suppress_debug_info = True

retriever = HybridRetriever.load()

FUNCTIONS_MAP = {"SemanticSearchEngine": retriever.invoke}


@weave.op
def format_function_response(function_response):
    return "\n\n".join([doc.as_str for doc in function_response])


@weave.op
async def call_retrieval(messages, tools):
    response = await acompletion(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="required",
    )
    return response.choices[0].message.model_dump(mode="json")


@weave.op
async def handle_tool_calls(messages, tool_calls):
    for tool_call in tool_calls:
        function_name = tool_call["function"]["name"]
        function_to_call = FUNCTIONS_MAP[function_name]
        function_args = json.loads(tool_call["function"]["arguments"])
        function_response = await function_to_call(
            query=function_args.get("query"), limit=function_args.get("limit", 5)
        )
        formatted_response = format_function_response(function_response)
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": function_name,
            "content": formatted_response,
        }
        messages.append(tool_message)

    return messages


@weave.op
async def generate_answer(messages):
    response = await acompletion(
        model="gpt-4o",
        messages=messages,
    )
    return response.choices[0].message.content


@weave.op
async def ask_expert(query: str) -> str:
    instructions = (PROMPTS_DIR / "ask_expert.md").read_text()
    tools = json.load((PROMPTS_DIR / "tools.json").open())
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": query},
    ]

    initial_message = await call_retrieval(messages=messages, tools=tools)
    tool_calls = initial_message["tool_calls"]
    messages.append(initial_message)
    messages = await handle_tool_calls(messages, tool_calls)
    answer = await generate_answer(messages)
    return answer


async def main():
    query = "What is contextual retrieval?"
    response = await ask_expert(query)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
