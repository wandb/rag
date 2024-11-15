import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor

import weave
from scrapegraphai.graphs import SmartScraperGraph

executor = ThreadPoolExecutor()


async def run_blocking_code_in_thread(blocking_func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, blocking_func, *args, **kwargs)


graph_config = {
    "llm": {
        "api_key": os.environ["OPENAI_API_KEY"],
        "model": "openai/gpt-4o-mini",
    },
    "verbose": False,
    "headless": True,
}


@weave.op
async def get_web_info(task: str, url: str) -> str:
    smart_scraper_graph = SmartScraperGraph(
        prompt=task,
        source=url,
        config=graph_config,
    )

    # Run the pipeline
    result = await run_blocking_code_in_thread(smart_scraper_graph.run)
    return json.dumps(result, indent=2)
