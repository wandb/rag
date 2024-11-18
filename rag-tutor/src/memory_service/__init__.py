import json

import weave
from loguru import logger
from mem0 import Memory


class MemoryTool:
    def __init__(self):
        self.config = {
            "llm": {
                "provider": "litellm",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.2,
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "agent_memory",
                    "path": "data/cache/memory",
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "embedding_dims": 512,
                },
            },
            "version": "v1.1",
        }

        self.mem = Memory.from_config(self.config)

    @weave.op(call_display_name="AddMemory")
    def add(self, text: str, session_id: str, metadata: dict | None = None):
        result = self.mem.add(text, run_id=session_id, metadata=metadata)
        logger.info(f"Memory added: {result}")
        return json.dumps(result)

    @weave.op(call_display_name="RetrieveMemories")
    def get_memories(self, session_id: str, limit: int = 10):
        result = self.mem.get_all(run_id=session_id, limit=limit)
        return json.dumps(result)

    @weave.op(call_display_name="SearchMemory")
    def search_memories(self, query: str, session_id: str):
        result = self.mem.search(query, run_id=session_id)
        return json.dumps(result)
