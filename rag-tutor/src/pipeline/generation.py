import asyncio
import json
import io
import wave
import base64

import litellm
from litellm import acompletion
from litellm.caching import Cache

litellm.cache = Cache(type="disk", disk_cache_dir="data/cache/litellm")
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


async def call_model(query: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    # First call remains the same since we need the tool calls
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

            # Now stream the second call
            stream_response = await acompletion(
                model="gpt-4o-audio-preview",
                messages=messages,
                modalities=["text", "audio"],
                audio={"voice": "alloy", "format": "pcm16"},
                stream=True,
            )

            # Return an async generator
            async def response_generator():
                text_chunks = []

                async for chunk in stream_response:
                    if not hasattr(chunk.choices[0], "delta"):
                        continue

                    delta = chunk.choices[0].delta

                    # Handle text content
                    if hasattr(delta, "content") and delta.content:
                        response_chunk = {"type": "text", "content": delta.content}
                        yield response_chunk
                        text_chunks.append(delta.content)

                    # Handle audio content
                    if hasattr(delta, "audio") and delta.audio:
                        audio_delta = delta.audio

                        if audio_delta.get("data"):
                            # Convert PCM to WAV format
                            wav_buffer = io.BytesIO()
                            with wave.open(wav_buffer, "wb") as wav_file:
                                wav_file.setnchannels(1)  # mono
                                wav_file.setsampwidth(2)  # 16-bit
                                wav_file.setframerate(24000)  # 24kHz

                                # Decode base64 PCM data
                                pcm_data = base64.b64decode(audio_delta["data"])
                                wav_file.writeframes(pcm_data)

                            # Get WAV data and encode as base64
                            wav_data = base64.b64encode(wav_buffer.getvalue()).decode(
                                "utf-8"
                            )

                            response_chunk = {
                                "type": "audio",
                                "data": wav_data,
                            }
                            yield response_chunk

                        # Handle transcript if present
                        if audio_delta.get("transcript"):
                            response_chunk = {
                                "type": "transcript",
                                "content": audio_delta["transcript"],
                            }
                            yield response_chunk

                # Send complete message at the end
                if text_chunks:
                    yield {"type": "complete", "content": "".join(text_chunks)}

            return response_generator()


async def main():
    import base64
    from pydub import AudioSegment

    query = "What is contextual retrieval?"
    response_generator = await call_model(query)

    # Initialize accumulators
    audio_data = bytearray()
    transcript = ""

    async for chunk in response_generator:
        if chunk["type"] == "text":
            print(f"Text: {chunk['content']}")
        elif chunk["type"] == "audio":
            # Accumulate audio data if present
            if chunk["data"]:
                decoded_audio = base64.b64decode(chunk["data"])
                audio_data.extend(decoded_audio)

            # Accumulate transcript if present
            if chunk["transcript"]:
                transcript += chunk["transcript"]
        elif chunk["type"] == "complete":
            print(f"Complete response: {chunk['content']}")

    # After accumulating all audio data, save to file
    if audio_data:
        # Convert PCM16 to MP3

        # Create AudioSegment from raw PCM data
        # Assuming 16-bit PCM, mono, 24000Hz (common for speech)
        audio_segment = AudioSegment(
            data=audio_data,
            sample_width=2,  # 16-bit = 2 bytes
            frame_rate=24000,
            channels=1,
        )

        # Export as MP3
        audio_segment.export("speech.mp3", format="mp3")

    # Print accumulated transcript
    if transcript:
        print(f"\nFull transcript: {transcript}")


if __name__ == "__main__":
    asyncio.run(main())
