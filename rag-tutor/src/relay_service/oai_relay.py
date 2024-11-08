import asyncio
import json
import os

import websockets
from pydantic import ValidationError

from src.pipeline.docstore import HybridRetriever
from src.relay_service.models import (
    server_events,
    client_events,
    ClientEventTypes,
    ServerEventTypes,
)

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

SEARCH_RETRIEVE_TOOL = client_events.Tool(
    type="function",
    name="SearchRetrieve",
    description="A tool to search for relevant information from a knowledge engine",
    parameters=client_events.ToolParameter(
        type="object",
        properties={
            "query": client_events.ToolParameterProperty(type="string"),
            "limit": client_events.ToolParameterProperty(type="integer"),
        },
        required=["query"],
    ),
)


class OpenAIRealtimeClient:
    def __init__(self, message_callback=None):
        self.ws = None
        self.url = (
            f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        )
        self.task = None
        self.message_callback = message_callback
        self.function_call_buffers = {}
        self.retriever = HybridRetriever.load()

    async def connect(self):
        """Establish WebSocket connection"""
        self.ws = await websockets.connect(
            self.url,
            extra_headers={
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                "OpenAI-Beta": "realtime=v1",
            },
        )
        # Note: We'll wait for the voice config before sending the session update

    async def configure_session(self, voice=None):
        """Configure the session with optional voice setting"""
        config_event = client_events.SessionUpdate(
            session=client_events.Session(
                modalities=["text", "audio"],
                instructions=SYSTEM_PROMPT,
                input_audio_transcription=client_events.InputAudioTranscription(
                    model="whisper-1"
                ),
                turn_detection=None,
                voice=voice,  # Add voice to session configuration
                tools=[SEARCH_RETRIEVE_TOOL],
                tool_choice="auto",
            )
        )

        update_event = config_event.model_dump(exclude_none=True, mode="json")
        update_event["session"]["turn_detection"] = None
        update_event = json.dumps(update_event)

        await self.send(update_event)

    async def start(self):
        """Start the WebSocket client"""
        try:
            await self.connect()
            # Create the task but don't await it
            self.task = asyncio.create_task(self.receive_messages())
        except Exception as e:
            print(f"WebSocket error: {e}")
            if self.ws:
                await self.ws.close()

    async def stop(self):
        """Stop the WebSocket client"""
        if self.task:
            self.task.cancel()
        if self.ws:
            await self.ws.close()

    async def send(self, data):
        """Send data through WebSocket"""
        if self.ws:
            await self.ws.send(data)

    async def handle_function_call(
        self, function_name: str, arguments: str, call_id: str
    ):
        """Handle function calls from the LLM"""
        try:
            args = json.loads(arguments)
            if function_name == "SearchRetrieve":
                results = await self.retriever.invoke(
                    query=args.get("query"), limit=args.get("limit", 5)
                )
                formatted_response = "\n---\n".join([doc.as_str for doc in results])

                # Create the function call output item
                function_output_item = client_events.ConversationItem(
                    type="function_call_output",
                    call_id=call_id,  # Link to the original function call
                    output=formatted_response,
                )

                # Create and send the conversation item create event
                create_event = client_events.ConversationItemCreate(
                    type=ClientEventTypes.CONVERSATION_ITEM_CREATE,
                    item=function_output_item,
                )

                # Send the function output
                await self.send(create_event.model_dump_json(exclude_none=True))

                # Create and send response create event to get the assistant's response
                response_event = client_events.ResponseCreate(
                    type=ClientEventTypes.RESPONSE_CREATE
                )
                await self.send(response_event.model_dump_json(exclude_none=True))

                return formatted_response

        except Exception as e:
            print(f"Error in function call: {e}")
            error_msg = f"Error executing function: {str(e)}"

            # error_output_item = client_events.ConversationItem(
            #     type="function_call_output",
            #     call_id=call_id,  # Link to the original function call
            #     output=error_msg,
            # )
            #
            # error_event = client_events.ConversationItemCreate(
            #     type=ClientEventTypes.CONVERSATION_ITEM_CREATE,
            #     item=error_output_item,
            # )
            #
            # await self.send(error_event.model_dump_json(exclude_none=True))
            #
            # # Even in case of error, we should trigger the assistant's response
            # response_event = client_events.ResponseCreate(
            #     type=ClientEventTypes.RESPONSE_CREATE
            # )
            # await self.send(response_event.model_dump_json(exclude_none=True))

            return error_msg

    async def receive_messages(self):
        """Receive and process WebSocket messages"""
        try:
            async for message in self.ws:
                try:
                    message_data = json.loads(message)
                    parsed_event = parse_server_event(message_data)

                    match parsed_event.type:
                        case (
                            ServerEventTypes.SESSION_CREATED
                            | ServerEventTypes.SESSION_UPDATED
                            | ServerEventTypes.CONVERSATION_CREATED
                            | ServerEventTypes.CONVERSATION_ITEM_CREATED
                            | ServerEventTypes.RESPONSE_CREATED
                            | ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DONE
                            | ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DELTA
                            | ServerEventTypes.RESPONSE_AUDIO_DELTA
                            | ServerEventTypes.RESPONSE_AUDIO_DONE
                            | ServerEventTypes.RESPONSE_DONE
                            | ServerEventTypes.ERROR
                        ):
                            if self.message_callback:
                                await self.message_callback(parsed_event)

                        case (
                            ServerEventTypes.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED
                        ):
                            response_event = client_events.ResponseCreate(
                                type=ClientEventTypes.RESPONSE_CREATE
                            )
                            await self.send(
                                response_event.model_dump_json(exclude_none=True)
                            )
                            if self.message_callback:
                                await self.message_callback(parsed_event)

                        case ServerEventTypes.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA:
                            # Accumulate function call deltas
                            buffer_key = (
                                f"{parsed_event.response_id}:{parsed_event.call_id}"
                            )
                            if buffer_key not in self.function_call_buffers:
                                self.function_call_buffers[buffer_key] = ""
                            self.function_call_buffers[buffer_key] += parsed_event.delta

                        case ServerEventTypes.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
                            buffer_key = (
                                f"{parsed_event.response_id}:{parsed_event.call_id}"
                            )
                            accumulated_json = self.function_call_buffers.pop(
                                buffer_key, ""
                            )
                            # print(f"Function call arguments: {accumulated_json}")

                            # Get the function name from the message data
                            function_name = message_data.get("name")
                            if not function_name:
                                # print("Warning: No function name found in message data")
                                return

                            # Execute function and get response
                            function_response = await self.handle_function_call(
                                function_name,
                                parsed_event.arguments,
                                parsed_event.call_id,
                            )
                            print(
                                f"Function response: {function_response[:100]} ... {function_response[-100:]}"
                            )

                            if self.message_callback:
                                await self.message_callback(parsed_event)

                except ValueError as e:
                    print(f"Error parsing event: {e}")
                except Exception as e:
                    print(f"Unexpected error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except Exception as e:
            print(f"Error receiving message: {e}")


def parse_server_event(event_data: dict) -> server_events.ServerEvent:
    event_type = event_data.get("type")
    if not event_type:
        raise ValueError("Event data is missing 'type' field")

    model_class = server_events.EVENT_TYPE_TO_MODEL.get(event_type)
    if not model_class:
        raise ValueError(f"Unknown event type: {event_type}")

    try:
        return model_class(**event_data)
    except ValidationError as e:
        raise ValueError(f"Failed to parse event of type {event_type}: {str(e)}")
