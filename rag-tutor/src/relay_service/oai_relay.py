import asyncio
import base64
import io
import json
import os
from datetime import datetime
from typing import Any

import weave
import websockets
from loguru import logger
from pydantic import ValidationError
from pydub import AudioSegment

from src.memory_service import MemoryTool
from src.rag_service.generation import ask_expert
from src.relay_service.models import (
    server_events,
    client_events,
    ClientEventTypes,
    ServerEventTypes,
)
from src.scrapegraph_service import get_web_info
from src.weave_logging import log_to_weave

SYSTEM_PROMPT = open("src/relay_service/instructions.md").read().strip()

AskExpert = client_events.Tool(
    type="function",
    name="AskExpert",
    description="An expert in Generative AI and LLM applications ",
    parameters=client_events.ToolParameter(
        type="object",
        properties={
            "query": client_events.ToolParameterProperty(
                type="string",
                description="A detailed question to research relevant information",
            ),
        },
        required=["query"],
    ),
)

ReadPage = client_events.Tool(
    type="function",
    name="ReadPage",
    description="Reads a web page and extract information for a specified task.",
    parameters=client_events.ToolParameter(
        type="object",
        properties={
            "task": client_events.ToolParameterProperty(
                type="string", description="The 'URL' of the Webpage to read."
            ),
            "url": client_events.ToolParameterProperty(
                type="string",
                description="Specify the task to be performed when reading the webpage.",
            ),
        },
        required=["task", "url"],
    ),
)

AddMemory = client_events.Tool(
    type="function",
    name="AddMemory",
    description="Store memories related to the conversation and the user",
    parameters=client_events.ToolParameter(
        type="object",
        properties={
            "memory": client_events.ToolParameterProperty(
                type="string",
                description="The memory to store. Should be descriptive and relevant to the conversation",
            ),
        },
        required=["memory"],
    ),
)

SearchMemory = client_events.Tool(
    type="function",
    name="SearchMemory",
    description="Search for memories related to the conversation and the user",
    parameters=client_events.ToolParameter(
        type="object",
        properties={
            "query": client_events.ToolParameterProperty(
                type="string",
                description="The query to search for in the memory",
            ),
        },
        required=["query"],
    ),
)

RetrieveMemories = client_events.Tool(
    type="function",
    name="RetrieveMemories",
    description="Retrieve all memories related to the conversation and the user",
    parameters=client_events.ToolParameter(
        type="object",
        properties={
            "limit": client_events.ToolParameterProperty(
                type="integer",
                description="The number of memories to retrieve. Defaults to the last 10 memories",
            )
        },
        required=[],
    ),
)

mem = MemoryTool()

FUNCTION_MAP = {
    "AskExpert": ask_expert,
    "ReadPage": get_web_info,
    "AddMemory": mem.add,
    "SearchMemory": mem.search_memories,
    "RetrieveMemories": mem.get_memories,
}


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


class OpenAIRealtimeRelay(weave.Model):

    ws: Any | None = None
    url: str = (
        f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    )
    task: Any | None = None
    client_send: Any | None = None
    assistant_message_created: Any | None = None
    send_conversation_message: Any | None = None
    update_conversation_message: Any | None = None
    send_function_call_message: Any | None = None
    session_id: str | None = None

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
                voice=voice,
                tools=[AskExpert, ReadPage, AddMemory, SearchMemory, RetrieveMemories],
                tool_choice="auto",
            )
        )

        update_event = config_event.model_dump(exclude_none=True, mode="json")
        update_event["session"]["turn_detection"] = None
        update_event = json.dumps(update_event)

        await self.send(update_event)

    async def _start(self, client_send):
        """Start the WebSocket client"""
        try:
            self.ws = await websockets.connect(
                self.url,
                extra_headers={
                    "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                    "OpenAI-Beta": "realtime=v1",
                },
            )
            self.client_send = client_send
            # Create the task but don't await it
            self.task = asyncio.create_task(self.invoke())
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if self.ws:
                await self.ws.close()

    async def on_connect(self, send):
        try:
            await self._start(send)
        except Exception as e:
            logger.error(f"Error sending initial message: {e}")

    async def _stop(self):
        """Stop the WebSocket client"""
        if self.task:
            self.task.cancel()
        if self.ws:
            await self.ws.close()

    async def on_disconnect(self):
        await self._stop()

    async def send(self, data: str):
        """Send data through WebSocket"""
        if self.ws:
            await self.ws.send(data)

    @weave.op(name="handle_function_call")
    async def handle_function_call(
        self, function_name: str, arguments: str, call_id: str
    ):
        """Handle function calls from the LLM"""
        try:
            if function_name in FUNCTION_MAP:

                @weave.op(name="execute_function")
                async def execute_function(fn_name: str, function_args: str) -> str:
                    function = FUNCTION_MAP.get(fn_name)
                    args = json.loads(function_args)
                    match fn_name:
                        case "AskExpert":
                            fn_output = await function(query=args.get("query"))
                        case "ReadPage":
                            fn_output = await function(
                                task=args.get("task"), url=args.get("url")
                            )
                        case "AddMemory":
                            fn_output = function(
                                text=args.get("memory"), session_id=self.session_id
                            )
                        case "SearchMemory":
                            fn_output = function(
                                query=args.get("query"), session_id=self.session_id
                            )
                        case "RetrieveMemories":
                            fn_output = function(
                                session_id=self.session_id, limit=args.get("limit", 10)
                            )
                        case _:
                            fn_output = "Function not found"
                    return fn_output

                fn_response = await execute_function(function_name, arguments)
                # Create the function call output item
                function_output_item = client_events.ConversationItem(
                    type="function_call_output",
                    call_id=call_id,
                    output=fn_response,
                )

                # Create and send the conversation item create event
                create_event = client_events.ConversationItemCreate(
                    type=ClientEventTypes.CONVERSATION_ITEM_CREATE,
                    item=function_output_item,
                )
                log_to_weave(create_event)
                # Send the function output
                await self.send(create_event.model_dump_json(exclude_none=True))

                # Create and send response create event to get the assistant's response
                response_event = client_events.ResponseCreate(
                    type=ClientEventTypes.RESPONSE_CREATE
                )
                await self.send(response_event.model_dump_json(exclude_none=True))
                log_to_weave(response_event)
                return fn_response

        except Exception as e:
            # Create the function call output item
            function_output_item = client_events.ConversationItem(
                type="function_call_output",
                call_id=call_id,
                output=f"Error executing function: {str(e)}",
            )

            create_event = client_events.ConversationItemCreate(
                type=ClientEventTypes.CONVERSATION_ITEM_CREATE,
                item=function_output_item,
            )
            log_to_weave(create_event)
            # Send the function output
            await self.send(create_event.model_dump_json(exclude_none=True))

            # Create and send response create event to get the assistant's response
            response_event = client_events.ResponseCreate(
                type=ClientEventTypes.RESPONSE_CREATE
            )
            await self.send(response_event.model_dump_json(exclude_none=True))
            log_to_weave(response_event)
            logger.error(f"Error in function call: {e}")
            error_msg = f"Error executing function: {str(e)}"
            return error_msg

    async def handle_input_audio(self, audio_data: str) -> None:
        """
        Process audio data and send it to OpenAI client

        Args:
            audio_data: Base64 encoded WAV data
        """

        # Decode base64 WAV data
        wav_data = base64.b64decode(audio_data)

        # Load into pydub and convert to mono
        audio = AudioSegment.from_wav(io.BytesIO(wav_data))
        # Resample to 24kHz mono pcm16
        pcm_audio = (
            audio.set_frame_rate(24000).set_channels(1).set_sample_width(2).raw_data
        )
        # Encode to base64 string
        pcm_base64 = base64.b64encode(pcm_audio).decode()
        # Create the audio buffer append event
        audio_event = client_events.InputAudioBufferAppend(
            type=ClientEventTypes.INPUT_AUDIO_BUFFER_APPEND, audio=pcm_base64
        )
        # Send audio buffer
        await self.send(audio_event.model_dump_json(exclude_none=True))

        # Send commit event
        commit_event = client_events.InputAudioBufferCommit(
            type=ClientEventTypes.INPUT_AUDIO_BUFFER_COMMIT
        )
        await self.send(commit_event.model_dump_json(exclude_none=True))

        log_to_weave(audio_event)

    async def handle_input_text(self, text_data: str) -> None:
        """Process text input and send it to OpenAI client"""
        # Create content for the conversation item
        content = client_events.MessageContent(type="input_text", text=text_data)

        # Create the conversation item
        conversation_item = client_events.ConversationItem(
            type="message", role="user", content=[content]
        )

        # Create the conversation item create event
        create_event = client_events.ConversationItemCreate(
            type=ClientEventTypes.CONVERSATION_ITEM_CREATE, item=conversation_item
        )

        # Send the event to OpenAI
        await self.send(create_event.model_dump_json(exclude_none=True))

        log_to_weave(create_event)

        # Create and send response create event to get the assistant's response
        response_event = client_events.ResponseCreate(
            type=ClientEventTypes.RESPONSE_CREATE
        )
        await self.send(response_event.model_dump_json(exclude_none=True))
        await self.send_conversation_message(self.client_send, "1", "User", text_data)
        log_to_weave(response_event)

    async def process_message(self, data: dict) -> None:
        """Process incoming WebSocket messages"""
        msg_type = data.get("type")

        match msg_type:
            case "voice_config":
                if self.ws and (voice := data.get("voice")):
                    await self.configure_session(voice=voice)
            case "cancel":
                if hasattr(self, "current_task"):
                    self.current_task.cancel()

                # Send a response.cancel event to OpenAI
                cancel_event = client_events.ResponseCancel(
                    type=ClientEventTypes.RESPONSE_CANCEL
                )
                await self.send(cancel_event.model_dump_json(exclude_none=True))

                log_to_weave(cancel_event)

            case "audio":
                if audio_data := data.get("data"):
                    try:
                        await self.handle_input_audio(audio_data)
                    except Exception as e:
                        logger.error(f"Failed to process audio: {str(e)}")

            case "text":
                if text_data := data.get("data"):
                    try:
                        await self.handle_input_text(text_data)
                    except Exception as e:
                        logger.error(f"Failed to process text: {str(e)}")

            case _:
                logger.error(
                    f"Unknown Event: Received unknown message type: {msg_type}"
                )

    @weave.op(call_display_name=f"Conversation-{datetime.now()}")
    async def invoke(self):
        """Receive and process WebSocket messages"""
        try:
            async for message in self.ws:
                try:
                    message_data = json.loads(message)
                    parsed_event = parse_server_event(message_data)

                    match parsed_event.type:
                        case ServerEventTypes.SESSION_CREATED:
                            self.session_id = parsed_event.event_id
                            log_to_weave(parsed_event)

                        case ServerEventTypes.SESSION_UPDATED:
                            log_to_weave(parsed_event)

                        case ServerEventTypes.CONVERSATION_ITEM_CREATED:
                            log_to_weave(parsed_event)

                        case ServerEventTypes.RESPONSE_CREATED:
                            # Relay the response.created event to the client
                            await self.client_send(
                                json.dumps(
                                    {
                                        "type": "response.created",
                                        "response_id": parsed_event.response.id,
                                    }
                                )
                            )
                            log_to_weave(parsed_event)

                        case ServerEventTypes.RESPONSE_AUDIO_DELTA:
                            if parsed_event.delta is not None:
                                audio_message = json.dumps(
                                    {
                                        "type": "audio",
                                        "data": parsed_event.delta,
                                    }
                                )
                                await self.client_send(audio_message)
                                log_to_weave(parsed_event)

                        case ServerEventTypes.RESPONSE_AUDIO_DONE:
                            # Relay the audio completion event to the client
                            await self.client_send(
                                json.dumps(
                                    {
                                        "type": "response.audio.done",
                                        "response_id": parsed_event.response_id,
                                    }
                                )
                            )

                            log_to_weave(parsed_event)

                        case ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DELTA:
                            # Check if the message item already exists, if not, create it
                            if (
                                not hasattr(self, "assistant_message_created")
                                or not self.assistant_message_created
                            ):
                                await self.send_conversation_message(
                                    self.client_send,
                                    parsed_event.item_id,
                                    "Assistant",
                                    "",
                                )
                                self.assistant_message_created = True
                            await self.update_conversation_message(
                                self.client_send,
                                parsed_event.item_id,
                                parsed_event.delta,
                            )

                        case ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DONE:
                            self.assistant_message_created = False
                            log_to_weave(parsed_event)

                        case (
                            ServerEventTypes.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED
                        ):
                            await self.send_conversation_message(
                                self.client_send,
                                parsed_event.item_id,
                                "User",
                                parsed_event.transcript,
                            )
                            response_event = client_events.ResponseCreate(
                                type=ClientEventTypes.RESPONSE_CREATE
                            )
                            await self.send(
                                response_event.model_dump_json(exclude_none=True)
                            )
                            log_to_weave(parsed_event)

                        case ServerEventTypes.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
                            function_name = message_data.get("name")
                            if not function_name:
                                logger.warning("No function name found in message data")
                                return

                            await self.send_function_call_message(
                                self.client_send,
                                parsed_event.item_id,
                                "Assistant",
                                f"{message_data.get('name')}({parsed_event.arguments})",
                            )
                            log_to_weave(parsed_event)

                            function_response = await self.handle_function_call(
                                function_name,
                                parsed_event.arguments,
                                parsed_event.call_id,
                            )

                            if function_name == "ReadPage":
                                function_response = (
                                    f"```json\n{function_response}\n```",
                                )

                            await self.send_function_call_message(
                                self.client_send,
                                parsed_event.item_id,
                                "Function",
                                function_response,
                            )

                        case ServerEventTypes.RESPONSE_DONE:
                            # Relay the response.done event to the client
                            log_to_weave(parsed_event)
                        case ServerEventTypes.ERROR:
                            log_to_weave(parsed_event)

                except ValueError as e:
                    logger.error(f"Error parsing event: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.error("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
