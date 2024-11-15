import asyncio
import json
import os

import weave
import websockets
from loguru import logger
from pydantic import ValidationError

from src.rag_service.generation import ask_expert
from src.relay_service.models import (
    server_events,
    client_events,
    ClientEventTypes,
    ServerEventTypes,
)
from src.scrapegraph_service.web_tool import get_web_info

SYSTEM_PROMPT = open("src/relay_service/instructions.md").read().strip()

AskExpert = client_events.Tool(
    type="function",
    name="AskExpert",
    description="Ask an GenAI expert to research and explain technical concepts",
    parameters=client_events.ToolParameter(
        type="object",
        properties={
            "query": client_events.ToolParameterProperty(type="string"),
        },
        required=["query"],
    ),
)

ReadPage = client_events.Tool(
    type="function",
    name="ReadPage",
    description="An assistant tool to read a web page and extract information. Use the 'url' parameter to specify the "
    "page URL and the 'task' parameter to specify the task to be performed.",
    parameters=client_events.ToolParameter(
        type="object",
        properties={
            "task": client_events.ToolParameterProperty(type="string"),
            "url": client_events.ToolParameterProperty(type="string"),
        },
        required=["task", "url"],
    ),
)

FUNCTIONS_MAP = {"AskExpert": ask_expert, "ReadPage": get_web_info}


class OpenAIRealtimeClient:
    def __init__(self, message_callback=None):
        self.ws = None
        self.url = (
            f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        )
        self.task = None
        self.message_callback = message_callback

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
                voice=voice,
                tools=[AskExpert],
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
            if function_name in FUNCTIONS_MAP:

                @weave.op(name="execute_function")
                async def execute_function(fn_name: str, function_args: str) -> str:
                    function = FUNCTIONS_MAP.get(fn_name)
                    args = json.loads(function_args)
                    match fn_name:
                        case "AskExpert":
                            fn_output = await function(query=args.get("query"))
                        case "ReadPage":
                            fn_output = await function(
                                task=args.get("task"), url=args.get("url")
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

                # Send the function output
                await self.send(create_event.model_dump_json(exclude_none=True))

                # Create and send response create event to get the assistant's response
                response_event = client_events.ResponseCreate(
                    type=ClientEventTypes.RESPONSE_CREATE
                )
                await self.send(response_event.model_dump_json(exclude_none=True))

                return fn_response

        except Exception as e:
            print(f"Error in function call: {e}")
            error_msg = f"Error executing function: {str(e)}"

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
                                await self.message_callback(parsed_event, message_data)

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
                                await self.message_callback(parsed_event, message_data)

                        case ServerEventTypes.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
                            function_name = message_data.get("name")
                            if self.message_callback:
                                await self.message_callback(parsed_event, message_data)

                            if not function_name:
                                logger.warning("No function name found in message data")
                                return

                            function_response = await self.handle_function_call(
                                function_name,
                                parsed_event.arguments,
                                parsed_event.call_id,
                            )
                            logger.debug(
                                f"Function response: {function_response[:100]} ... {function_response[-100:]}"
                            )

                            if self.message_callback:
                                await self.message_callback(
                                    parsed_event, function_response
                                )

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
