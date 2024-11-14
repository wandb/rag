import asyncio
import json
from datetime import datetime

from fasthtml.common import *
from loguru import logger
from pydub import AudioSegment

from src.components import create_layout
from src.relay_service.models import (
    ClientEventTypes,
    ServerEvent,
    ServerEventTypes,
)
from src.relay_service.models import client_events
from src.relay_service.oai_relay import OpenAIRealtimeClient

static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))

app, rt = fast_app(
    static_path=os.path.dirname(static_dir),
    # live=True,
    pico=False,
    htmx=True,
    ws_hdr=True,
    exts="ws",
    hdrs=(
        Script(src="https://cdn.tailwindcss.com"),
        Meta(
            name="viewport",
            content="width=device-width, height=device-height, initial-scale=1.0",
        ),
        Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;500;600&display=swap",
        ),
        Script(src="/static/js/index.js", type="module"),
        MarkdownJS(".markdown"),
    ),
)

# Store start time for event timestamps
start_time = None


@rt("/")
def get():
    return create_layout()


@rt("/connect")
def post():
    global start_time
    start_time = datetime.now()

    return create_layout(
        button_text="disconnect",
        disabled=False,
        include_audio_stream=True,
        ws_props={
            "hx_ext": "ws",
            "ws_connect": "/wscon",
            "ws_send": True,
            "hx_trigger": "audioMessage, textMessage",
            "hx_swap_oob": "outerHTML",
        },
    )


@rt("/disconnect")
def post():
    global start_time
    start_time = None

    return create_layout(
        button_text="connect",
        audio_player_disabled=True,
        ws_props={"hx_swap_oob": "true", "hx_swap": "delete"},
    )


def send_event_log(event_type: str, event_data: str):
    """Utility function to send event log updates via HTMX"""
    logger.info(f"{event_type}: {event_data}")


async def send_conversation_message(send, item_id: str, origin: str, message: str = ""):
    """Utility function to send conversation updates via HTMX"""
    await send(
        Div(
            Div(
                Div(
                    origin,
                    cls="mx-4 w-1/12 text-xs font-sans font-light text-[#ffcc33] text-left hover:text-sm",
                ),
                Div(
                    message,
                    cls=f"mx-1 flex-1 p-1 text-xs font-sans font-light text-[#FFFFFF] "
                    f"text-wrap text-left whitespace-pre markdown hover:text-sm",
                    id=item_id,
                ),
                cls="flex flex-row content-start",
            ),
            cls="flex flex-col h-full flex-initial grow-0 overscroll-auto max-h-max gap-2 bg-[#1A1D24] overflow-y-auto",
            id="conversation-content",
            hx_swap_oob="beforeend",
        )
    )


async def send_function_call_message(
    send, item_id: str, origin: str, message: str = ""
):
    """Utility function to send function call updates via HTMX"""
    await send(
        Div(
            Div(
                Div(
                    origin,
                    cls="mx-4 w-1/12 text-xs font-sans font-light text-[#ffcc33] text-left hover:text-sm",
                ),
                Div(
                    message,
                    cls=f"mx-1 flex-1 p-1 text-xs font-sans font-light text-[#FFFFFF] "
                    f"text-wrap text-left whitespace-pre markdown hover:text-sm",
                    id=item_id,
                ),
                cls="flex flex-row content-start",
            ),
            cls="flex flex-col gap-1",
            id="function-call-content",
            hx_swap_oob="beforeend",
        )
    )


async def update_conversation_message(
    send, item_id: str, delta: str, is_error: bool = False
):
    """Utility function to update conversation message with deltas via HTMX"""
    await send(
        Div(
            delta,
            # cls=f"event-data{' error' if is_error else ''}",
            id=item_id,
            hx_swap_oob="beforeend",
        )
    )


class OpenAIMessageHandler:
    def __init__(self):
        self.assistant_message_created = None
        self.openai_client = None

    async def relay_openai_message(
        self, send, parsed_event: ServerEvent, message: dict | str | None = None
    ):
        """
        Relay OpenAI messages to the client with appropriate UI updates.

        Args:
            send: Websocket send function
            parsed_event: Parsed server event from OpenAI
            message: Optional message data to relay to the client
        """
        match parsed_event.type:
            case ServerEventTypes.SESSION_CREATED:
                send_event_log(
                    "Session Created", f"Session ID: {parsed_event.event_id}"
                )

            case ServerEventTypes.SESSION_UPDATED:
                send_event_log(
                    "Session Updated", f"Session ID: {parsed_event.event_id}"
                )

            case ServerEventTypes.CONVERSATION_CREATED:
                send_event_log(
                    "Conversation Started",
                    f"Conversation ID: {parsed_event.conversation.id}",
                )

            case ServerEventTypes.CONVERSATION_ITEM_CREATED:
                send_event_log(
                    "Conversation Item Created",
                    f"Conversation Item ID: {parsed_event.event_id}",
                )

            case ServerEventTypes.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
                await send_conversation_message(
                    send, parsed_event.item_id, "User", parsed_event.transcript
                )

            case ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DONE:
                # Reset the flag for the next message
                self.assistant_message_created = False

            case ServerEventTypes.RESPONSE_AUDIO_DELTA:
                if parsed_event.delta is not None:
                    audio_message = json.dumps(
                        {
                            "type": "audio",
                            "data": parsed_event.delta,
                        }
                    )
                    await send(audio_message)

            case ServerEventTypes.RESPONSE_AUDIO_DONE:
                # Relay the audio completion event to the client
                await send(
                    json.dumps(
                        {
                            "type": "response.audio.done",
                            "response_id": parsed_event.response_id,
                        }
                    )
                )
                send_event_log(
                    "Audio Complete", f"Response ID: {parsed_event.response_id}"
                )

            case ServerEventTypes.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
                if isinstance(message, dict):
                    await send_function_call_message(
                        send,
                        parsed_event.item_id,
                        "Assistant",
                        f"{message.get('name')}({parsed_event.arguments})",
                    )
                    send_event_log(
                        "Function Call",
                        f"Calling function: {message.get('name')} with arguments {parsed_event.arguments}",
                    )
                else:
                    await send_function_call_message(
                        send,
                        parsed_event.item_id,
                        "Function",
                        message,
                    )
                    send_event_log("Function", message)

            case ServerEventTypes.ERROR:
                send_event_log("Error", str(parsed_event.error.message))

            case ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DELTA:
                # Check if the message item already exists, if not, create it
                if (
                    not hasattr(self, "assistant_message_created")
                    or not self.assistant_message_created
                ):
                    await send_conversation_message(
                        send, parsed_event.item_id, "Assistant", ""
                    )
                    self.assistant_message_created = True
                await update_conversation_message(
                    send, parsed_event.item_id, parsed_event.delta
                )

            case ServerEventTypes.RESPONSE_CREATED:
                # Relay the response.created event to the client
                await send(
                    json.dumps(
                        {
                            "type": "response.created",
                            "response_id": parsed_event.response.id,
                        }
                    )
                )
                send_event_log(
                    "Response Created", f"Response ID: {parsed_event.response.id}"
                )

    async def on_connect(self, send):
        try:

            async def message_callback(event, message):
                return await self.relay_openai_message(send, event, message)

            self.openai_client = OpenAIRealtimeClient(message_callback=message_callback)
            await self.openai_client.start()

            send_event_log("Connected", "New WebSocket connection established")
        except Exception as e:
            print(f"Error sending initial message: {e}")

    async def on_disconnect(self):
        if self.openai_client:
            await self.openai_client.stop()
            self.openai_client = None

    async def process_user_audio(self, audio_data: str) -> None:
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
        await self.openai_client.send(audio_event.model_dump_json(exclude_none=True))

        # Send commit event
        commit_event = client_events.InputAudioBufferCommit(
            type=ClientEventTypes.INPUT_AUDIO_BUFFER_COMMIT
        )
        await self.openai_client.send(commit_event.model_dump_json(exclude_none=True))

    async def process_user_text(self, send, text_data: str) -> None:
        """Process text input and send it to OpenAI client"""
        # Create content for the conversation item
        content = client_events.MessageContent(  # Changed from ConversationItemContent
            type="input_text", text=text_data
        )

        # Create the conversation item
        conversation_item = client_events.ConversationItem(
            type="message", role="user", content=[content]
        )

        # Create the conversation item create event
        create_event = client_events.ConversationItemCreate(
            type=ClientEventTypes.CONVERSATION_ITEM_CREATE, item=conversation_item
        )

        # Send the event to OpenAI
        await self.openai_client.send(create_event.model_dump_json(exclude_none=True))

        # Create and send response create event to get the assistant's response
        response_event = client_events.ResponseCreate(
            type=ClientEventTypes.RESPONSE_CREATE
        )
        await self.openai_client.send(response_event.model_dump_json(exclude_none=True))

        await send_conversation_message(send, "1", "User", text_data)

    async def process_message(self, data: dict, send) -> None:
        """Process incoming WebSocket messages"""
        msg_type = data.get("type")

        match msg_type:
            case "voice_config":
                if self.openai_client and (voice := data.get("voice")):
                    await self.openai_client.configure_session(voice=voice)
                    send_event_log("Voice Configuration", f"Set voice to: {voice}")

            case "cancel":
                if hasattr(self, "current_task"):
                    self.current_task.cancel()

                # Send a response.cancel event to OpenAI
                cancel_event = client_events.ResponseCancel(
                    type=ClientEventTypes.RESPONSE_CANCEL
                )
                await self.openai_client.send(
                    cancel_event.model_dump_json(exclude_none=True)
                )

                send_event_log("Cancelled", "Audio streaming cancelled")

            case "audio":
                if audio_data := data.get("data"):
                    try:
                        await self.process_user_audio(audio_data)
                        send_event_log(
                            "Audio received", f"Length: {len(audio_data)} bytes"
                        )
                    except Exception as e:
                        send_event_log("Error", f"Failed to process audio: {str(e)}")

            case "text":
                if text_data := data.get("data"):
                    try:
                        await self.process_user_text(send, text_data)
                        send_event_log("Text received", f"{text_data}")
                    except Exception as e:
                        send_event_log("Error", f"Failed to process text: {str(e)}")

            case _:
                send_event_log(
                    "Unknown Event", f"Received unknown message type: {msg_type}"
                )


message_handler = OpenAIMessageHandler()


@app.ws(
    "/wscon", conn=message_handler.on_connect, disconn=message_handler.on_disconnect
)
async def myws(data, send):
    if not data:
        print("Skipping empty message")
        return

    try:
        await message_handler.process_message(data, send)
    except asyncio.CancelledError:
        print("Task was cancelled")
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        import traceback

        traceback.print_exc()


serve()
