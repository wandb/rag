import asyncio
import json
from datetime import datetime

from fasthtml.common import *
from pydub import AudioSegment

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
        Style,
    ),
)


# Store start time for event timestamps
start_time = None


def create_top_bar(item_id, button_text="connect"):
    return Div(
        H2(
            "RagTutor Console",
            cls="mx-4 h-1/2 text-md font-sans font-semibold text-[#FFFFFF]",
        ),
        Div(cls="grow"),
        Select(
            Option("alloy", value="alloy", selected=True),
            Option("ash", value="ash"),
            Option("ballad", value="ballad"),
            Option("coral", value="coral"),
            Option("echo", value="echo"),
            Option("sage", value="sage"),
            Option("shimmer", value="shimmer"),
            Option("verse", value="verse"),
            id="voice-selector",
            cls="w-32 h-1/2 mx-2 px-4 select bg-[#ffcc33] border text-[#1A1C1F] "
            "focus:outline-none focus:bg-white focus:border-gray-500 leading-tight text-md rounded-lg "
            "font-sans disabled:opacity-75 disabled:bg-[#EE4B2B] disabled:cursor-not-allowed",
        ),
        Button(
            button_text,
            id="connect-btn",
            hx_post=("/connect" if button_text == "connect" else "/disconnect"),
            hx_swap="outerHTML",
            cls="h-1/2 mx-4 p-1 rounded-lg text-md border text-[#1A1C1F] whitespace-nowrap "
            "font-sans bg-[#ffcc33] disabled:opacity-75 disabled:bg-[#EE4B2B] "
            "disabled:cursor-not-allowed",
        ),
        cls="h-16 flex flex-row bg-[#242629] border shadow-xl rounded-lg items-center justify-between overflow-hidden",
        id=item_id,
    )


def create_visualization_canvas(name, item_id):
    return Div(
        H4(
            name,
            cls="mx-4 text-md font-sans font-semibold text-[#FFFFFF]",
        ),
        Canvas(
            id=item_id,
            cls="h-24 m-4 p-4 w-9/12 border rounded-lg",
        ),
        cls="flex flex-1 flex-col bg-[#242629] shadow-xl rounded-lg items-center justify-between"
        "overflow-hidden",
    )


def create_visualization_panel(item_id):
    return (
        Div(
            create_visualization_canvas(name="User", item_id="client-canvas"),
            create_visualization_canvas(name="Assistant", item_id="server-canvas"),
            cls="flex flex-row gap-4 w-full bg-[#242629] border shadow-xl rounded-lg rounded-lg items-center "
            "justify-between",
            id=item_id,
        ),
    )


def create_conversation_panel(item_id, disabled=True):
    return Div(
        H3(
            "Conversation",
            cls="m-4 text-md font-sans font-semibold text-[#FFFFFF]",
        ),
        Div(
            (
                P(
                    "awaiting connection...",
                    cls="mx-4 text-xs font-sans font-light text-[#FFFFFF]",
                )
                if disabled
                else None
            ),
            id="conversation-content",
            cls="flex flex-col h-full flex-initial grow-0 overscroll-auto max-h-max gap-2 bg-[#242629] "
            "overflow-y-auto",
        ),
        cls="flex flex-col flex-initial grow-0 overscroll-auto h-3/4 max-h-full gap-2 bg-[#242629] border shadow-xl "
        "rounded-lg overflow-hidden overflow-y-auto",
        id=item_id,
    )


def create_audio_panel(item_id, disabled=False):
    return (
        Div(
            Audio(
                id="audio-player",
                controls=True,
                preload="auto",
                disabled=disabled,
                cls="w-1/2 h-full rounded-lg disabled:opacity-75 disabled:cursor-not-allowed",
            ),
            cls="flex h-12 p-2 bg-[#242629] border shadow-xl rounded-lg overflow-hidden items-center justify-center",
            id=item_id,
        ),
    )


def create_inputs_panel(item_id, disabled=False):
    return Div(
        Input(
            # type="text",
            id="text-input",
            placeholder="Type your message...",
            disabled="disabled" if disabled else None,
            cls="h-1/2 w-1/2 p-1 rounded-lg text-md border text-[#1A1C1F] whitespace-nowrap font-['Source Sans 3, "
            "Sans-serif'] bg-white disabled:opacity-75 disabled:opacity-75 disabled:bg-gray-200 "
            "disabled:cursor-not-allowed",
        ),
        Button(
            "Send",
            id="send-btn",
            disabled="disabled" if disabled else None,
            cls="m-1 h-1/2 p-1 rounded-lg text-md border text-[#1A1C1F] whitespace-nowrap font-['Source Sans 3, "
            "Sans-serif'] bg-[#ffcc33] disabled:opacity-75 disabled:opacity-75 disabled:bg-[#EE4B2B] "
            "disabled:cursor-not-allowed",
        ),
        Button(
            "Push to Talk",
            id="ptt-btn",
            disabled="disabled" if disabled else None,
            cls="m-1 h-1/2 p-1 rounded-lg text-md border text-[#1A1C1F] whitespace-nowrap font-['Source Sans 3, "
            "Sans-serif'] bg-[#ffcc33] disabled:opacity-75 disabled:bg-[#EE4B2B] disabled:cursor-not-allowed",
        ),
        cls="flex flex-row h-16 gap-1 bg-[#242629] border shadow-xl rounded-lg overflow-hidden items-center "
        "justify-center",
        id=item_id,
    )


def create_layout(
    button_text="connect",
    ws_props={},
    disabled=True,
    audio_player_disabled=False,
    include_audio_stream=False,
):
    """Creates the main layout with configurable options"""

    # Common container properties
    container_props = {
        "cls": "h-screen w-screen p-4 bg-[#333] flex flex-col gap-4",
        "id": "ws-container",
    }
    container_props.update(ws_props)

    return Main(
        # Main Content
        # Top Bar Panel
        create_top_bar(item_id="top-bar", button_text=button_text),
        # Visualization Panel
        Div(
            Div(
                create_visualization_panel(item_id="visualization-panel"),
                # Conversation Panel
                create_conversation_panel(
                    item_id="conversation-panel", disabled=disabled
                ),
                # Audio Player
                create_audio_panel(
                    item_id="audio-panel", disabled=audio_player_disabled
                ),
                cls="flex flex-col flex-1 gap-4",
            ),
            Div(
                cls="flex flex-col flex-1 gap-4",
            ),
            cls="flex flex-row flex-initial h-5/6 grow-0 gap-4",
        ),
        # Inputs Panel
        create_inputs_panel(item_id="inputs-panel", disabled=disabled),
        # Optional audio stream container
        (
            Div(id="audio-stream-container", style="display:none;")
            if include_audio_stream
            else None
        ),
        **container_props,
    )


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


async def send_event_log(send, event_type: str, event_data: str):
    """Utility function to send event log updates via HTMX"""
    await send(
        Div(
            Div(
                Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
                Span(event_type, cls="event-type"),
                Span(event_data, cls="event-data"),
                cls="event-item",
            ),
            cls="event-log",
            id="event-log",
            hx_swap_oob="beforeend",
        )
    )


async def send_conversation_message(
    send, item_id: str, speaker: str, message: str = "", is_error: bool = False
):
    """Utility function to send conversation updates via HTMX"""
    await send(
        Div(
            Div(
                Div(
                    speaker,
                    cls="mx-4 w-1/12 text-xs font-sans font-light text-[#ffcc33] text-left",
                ),
                Div(
                    message,
                    cls=f"mx-1 flex-1 p-1 text-xs font-sans font-light text-[#FFFFFF] "
                    f"text-wrap text-left whitespace-pre",
                    id=item_id,
                ),
                cls="flex flex-row content-start",
            ),
            cls="flex flex-col gap-1",
            id="conversation-content",
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
            cls=f"event-data{' error' if is_error else ''}",
            id=item_id,
            hx_swap_oob="beforeend",
        )
    )


class OpenAIMessageHandler:
    def __init__(self):
        self.openai_client = None

    async def relay_openai_message(self, send, parsed_event: ServerEvent):
        """
        Relay OpenAI messages to the client with appropriate UI updates.

        Args:
            send: Websocket send function
            parsed_event: Parsed server event from OpenAI
        """
        match parsed_event.type:
            case ServerEventTypes.SESSION_CREATED:
                await send_event_log(
                    send, "Session Created", f"Session ID: {parsed_event.event_id}"
                )

            case ServerEventTypes.SESSION_UPDATED:
                await send_event_log(
                    send, "Session Updated", f"Session ID: {parsed_event.event_id}"
                )

            case ServerEventTypes.CONVERSATION_CREATED:
                await send_event_log(
                    send,
                    "Conversation Started",
                    f"Conversation ID: {parsed_event.conversation.id}",
                )

            case ServerEventTypes.CONVERSATION_ITEM_CREATED:
                await send_event_log(
                    send,
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
                await send_event_log(
                    send, "Audio Complete", f"Response ID: {parsed_event.response_id}"
                )

            case ServerEventTypes.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
                await send_event_log(
                    send,
                    "Function Call",
                    f"Function call completed: {parsed_event.arguments}",
                )

            case ServerEventTypes.ERROR:
                await send_event_log(send, "Error", str(parsed_event.error.message))

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
                await send_event_log(
                    send, "Response Created", f"Response ID: {parsed_event.response.id}"
                )

    async def on_connect(self, send):
        try:
            message_handler = lambda event: self.relay_openai_message(send, event)
            self.openai_client = OpenAIRealtimeClient(message_callback=message_handler)
            await self.openai_client.start()

            await send_event_log(
                send, "Connected", "New WebSocket connection established"
            )
        except Exception as e:
            print(f"Error sending initial message: {e}")

    async def on_disconnect(self):
        if self.openai_client:
            await self.openai_client.stop()
            self.openai_client = None

    async def process_user_audio(self, audio_data: str, metadata: dict) -> None:
        """
        Process audio data and send it to OpenAI client

        Args:
            audio_data: Base64 encoded WAV data
            metadata: Audio metadata containing sampleRate and duration
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
                    await send_event_log(
                        send, "Voice Configuration", f"Set voice to: {voice}"
                    )

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

                await send_event_log(send, "Cancelled", "Audio streaming cancelled")

            case "audio":
                if audio_data := data.get("data"):
                    try:
                        await self.process_user_audio(
                            audio_data, data.get("metadata", {})
                        )
                        await send_event_log(
                            send, "Audio received", f"Length: {len(audio_data)} bytes"
                        )
                    except Exception as e:
                        await send_event_log(
                            send, "Error", f"Failed to process audio: {str(e)}"
                        )

            case "text":
                if text_data := data.get("data"):
                    try:
                        await self.process_user_text(send, text_data)
                        await send_event_log(send, "Text received", f"{text_data}")
                    except Exception as e:
                        await send_event_log(
                            send, "Error", f"Failed to process text: {str(e)}"
                        )

            case _:
                await send_event_log(
                    send, "Unknown Event", f"Received unknown message type: {msg_type}"
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
