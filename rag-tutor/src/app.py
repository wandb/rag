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

tlink = Script(src="https://cdn.tailwindcss.com")
custom_style = StyleX(fname=f"{static_dir}/css/style.css")

dlink = Link(
    rel="stylesheet",
    href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css",
)
leaflet_css = Link(
    rel="stylesheet", href="https://unpkg.com/leaflet@1.6.0/dist/leaflet.css"
)

leaflet_js = Script(src="https://unpkg.com/leaflet@1.6.0/dist/leaflet.js")
htmx_ws = Script(src="https://unpkg.com/htmx-ext-ws@2.0.0/ws.js")
fonts = (
    Link(
        rel="stylesheet",
        href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;600&display=swap",
    ),
)
app, rt = fast_app(
    static_path=os.path.dirname(static_dir),
    pico=True,
    htmx=True,
    ws_hdr=True,
    exts="ws",
    hdrs=(
        tlink,
        dlink,
        leaflet_css,
        custom_style,
        leaflet_js,
        htmx_ws,
        fonts,
        Script(src="https://unpkg.com/audiomotion-analyzer@4.5.0/dist/index.js"),
        Script(src="/static/js/index.js", type="module"),
    ),
)

# Store start time for event timestamps
start_time = None


def create_layout(
    button_text="connect",
    button_props={},
    ws_props={},
    show_initial_messages=True,
    audio_player_disabled=False,
    controls_disabled=True,
    include_audio_stream=False,
):
    """Creates the main layout with configurable options"""

    # Common button properties
    default_button_props = {
        "id": "connect-btn",
        "hx_post": "/connect" if button_text == "connect" else "/disconnect",
        "hx_swap": "outerHTML",
    }
    button_props = {**default_button_props, **button_props}

    # Common container properties
    container_props = {
        "cls": "console-layout",
        "id": "ws-container",
    }
    container_props.update(ws_props)

    return Div(
        # Top Bar
        Div(
            Span("realtime console", style="margin-left:12px"),
            Div(style="flex-grow:1"),
            # Add voice selector dropdown
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
                cls="select select-bordered select-sm w-32 mr-2",
            ),
            Button(button_text, **button_props),
            cls="top-bar",
        ),
        # Main Content
        Div(
            # Events Panel
            Div(
                # Events Section
                Div(
                    H3("events", style="margin:0 0 16px 0"),
                    Div(
                        P("awaiting connection...") if show_initial_messages else None,
                        id="event-log",
                        cls="event-log",
                    ),
                    cls="events-section",
                ),
                # Visualization Panel
                Div(
                    Div(
                        Div(
                            Div(
                                H4(
                                    "User",
                                    style="margin:0 0 8px 0; text-align:center",
                                ),
                                Canvas(id="client-canvas"),
                                cls="visualization-entry client",
                            ),
                            Div(
                                H4(
                                    "Assistant",
                                    style="margin:0 0 8px 0; text-align:center",
                                ),
                                Canvas(id="server-canvas"),
                                cls="visualization-entry server",
                            ),
                            cls="visualization",
                        ),
                        cls="visualization-panel",
                    ),
                    cls="visualization-section",
                ),
                # Conversation Section
                Div(
                    H3("conversation", style="margin:0 0 16px 0"),
                    Div(
                        P("awaiting connection...") if show_initial_messages else None,
                        id="conversation-content",
                        cls="conversation-content",
                    ),
                    Div(
                        Audio(
                            id="audio-player",
                            controls=True,
                            preload="auto",
                            disabled=audio_player_disabled,
                        ),
                        cls="audio-player-container",
                    ),
                    cls="conversation",
                ),
                cls="events-panel",
            ),
            # Controls moved here, outside of events-panel
            Div(
                Div(
                    Input(
                        type="text",
                        id="text-input",
                        placeholder="Type your message...",
                        disabled="disabled" if controls_disabled else None,
                    ),
                    Button(
                        "Send",
                        id="send-btn",
                        disabled="disabled" if controls_disabled else None,
                    ),
                    Button(
                        "Push to Talk",
                        id="ptt-btn",
                        disabled="disabled" if controls_disabled else None,
                    ),
                    cls="controls-inner",
                ),
                cls="controls",
            ),
            cls="main-content",
        ),
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
    return Container(create_layout())


@rt("/connect")
def post():
    global start_time
    start_time = datetime.now()

    return create_layout(
        button_text="disconnect",
        show_initial_messages=False,
        controls_disabled=False,
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
                Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
                Span(speaker, cls="event-type"),
                Div(
                    message, cls=f"event-data{' error' if is_error else ''}", id=item_id
                ),
                cls="event-item",
            ),
            cls="conversation-content",
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
