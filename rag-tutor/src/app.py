import asyncio
import json
from datetime import datetime

from fasthtml.common import *
from pydub import AudioSegment

from src.components.models import (
    ClientEventTypes,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    ServerEvent,
    ServerEventTypes,
)
from src.components.oai_relay import OpenAIRealtimeClient

static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))

tlink = Script(src="https://cdn.tailwindcss.com")
custom_style = StyleX(fname=f"{static_dir}/style.css")

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
        href="https://fonts.googleapis.com/css2?family=Roboto+Mono:ital,wght@0,100..700;1,100..700&display=swap",
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
        Script(src="/static/wavtools/index.js", type="module"),
        Script(src="/static/events.js", type="text/javascript"),
    ),
)

# Store start time for event timestamps
start_time = None


openai_client = None


def create_layout(
    button_text="connect",
    button_props={},
    ws_props={},
    show_initial_messages=True,
    audio_player_disabled=False,
    ptt_disabled=True,
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
                # Controls
                Div(
                    Button(
                        "Push to Talk",
                        id="ptt-btn",
                        disabled="disabled" if ptt_disabled else None,
                    ),
                    cls="controls",
                ),
                cls="events-panel",
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
        ptt_disabled=False,
        include_audio_stream=True,
        ws_props={
            "hx_ext": "ws",
            "ws_connect": "/wscon",
            "ws_send": True,
            "hx_trigger": "audioMessage",
            "_": "on htmx:wsAfterMessage if event.detail.message.type === 'audio' call processAudioChunk(event.detail.message.data)",
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
        ws_props={
            "hx_ext": "ws",
            "hx_swap_oob": "true",
        },
    )


async def relay_openai_message(send, parsed_event: ServerEvent):
    """
    Relay OpenAI messages to the client with appropriate UI updates.

    Args:
        send: Websocket send function
        parsed_event: Parsed server event from OpenAI
    """
    event_type = parsed_event.type

    if event_type == ServerEventTypes.SESSION_CREATED:
        await send(
            Div(
                Div(
                    Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
                    Span("Session Created", cls="event-type"),
                    Span(f"Session ID: {parsed_event.event_id}", cls="event-data"),
                    cls="event-item",
                ),
                cls="event-log",
                id="event-log",
                hx_swap_oob="beforeend",
            )
        )
    elif event_type == ServerEventTypes.SESSION_UPDATED:
        await send(
            Div(
                Div(
                    Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
                    Span("Session Updated", cls="event-type"),
                    Span(f"Session ID: {parsed_event.event_id}", cls="event-data"),
                    cls="event-item",
                ),
                cls="event-log",
                id="event-log",
                hx_swap_oob="beforeend",
            )
        )
    elif event_type == ServerEventTypes.CONVERSATION_CREATED:
        await send(
            Div(
                Div(
                    Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
                    Span("Conversation Started", cls="event-type"),
                    Span(
                        f"Conversation ID: {parsed_event.conversation.id}",
                        cls="event-data",
                    ),
                    cls="event-item",
                ),
                cls="event-log",
                id="event-log",
                hx_swap_oob="beforeend",
            )
        )
    elif event_type == ServerEventTypes.CONVERSATION_ITEM_CREATED:
        await send(
            Div(
                Div(
                    Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
                    Span("Conversation Item Created", cls="event-type"),
                    Span(
                        f"Conversation Item ID: {parsed_event.event_id}",
                        cls="event-data",
                    ),
                    cls="event-item",
                ),
                cls="event-log",
                id="event-log",
                hx_swap_oob="beforeend",
            )
        )
    elif (
        event_type
        == ServerEventTypes.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED
    ):
        await send(
            Div(
                Div(
                    Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
                    Span("User", cls="event-type"),
                    Span(parsed_event.transcript, cls="event-data"),
                    cls="event-item",
                ),
                cls="conversation-content",
                id="conversation-content",
                hx_swap_oob="beforeend",
            )
        )
    elif event_type == ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DONE:
        await send(
            Div(
                Div(
                    Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
                    Span("Assistant", cls="event-type"),
                    Span(parsed_event.transcript, cls="event-data"),
                    cls="event-item",
                ),
                cls="conversation-content",
                id="conversation-content",
                hx_swap_oob="beforeend",
            )
        )
    elif event_type == ServerEventTypes.RESPONSE_AUDIO_DELTA:
        if parsed_event.delta is not None:
            audio_message = json.dumps(
                {
                    "type": "audio",
                    "data": parsed_event.delta,  # OpenAI already provides base64-encoded PCM16
                }
            )
            await send(audio_message)
    elif event_type == ServerEventTypes.RESPONSE_DONE:
        pass
    elif event_type == ServerEventTypes.ERROR:
        await send(
            Div(
                Div(
                    Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
                    Span("Error", cls="event-type"),
                    Span(str(parsed_event.error), cls="event-data error"),
                    cls="event-item",
                ),
                cls="conversation-content",
                id="conversation-content",
                hx_swap_oob="beforeend",
            )
        )


async def on_connect(send):
    try:
        print("New WebSocket connection established")
        global openai_client

        # Create partial function with send already bound
        message_handler = lambda event: relay_openai_message(send, event)

        openai_client = OpenAIRealtimeClient(message_callback=message_handler)
        await openai_client.start()

        message = Div(
            Div(
                Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
                Span("Connected", cls="event-type"),
                Span("New WebSocket connection established", cls="event-data"),
                cls="event-item",
            ),
            cls="event-log",
            id="event-log",
            hx_swap_oob="beforeend",
        )
        await send(message)
    except Exception as e:
        print(f"Error sending initial message: {e}")


async def on_disconnect():
    global openai_client
    if openai_client:
        await openai_client.stop()
        openai_client = None
    print("WebSocket disconnected")


async def process_user_audio(audio_data: str, metadata: dict) -> None:
    """
    Process audio data and send it to OpenAI client

    Args:
        audio_data: Base64 encoded WAV data
        metadata: Audio metadata containing sampleRate and duration
    """
    global openai_client
    print("Received audio data")

    # Decode base64 WAV data
    wav_data = base64.b64decode(audio_data)

    # Load into pydub and convert to mono
    audio = AudioSegment.from_wav(io.BytesIO(wav_data))
    # Resample to 24kHz mono pcm16
    pcm_audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2).raw_data
    # Encode to base64 string
    pcm_base64 = base64.b64encode(pcm_audio).decode()

    # Create the audio buffer append event
    audio_event = InputAudioBufferAppend(
        type=ClientEventTypes.INPUT_AUDIO_BUFFER_APPEND, audio=pcm_base64
    )

    # Send audio buffer
    await openai_client.send(audio_event.model_dump_json(exclude_none=True))

    # Send commit event
    commit_event = InputAudioBufferCommit(
        type=ClientEventTypes.INPUT_AUDIO_BUFFER_COMMIT
    )
    await openai_client.send(commit_event.model_dump_json(exclude_none=True))

    print(
        f"Sent audio buffer: {metadata['sampleRate']}Hz, mono channel, {metadata['duration']}s"
    )


@app.ws("/wscon", conn=on_connect, disconn=on_disconnect)
async def myws(data, send):
    if not data:
        print("Skipping empty message")
        return

    try:
        msg_type = data.get("type")

        # Handle cancel event
        if msg_type == "cancel":
            print("Audio streaming cancelled")
            # Clear any pending tasks
            if hasattr(myws, "current_task"):
                myws.current_task.cancel()
            await send(
                Div(
                    Div(
                        Span(
                            datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"
                        ),
                        Span("Cancelled", cls="event-type"),
                        Span("Audio streaming cancelled", cls="event-data"),
                        cls="event-item",
                    ),
                    cls="event-log",
                    id="event-log",
                    hx_swap_oob="beforeend",
                )
            )
            return

        if msg_type == "audio":
            audio_data = data.get("data")
            metadata = data.get("metadata", {})
            if audio_data:
                await process_user_audio(audio_data, metadata)

                # Send UI update
                await send(
                    Div(
                        Div(
                            Span(
                                datetime.now().strftime("%H:%M:%S"),
                                cls="event-timestamp",
                            ),
                            Span("Audio received", cls="event-type"),
                            Span(f"Length: {len(audio_data)} bytes", cls="event-data"),
                            cls="event-item",
                        ),
                        cls="event-log",
                        id="event-log",
                        hx_swap_oob="beforeend",
                    )
                )

        else:
            # Handle other event types
            await send(
                Div(
                    Div(
                        Span(
                            datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"
                        ),
                        Span("Event received", cls="event-type"),
                        Span(str(data.get("data", "")), cls="event-data"),
                        cls="event-item",
                    ),
                    cls="event-log",
                    id="event-log",
                    hx_swap_oob="beforeend",
                )
            )

    except asyncio.CancelledError:
        print("Task was cancelled")
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        import traceback

        traceback.print_exc()


serve()
