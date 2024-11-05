import asyncio
import json
from datetime import datetime

from fasthtml.common import *
from pydub import AudioSegment

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

# Store conversation state
conversation_items = []


def get_audio_chunks(file_path, chunk_duration_ms=10000):
    """
    Converts MP3 to WAV format and splits it into 10-second chunks
    Returns: List of base64-encoded PCM16 chunks
    """
    # Load and convert MP3 to WAV format
    audio = AudioSegment.from_mp3(file_path)

    # Convert to mono and set sample rate to 44100Hz
    audio = audio.set_channels(1).set_frame_rate(44100)

    # Split into chunks
    chunks = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i : i + chunk_duration_ms]
        # Convert to raw PCM16 data
        buffer = io.BytesIO()
        chunk.export(buffer, format="s16le")
        # Convert to base64
        base64_data = base64.b64encode(buffer.getvalue()).decode()
        chunks.append(base64_data)

    return chunks


@rt("/")
def get():
    return Container(
        Div(
            # Top Bar
            Div(
                Span("realtime console", style="margin-left:12px"),
                Div(style="flex-grow:1"),  # Spacer
                Button(
                    "connect", id="connect-btn", hx_post="/connect", hx_swap="outerHTML"
                ),
                cls="top-bar",
            ),
            # Main Content
            Div(
                # Left Panel (Events & Conversation)
                Div(
                    # Events Panel
                    Div(
                        # Events Section
                        Div(
                            H3("events", style="margin:0 0 16px 0"),
                            Div(
                                P("awaiting connection..."),
                                id="event-log",
                                cls="event-log",
                            ),
                            cls="events-section",
                        ),
                        # Conversation Section
                        Div(
                            H3("conversation", style="margin:0 0 16px 0"),
                            Div(
                                Audio(
                                    id="audio-player",
                                    controls=True,
                                    disabled="disabled",
                                ),
                                P("awaiting connection..."),
                                id="conversation-content",
                            ),
                            cls="conversation",
                        ),
                        # Controls
                        Div(
                            Button(
                                "Push to Talk",
                                id="ptt-btn",
                                disabled="disabled",
                            ),
                            cls="controls",
                        ),
                        cls="events-panel",
                    ),
                    cls="main-content",
                ),
                cls="console-layout",
                id="ws-container",  # Keep only the ID
            ),
        )
    )


@rt("/connect")
def post():
    global start_time
    start_time = datetime.now()

    return (
        Button(
            "disconnect", id="connect-btn", hx_post="/disconnect", hx_swap="outerHTML"
        ),
        Div(
            Div(
                Div(
                    # Events Panel
                    Div(
                        H3("events", style="margin:0 0 16px 0"),
                        Div(
                            id="event-log",
                            cls="event-log",
                        ),
                        cls="events-section",
                    ),
                    # Conversation Section
                    Div(
                        H3("conversation", style="margin:0 0 16px 0"),
                        Div(
                            Audio(
                                id="audio-player",
                                controls=True,
                                preload="auto",
                                disabled=False,
                            ),
                            id="conversation-content",
                        ),
                        cls="conversation",
                    ),
                    # Controls Section
                    Div(
                        Button(
                            "Push to Talk",
                            id="ptt-btn",
                            disabled=None,
                        ),
                        cls="controls",
                    ),
                    cls="events-panel",
                ),
                cls="main-content",
            ),
            # Add a hidden div to hold the audio stream
            Div(id="audio-stream-container", style="display:none;"),
            cls="console-layout",
            id="ws-container",
            hx_ext="ws",
            ws_connect="/wscon",
            ws_send=True,
            hx_trigger="audioMessage",
            _="on htmx:wsAfterMessage if event.detail.message.type === 'audio' call processAudioChunk(event.detail.message.data)",
            hx_swap_oob="true",
        ),
        # Update the event log
        Div(
            Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
            Span("Connected", cls="event-type"),
            Span("WebSocket connection established", cls="event-data"),
            cls="event-item",
            id="event-log",
            hx_swap_oob="true",
        ),
    )


@rt("/disconnect")
def post():
    global start_time
    start_time = None

    return (
        Button("connect", id="connect-btn", hx_post="/connect", hx_swap="outerHTML"),
        Button(
            "Push to Talk",
            id="ptt-btn",
            disabled="disabled",
            hx_swap_oob="true",
        ),
        Div(
            Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
            Span("Disconnected", cls="event-type"),
            Span("WebSocket connection closed", cls="event-data"),
            cls="event-item",
            id="event-log",
            hx_swap_oob="true",
        ),
    )


async def on_connect(send):
    try:
        print("New WebSocket connection established")
        message = Div(
            Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
            Span("Connected", cls="event-type"),
            Span("New WebSocket connection established", cls="event-data"),
            cls="event-item",
            id="event-log",
            hx_swap_oob="beforeend",
        )
        await send(message)
    except Exception as e:
        print(f"Error sending initial message: {e}")


async def on_disconnect():
    print("WebSocket disconnected")


@app.ws("/wscon", conn=on_connect, disconn=on_disconnect)
async def myws(data, send):
    if not data:
        print("Skipping empty message")
        return

    try:
        msg_type = data.get("type")
        if msg_type == "audio":
            audio_data = data.get("data")
            if audio_data:
                print("Received audio data")
                await send(
                    Div(
                        Span(
                            datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"
                        ),
                        Span("Audio received", cls="event-type"),
                        Span(f"Length: {len(audio_data)} bytes", cls="event-data"),
                        cls="event-item",
                        id="event-log",
                        hx_swap_oob="beforeend",
                    )
                )
                # For testing: Instead of using received audio, load speech.mp3
                chunks = get_audio_chunks("speech.mp3")

                print(f"Sending {len(chunks)} audio chunks")
                for i, chunk in enumerate(chunks):
                    # Send event log update
                    await send(
                        Div(
                            Span(
                                datetime.now().strftime("%H:%M:%S"),
                                cls="event-timestamp",
                            ),
                            Span("Audio chunk sent", cls="event-type"),
                            Span(f"Chunk {i+1}/{len(chunks)}", cls="event-data"),
                            cls="event-item",
                            id="event-log",
                            hx_swap_oob="beforeend",
                        )
                    )

                    # Send the actual audio data with type information
                    await send(
                        json.dumps(
                            {
                                "type": "audio",
                                "data": chunk,
                                "chunk": i + 1,
                                "total": len(chunks),
                            }
                        )
                    )

                    # Add a small delay between chunks to simulate streaming
                    await asyncio.sleep(1)
        else:
            # Handle other event types
            await send(
                Div(
                    Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
                    Span("Event received", cls="event-type"),
                    Span(
                        str(data.get("data", "")),
                        cls="event-item",
                        id="event-log",
                        hx_swap_oob="beforeend",
                    ),
                )
            )

    except Exception as e:
        print(f"Error processing message: {str(e)}")
        import traceback

        traceback.print_exc()


serve()
