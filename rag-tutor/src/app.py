import json
from datetime import datetime

from fasthtml.common import *

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
                # Add a hidden div to hold the audio stream
                Div(id="audio-stream-container", style="display:none;"),
                cls="console-layout",
                id="ws-container",  # Keep the ID but remove WebSocket attributes
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
                            id="conversation-content",
                            cls="conversation",
                        ),
                        cls="conversation",
                    ),
                    # Controls Section
                    Div(
                        Button(
                            "Push to Talk",
                            id="ptt-btn",
                            disabled=None,
                            # ws_send="audioMessage",
                            # hx_trigger="audioMessage",
                        ),
                        cls="controls",
                    ),
                    cls="events-panel",
                ),
                cls="main-content",
            ),
            cls="console-layout",
            id="ws-container",
            hx_ext="ws",
            ws_connect="/wscon",
            ws_send=True,
            hx_trigger="audioMessage",
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

    # Skip empty messages (like initial connection)
    if not data:
        print("Skipping empty message")
        return

    try:
        try:
            # Handle different message types
            msg_type = data.get("type")
            if msg_type == "event":
                await send(
                    Div(
                        Span(
                            datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"
                        ),
                        Span("Event received", cls="event-type"),
                        Span(data.get("data", ""), cls="event-data"),
                        cls="event-item",
                        id="event-log",
                        hx_swap_oob="beforeend",
                    )
                )
            elif msg_type == "audio":
                audio_data = data.get("data")
                if audio_data:
                    await send(
                        Div(
                            # Audio player container with visualization
                            Div(
                                # Audio controls
                                Button(
                                    "▶️ Play",
                                    cls="play-btn",
                                    onclick="playAudioChunk(this.parentElement)",
                                ),
                                # Visualization canvas
                                Canvas(
                                    cls="audio-visualizer",
                                    style="width:100%; height:50px; background:#f0f0f0; margin:4px 0;",
                                ),
                                # Hidden audio data
                                data_audio=audio_data,
                                cls="audio-player-container",
                            ),
                            # Keep the timestamp
                            P(
                                datetime.now().strftime("%H:%M:%S"),
                                style="margin:4px 0; color:#666;",
                            ),
                            id="conversation-content",
                            hx_swap_oob="beforeend",
                        )
                    )

                    # Also log the event
                    await send(
                        Div(
                            Span(
                                datetime.now().strftime("%H:%M:%S"),
                                cls="event-timestamp",
                            ),
                            Span("Audio received", cls="event-type"),
                            Span(f"Length: {len(audio_data)} bytes", cls="event-data"),
                            cls="event-item",
                            id="event-log",
                            hx_swap_oob="beforeend",
                        )
                    )
            else:
                # Handle raw text as a generic message
                await send(
                    Div(
                        Span(
                            datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"
                        ),
                        Span("Message received", cls="event-type"),
                        Span(str(data), cls="event-data"),
                        cls="event-item",
                        id="event-log",
                        hx_swap_oob="beforeend",
                    )
                )

        except json.JSONDecodeError:
            # Handle raw text messages
            print("Received raw text message")
            await send(
                Div(
                    Span(datetime.now().strftime("%H:%M:%S"), cls="event-timestamp"),
                    Span("Text received", cls="event-type"),
                    Span(msg, cls="event-data"),
                    cls="event-item",
                    id="event-log",
                    hx_swap_oob="beforeend",
                )
            )

    except Exception as e:
        print(f"Error processing message: {str(e)}")
        import traceback

        traceback.print_exc()


serve()
