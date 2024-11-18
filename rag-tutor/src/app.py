import asyncio
from datetime import datetime

from fasthtml.common import *
from loguru import logger

from src.components import create_layout
from src.relay_service.oai_relay import OpenAIRealtimeRelay

static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
start_time = None
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
    send,
    item_id: str,
    delta: str,
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


client_handlers = {
    "send_conversation_message": send_conversation_message,
    "send_function_call_message": send_function_call_message,
    "update_conversation_message": update_conversation_message,
}
message_handler = OpenAIRealtimeRelay(**client_handlers)


@app.ws(
    "/wscon", conn=message_handler.on_connect, disconn=message_handler.on_disconnect
)
async def myws(data):
    if not data:
        return

    try:
        await message_handler.process_message(data)
    except asyncio.CancelledError:
        logger.error("Task was cancelled")
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")


serve()
