from fasthtml.components import (
    Div,
    H2,
    Select,
    Option,
    Button,
    H4,
    Canvas,
    H3,
    P,
    Audio,
    Input,
    Main,
)


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
            cls="mx-4 mt-4 text-md font-sans font-semibold text-[#FFFFFF]",
        ),
        Canvas(
            id=item_id,
            cls="h-24 m-4 p-4 w-9/12 border rounded-lg",
        ),
        cls="flex flex-1 flex-col bg-[#242629] shadow-xl rounded-lg items-center justify-items-center"
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
                cls="w-3/4 h-full rounded-lg disabled:opacity-75 disabled:cursor-not-allowed",
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
            cls="h-1/2 w-1/2 p-1 rounded-lg text-md border text-[#FFFFFF] whitespace-nowrap font-sans bg-[#45494f] "
            "disabled:opacity-75 disabled:opacity-75 disabled:bg-gray-200 disabled:cursor-not-allowed",
        ),
        Button(
            "Send",
            id="send-btn",
            disabled="disabled" if disabled else None,
            cls="m-1 h-1/2 p-1 rounded-lg text-md border text-[#1A1C1F] whitespace-nowrap font-sans bg-[#ffcc33] "
            "disabled:opacity-75 disabled:opacity-75 disabled:bg-[#EE4B2B] disabled:cursor-not-allowed",
        ),
        Button(
            "Push to Talk",
            id="ptt-btn",
            disabled="disabled" if disabled else None,
            cls="m-1 h-1/2 p-1 rounded-lg text-md border text-[#1A1C1F] whitespace-nowrap font-sans bg-[#ffcc33] "
            "disabled:opacity-75 disabled:bg-[#EE4B2B] disabled:cursor-not-allowed",
        ),
        cls="flex flex-row h-16 gap-1 bg-[#242629] border shadow-xl rounded-lg overflow-hidden items-center "
        "justify-center",
        id=item_id,
    )


def create_function_calls_panel(item_id, disabled=True):
    return Div(
        H3(
            "Function Calls",
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
            id="function-call-content",
            cls="flex flex-col flex-initial h-full w-full grow-0 overscroll-auto max-h-max gap-2 bg-[#242629] "
            "overflow-y-auto overflow-x-auto",
        ),
        cls="flex flex-col flex-1 h-full w-full overscroll-auto gap-2 bg-[#242629] border shadow-xl "
        "rounded-lg overflow-hidden overflow-y-auto overflow-x-auto overflow-hidden",
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
                create_function_calls_panel(
                    item_id="function-calls-panel", disabled=disabled
                ),
                cls="flex flex-col flex-1 gap-4 overflow-hidden",
            ),
            cls="flex flex-row flex-initial h-5/6 grow-0 gap-4 overflow-hidden",
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
