import json

from fasthtml.common import *
from starlette.middleware.cors import CORSMiddleware

from pipeline.generation import call_model

# Update script/link definitions near the top of the file
tlink = Script(
    src="https://cdn.tailwindcss.com",
    referrerpolicy="no-referrer",
)
dlink = Link(
    rel="stylesheet",
    href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css",
    crossorigin="anonymous",
)

app = FastHTML(hdrs=(tlink, dlink), exts="ws")
rt = app.route

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Chat message component with unique ID for content and audio
def ChatMessage(msg_idx, content="", is_user=False, **kwargs):
    bubble_class = "chat-bubble-primary" if not is_user else "chat-bubble-secondary"
    chat_class = "chat-end" if not is_user else "chat-start"
    header_text = "RagTutor" if not is_user else "You"

    return Div(
        Div(header_text, cls="chat-header"),
        Div(
            content,
            id=f"chat-content-{msg_idx}",
            cls=f"chat-bubble {bubble_class}",
        ),
        id=f"chat-message-{msg_idx}",
        cls=f"chat {chat_class}",
        **kwargs,
    )


# The input field component
def ChatInput():
    return Input(
        type="text",
        name="msg",
        id="msg-input",
        placeholder="Type your message...",
        cls="input input-bordered w-full",
        hx_swap_oob="true",
    )


@rt("/")
def get():
    return Titled(
        "RagTutor",
        Body(
            Div(
                H1("RagTutor Chat"),
                Div(id="chatlist", cls="chat-box h-[73vh] overflow-y-auto"),
                Form(
                    Group(ChatInput(), Button("Send", cls="btn btn-primary")),
                    ws_send=True,
                    hx_ext="ws",
                    ws_connect="/wscon",
                    cls="flex space-x-2 mt-2",
                ),
                # Make the Audio component visible by default and add some styling
                Audio(
                    id="chat-audio",
                    controls=True,
                    style="display: block; margin: 1rem 0;",  # Changed from display: none to block
                    cls="w-full",  # Added width class
                ),
                cls="p-4 max-w-lg mx-auto",
            ),
            Script(
                """
                // Create an audio context and buffer for continuous playback
                let audioContext;
                let audioBuffers = [];
                let isPlaying = false;

                async function initAudioContext() {
                    if (!audioContext) {
                        audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    }
                }

                async function playNextBuffer() {
                    if (audioBuffers.length > 0 && !isPlaying) {
                        isPlaying = true;
                        const audioBuffer = audioBuffers.shift();
                        const source = audioContext.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(audioContext.destination);
                        
                        source.onended = () => {
                            isPlaying = false;
                            playNextBuffer(); // Play next buffer when current one ends
                        };
                        
                        source.start(0);
                    }
                }

                // Handle incoming WebSocket messages
                document.body.addEventListener('htmx:wsAfterMessage', async function(evt) {
                    const message = evt.detail.message;
                    
                    // Only try to parse as JSON if it starts with '{'
                    if (typeof message === 'string' && message.trim().startsWith('{')) {
                        try {
                            let data = JSON.parse(message);
                            
                            if (data.type === "audio" && data.data) {
                                await initAudioContext();
                                
                                const audioData = atob(data.data);
                                const arrayBuffer = new ArrayBuffer(audioData.length);
                                const view = new Uint8Array(arrayBuffer);
                                for (let i = 0; i < audioData.length; i++) {
                                    view[i] = audioData.charCodeAt(i);
                                }
                                
                                // Decode the audio data
                                try {
                                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                                    audioBuffers.push(audioBuffer);
                                    playNextBuffer();
                                } catch (error) {
                                    console.error('[Client] Error decoding audio:', error);
                                }
                            }
                        } catch (e) {
                            console.error('[Client] Error processing audio message:', e);
                        }
                    }
                    // Ignore non-JSON messages (HTML content)
                });
                """,
                type="module",
            ),
        ),
    )


@app.ws("/wscon")
async def ws(msg: str, send):
    if not msg:
        return

    try:
        msg_count = 0
        accumulated_transcript = ""

        # Send user message
        await send(
            Div(
                ChatMessage(msg_count, msg.rstrip(), is_user=True),
                hx_swap_oob="beforeend",
                id="chatlist",
            )
        )
        msg_count += 1

        # Send initial empty assistant message
        await send(
            Div(
                ChatMessage(msg_count, ""),
                hx_swap_oob="beforeend",
                id="chatlist",
            )
        )

        # Get streaming response generator
        response_generator = await call_model(query=msg.rstrip())

        async for chunk in response_generator:
            if chunk["type"] == "audio":
                # Make sure to send as a JSON string
                await send(json.dumps({"type": "audio", "data": chunk["data"]}))

            elif chunk["type"] == "transcript":
                # Update accumulated transcript and display
                accumulated_transcript += chunk["content"]
                await send(
                    Div(
                        accumulated_transcript,
                        id=f"chat-content-{msg_count}",
                        hx_swap_oob="true",
                        cls="chat-bubble chat-bubble-primary",
                    )
                )

        # Clear input after completion
        await send(ChatInput())

    except Exception as e:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    serve()
