import json

from fasthtml.common import *

from pipeline.generation import call_model

# Set up the app with DaisyUI and Tailwind CSS
tlink = Script(src="https://cdn.tailwindcss.com")
dlink = Link(
    rel="stylesheet",
    href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css",
)
app, rt = fast_app(hdrs=(tlink, dlink, picolink), exts="ws")


# Chat message component
def ChatMessage(msg):
    bubble_class = (
        "chat-bubble-primary" if msg["sender"] == "User" else "chat-bubble-secondary"
    )
    chat_class = "chat-end" if msg["sender"] == "User" else "chat-start"
    return Div(
        Div(msg["sender"], cls="chat-header"),
        Div(msg["content"], cls=f"chat-bubble {bubble_class}"),
        cls=f"chat {chat_class}",
    )


@rt("/")
def get():
    return Titled(
        "RagTutor",
        Div(
            H1("RagTutor Chat"),
            Div(id="chatlist", cls="chat-box h-[73vh] overflow-y-auto p-4"),
            Form(
                Group(
                    Input(
                        type="text",
                        id="message-input",
                        name="msg",
                        placeholder="Type your message...",
                        cls="input input-bordered w-full",
                    ),
                    Button("Send", type="submit", cls="btn btn-primary"),
                ),
                id="chat-form",
                cls="flex space-x-2 mt-2 p-4",
            ),
            Audio(id="response-audio", controls=True, style="display: none"),
            cls="max-w-3xl mx-auto",
        ),
        Script(
            """
            const socket = new WebSocket(`ws://${window.location.host}/ws`);
            const chatList = document.getElementById('chatlist');
            const chatForm = document.getElementById('chat-form');
            const messageInput = document.getElementById('message-input');
            const audioPlayer = document.getElementById('response-audio');

            function addMessage(message) {
                const messageElement = document.createElement('div');
                const bubbleClass = message.sender === 'User' ? 'chat-bubble-primary' : 'chat-bubble-secondary';
                const chatClass = message.sender === 'User' ? 'chat-end' : 'chat-start';
                
                messageElement.className = `chat ${chatClass}`;
                messageElement.innerHTML = `
                    <div class="chat-header">${message.sender}</div>
                    <div class="chat-bubble ${bubbleClass}">${message.text}</div>
                `;
                
                chatList.appendChild(messageElement);
                chatList.scrollTop = chatList.scrollHeight;
            }

            socket.onmessage = function(event) {
                const message = JSON.parse(event.data);
                addMessage(message);
                
                if (message.audio) {
                    const audioData = atob(message.audio);
                    const arrayBuffer = new ArrayBuffer(audioData.length);
                    const view = new Uint8Array(arrayBuffer);
                    for (let i = 0; i < audioData.length; i++) {
                        view[i] = audioData.charCodeAt(i);
                    }
                    const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(blob);
                    audioPlayer.src = audioUrl;
                    audioPlayer.style.display = 'block';
                    audioPlayer.play();
                }
            };

            chatForm.onsubmit = function(e) {
                e.preventDefault();
                if (messageInput.value) {
                    const message = {
                        sender: 'User',
                        text: messageInput.value
                    };
                    // Add message to chat immediately
                    addMessage(message);
                    // Send to server
                    socket.send(JSON.stringify(message));
                    messageInput.value = '';
                }
            };

            // Handle WebSocket connection status
            socket.onopen = function(e) {
                console.log("WebSocket connection established");
            };

            socket.onclose = function(e) {
                console.log("WebSocket connection closed");
            };

            socket.onerror = function(e) {
                console.error("WebSocket error:", e);
            };
            """
        ),
    )


@app.websocket_route("/ws")
async def websocket_endpoint(websocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            user_message = json.loads(data)

            # Call the model
            response = await call_model(query=user_message["text"])

            # Send assistant response
            response_data = {
                "sender": "RagTutor",
                "text": (
                    response["transcript"]
                    if response["transcript"]
                    else response["text_response"]
                ),
            }
            if response.get("audio_data"):
                response_data["audio"] = response.get("audio_data")

            await websocket.send_text(json.dumps(response_data))

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    serve()
