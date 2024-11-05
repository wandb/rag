import asyncio
import json
import os

import websockets

from src.components.models import (
    InputAudioTranscription,
    Session,
    SessionUpdate,
    parse_server_event,
)


class OpenAIRealtimeClient:
    def __init__(self, message_callback=None):
        self.ws = None
        self.url = (
            f"wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        )
        self.task = None
        self.message_callback = message_callback

    async def connect(self):
        """Establish WebSocket connection"""
        self.ws = await websockets.connect(
            self.url,
            extra_headers={
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                "OpenAI-Beta": "realtime=v1",
            },
        )

        # Send initial configuration message
        config_event = SessionUpdate(
            session=Session(
                modalities=["text", "audio"],
                input_audio_transcription=InputAudioTranscription(model="whisper-1"),
                turn_detection=None,
            )
        )

        update_event = config_event.model_dump(exclude_none=True, mode="json")

        update_event["session"]["turn_detection"] = None
        update_event = json.dumps(update_event)

        await self.send(update_event)

    async def start(self):
        """Start the WebSocket client"""
        try:
            await self.connect()
            # Create the task but don't await it
            self.task = asyncio.create_task(self.receive_messages())
            # Don't await self.task here
        except Exception as e:
            print(f"WebSocket error: {e}")
            if self.ws:
                await self.ws.close()

    async def stop(self):
        """Stop the WebSocket client"""
        if self.task:
            self.task.cancel()
        if self.ws:
            await self.ws.close()

    async def send(self, data):
        """Send data through WebSocket"""
        if self.ws:
            await self.ws.send(data)

    async def receive_messages(self):
        """Receive and process WebSocket messages"""
        try:
            async for message in self.ws:
                print("Received message from OpenAI:", message)
                try:
                    # Parse the message string as JSON
                    message_data = json.loads(message)
                    # Parse the event using our models
                    parsed_event = parse_server_event(message_data)

                    if self.message_callback:
                        await self.message_callback(parsed_event)
                except ValueError as e:
                    print(f"Error parsing event: {e}")
                except Exception as e:
                    print(f"Unexpected error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except Exception as e:
            print(f"Error receiving message: {e}")
