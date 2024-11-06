import asyncio
import json
import os

import websockets

from src.components.models import (
    ClientEventTypes, InputAudioTranscription, ResponseCreate, ServerEventTypes, Session, SessionUpdate,
    parse_server_event)


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
                try:
                    message_data = json.loads(message)
                    parsed_event = parse_server_event(message_data)

                    # Handle different event types
                    if parsed_event.type == ServerEventTypes.SESSION_CREATED:
                        # Session has been created
                        print("Session created:", parsed_event.event_id)
                        if self.message_callback:
                            await self.message_callback(parsed_event)

                    elif parsed_event.type == ServerEventTypes.SESSION_UPDATED:
                        # Session has been updated
                        print("Session updated:", parsed_event.event_id)
                        if self.message_callback:
                            await self.message_callback(parsed_event)

                    elif parsed_event.type == ServerEventTypes.CONVERSATION_CREATED:
                        # New conversation started
                        print("Conversation created:", parsed_event.conversation.id)
                        if self.message_callback:
                            await self.message_callback(parsed_event)

                    elif (
                        parsed_event.type == ServerEventTypes.CONVERSATION_ITEM_CREATED
                    ):
                        print("Conversation item created:", parsed_event.event_id)
                        if self.message_callback:
                            await self.message_callback(parsed_event)

                    elif (
                        parsed_event.type
                        == ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DONE
                    ):
                        # Assistant's response transcript is complete
                        print(
                            f"Assistant's response transcript is complete: {parsed_event.event_id}"
                        )
                        if self.message_callback:
                            await self.message_callback(parsed_event)

                    elif (
                        parsed_event.type
                        == ServerEventTypes.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED
                    ):
                        # User's audio has been transcribed
                        print(
                            "User audio transcription completed:", parsed_event.event_id
                        )
                        # Create and send response create event
                        response_event = ResponseCreate(
                            type=ClientEventTypes.RESPONSE_CREATE
                        )
                        await self.send(
                            response_event.model_dump_json(exclude_none=True)
                        )
                        if self.message_callback:
                            await self.message_callback(parsed_event)

                    elif parsed_event.type == ServerEventTypes.RESPONSE_AUDIO_DELTA:
                        print(
                            f"Received chunk of assistant's audio response: {parsed_event.event_id}"
                        )
                        # Received chunk of assistant's audio response
                        if self.message_callback:
                            await self.message_callback(parsed_event)

                    elif parsed_event.type == ServerEventTypes.RESPONSE_DONE:
                        print(
                            f"Assistant's complete response is done: {parsed_event.event_id}"
                        )
                        # Assistant's complete response is done
                        if self.message_callback:
                            await self.message_callback(parsed_event)

                    elif parsed_event.type == ServerEventTypes.ERROR:
                        print(
                            f"Error from server: {parsed_event.error.model_dump_json(exclude_none=True)}"
                        )
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
