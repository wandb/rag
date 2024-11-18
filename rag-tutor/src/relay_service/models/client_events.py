from typing import Literal, Optional, Dict, List, Union

from pydantic import BaseModel, Field

from src.relay_service.models import ClientEventTypes
from src.relay_service.models.base import BaseEvent


class TurnDetection(BaseModel):
    type: Literal["server_vad"]
    threshold: float = Field(..., ge=0.0, le=1.0)
    prefix_padding_ms: int
    silence_duration_ms: int


class InputAudioTranscription(BaseModel):
    model: Optional[str] = None


class ToolParameterProperty(BaseModel):
    type: str
    description: str


class ToolParameter(BaseModel):
    type: str
    properties: Dict[str, ToolParameterProperty]
    required: List[str]


class Tool(BaseModel):
    type: Literal["function", "code_interpreter", "file_search"]
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[ToolParameter] = None


class Session(BaseModel):
    modalities: Optional[List[str]] = None
    instructions: Optional[str] = None
    voice: Optional[str] = None
    input_audio_format: Optional[str] = None
    output_audio_format: Optional[str] = None
    input_audio_transcription: Optional[InputAudioTranscription] = None
    turn_detection: Optional[TurnDetection] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None


class SessionUpdate(BaseEvent):
    type: Literal[ClientEventTypes.SESSION_UPDATE] = ClientEventTypes.SESSION_UPDATE
    session: Session


class InputAudioBufferAppend(BaseEvent):
    type: Literal[ClientEventTypes.INPUT_AUDIO_BUFFER_APPEND] = (
        ClientEventTypes.INPUT_AUDIO_BUFFER_APPEND
    )
    audio: str


class InputAudioBufferCommit(BaseEvent):
    type: Literal[ClientEventTypes.INPUT_AUDIO_BUFFER_COMMIT] = (
        ClientEventTypes.INPUT_AUDIO_BUFFER_COMMIT
    )


class InputAudioBufferClear(BaseEvent):
    type: Literal[ClientEventTypes.INPUT_AUDIO_BUFFER_CLEAR] = (
        ClientEventTypes.INPUT_AUDIO_BUFFER_CLEAR
    )


class MessageContent(BaseModel):
    """Content for message-type items only"""

    type: Literal["input_text", "input_audio", "text"]
    text: Optional[str] = None
    audio: Optional[str] = None
    transcript: Optional[str] = None


class ConversationItem(BaseModel):
    """Represents an item in the conversation"""

    id: Optional[str] = None
    type: Literal["message", "function_call", "function_call_output"]
    status: Optional[Literal["completed", "incomplete"]] = None
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[List[MessageContent]] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None


class ConversationItemCreate(BaseEvent):
    type: Literal[ClientEventTypes.CONVERSATION_ITEM_CREATE] = (
        ClientEventTypes.CONVERSATION_ITEM_CREATE
    )
    previous_item_id: Optional[str] = None
    item: ConversationItem


class ConversationItemTruncate(BaseEvent):
    type: Literal[ClientEventTypes.CONVERSATION_ITEM_TRUNCATE] = (
        ClientEventTypes.CONVERSATION_ITEM_TRUNCATE
    )
    item_id: str
    content_index: int
    audio_end_ms: int


class ConversationItemDelete(BaseEvent):
    type: Literal[ClientEventTypes.CONVERSATION_ITEM_DELETE] = (
        ClientEventTypes.CONVERSATION_ITEM_DELETE
    )
    item_id: str


class ResponseCreate(BaseEvent):
    type: Literal[ClientEventTypes.RESPONSE_CREATE] = ClientEventTypes.RESPONSE_CREATE


class ResponseCancel(BaseEvent):
    type: Literal[ClientEventTypes.RESPONSE_CANCEL] = ClientEventTypes.RESPONSE_CANCEL


ClientEvent = Union[
    SessionUpdate,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    InputAudioBufferClear,
    ConversationItemCreate,
    ConversationItemTruncate,
    ConversationItemDelete,
    ResponseCreate,
    ResponseCancel,
]
