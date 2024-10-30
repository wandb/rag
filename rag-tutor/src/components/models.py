from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ValidationError


class BaseEvent(BaseModel):
    type: Union["ClientEventTypes", "ServerEventTypes"]
    event_id: Optional[str] = None  # Add event_id as an optional field for all events


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: float


""" CLIENT EVENTS """


class ClientEventTypes(str, Enum):
    SESSION_UPDATE = "session.update"
    CONVERSATION_ITEM_CREATE = "conversation.item.create"
    CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
    CONVERSATION_ITEM_DELETE = "conversation.item.delete"
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
    ERROR = "error"


#### Session Update
class TurnDetection(BaseModel):
    type: Literal["server_vad"]
    threshold: float = Field(..., ge=0.0, le=1.0)
    prefix_padding_ms: int
    silence_duration_ms: int


class InputAudioTranscription(BaseModel):
    model: Optional[str] = None


class ToolParameterProperty(BaseModel):
    type: str


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


#### Audio Buffers
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


#### Messages
class MessageContent(BaseModel):
    type: Literal["input_audio"]
    audio: str


class ConversationItemContent(BaseModel):
    type: Literal["input_text", "input_audio", "text", "audio"]
    text: Optional[str] = None
    audio: Optional[str] = None
    transcript: Optional[str] = None


class FunctionCallContent(BaseModel):
    call_id: str
    name: str
    arguments: str


class FunctionCallOutputContent(BaseModel):
    output: str


class ConversationItem(BaseModel):
    id: Optional[str] = None
    type: Literal["message", "function_call", "function_call_output"]
    status: Optional[Literal["completed", "in_progress", "incomplete"]] = None
    role: Literal["user", "assistant", "system"]
    content: List[
        Union[ConversationItemContent, FunctionCallContent, FunctionCallOutputContent]
    ]
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None


class ConversationItemCreate(BaseEvent):
    type: Literal[ClientEventTypes.CONVERSATION_ITEM_CREATE] = (
        ClientEventTypes.CONVERSATION_ITEM_CREATE
    )
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


#### Responses
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

""" SERVER EVENTS """


class ServerEventTypes(str, Enum):
    ERROR = "error"
    RESPONSE_AUDIO_TRANSCRIPT_DONE = "response.audio_transcript.done"
    RESPONSE_AUDIO_TRANSCRIPT_DELTA = "response.audio_transcript.delta"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    CONVERSATION_CREATED = "conversation.created"
    INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
    INPUT_AUDIO_BUFFER_CLEARED = "input_audio_buffer.cleared"
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = (
        "conversation.item.input_audio_transcription.completed"
    )
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED = (
        "conversation.item.input_audio_transcription.failed"
    )
    CONVERSATION_ITEM_TRUNCATED = "conversation.item.truncated"
    CONVERSATION_ITEM_DELETED = "conversation.item.deleted"
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_TEXT_DONE = "response.text.done"
    RESPONSE_AUDIO_DONE = "response.audio.done"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
    RATE_LIMITS_UPDATED = "rate_limits.updated"


#### Errors
class ErrorDetails(BaseModel):
    type: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None


class ErrorEvent(BaseEvent):
    type: Literal[ServerEventTypes.ERROR] = ServerEventTypes.ERROR
    error: ErrorDetails


#### Session
class SessionCreated(BaseEvent):
    type: Literal[ServerEventTypes.SESSION_CREATED] = ServerEventTypes.SESSION_CREATED
    session: Session


class SessionUpdated(BaseEvent):
    type: Literal[ServerEventTypes.SESSION_UPDATED] = ServerEventTypes.SESSION_UPDATED
    session: Session


#### Conversation
class Conversation(BaseModel):
    id: str
    object: Literal["realtime.conversation"]


class ConversationCreated(BaseEvent):
    type: Literal[ServerEventTypes.CONVERSATION_CREATED] = (
        ServerEventTypes.CONVERSATION_CREATED
    )
    conversation: Conversation


class ConversationItemCreated(BaseEvent):
    type: Literal[ServerEventTypes.CONVERSATION_ITEM_CREATED] = (
        ServerEventTypes.CONVERSATION_ITEM_CREATED
    )
    previous_item_id: Optional[str] = None
    item: ConversationItem


class ConversationItemInputAudioTranscriptionCompleted(BaseEvent):
    type: Literal[
        ServerEventTypes.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED
    ] = ServerEventTypes.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED
    item_id: str
    content_index: int
    transcript: str


class ConversationItemInputAudioTranscriptionFailed(BaseEvent):
    type: Literal[
        ServerEventTypes.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED
    ] = ServerEventTypes.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED
    item_id: str
    content_index: int
    error: Dict[str, Any]


class ConversationItemTruncated(BaseEvent):
    type: Literal[ServerEventTypes.CONVERSATION_ITEM_TRUNCATED] = (
        ServerEventTypes.CONVERSATION_ITEM_TRUNCATED
    )
    item_id: str
    content_index: int
    audio_end_ms: int


class ConversationItemDeleted(BaseEvent):
    type: Literal[ServerEventTypes.CONVERSATION_ITEM_DELETED] = (
        ServerEventTypes.CONVERSATION_ITEM_DELETED
    )
    item_id: str


#### Response
class ResponseUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int
    input_token_details: Optional[Dict[str, int]] = None
    output_token_details: Optional[Dict[str, int]] = None


class ResponseOutput(BaseModel):
    id: str
    object: Literal["realtime.item"]
    type: str
    status: str
    role: str
    content: List[Dict[str, Any]]


class ResponseContentPart(BaseModel):
    type: str
    text: Optional[str] = None


class ResponseOutputItemContent(BaseModel):
    type: str
    text: Optional[str] = None


class ResponseStatusDetails(BaseModel):
    type: str
    reason: str


class ResponseOutputItem(BaseModel):
    id: str
    object: Literal["realtime.item"]
    type: str
    status: str
    role: str
    content: List[ResponseOutputItemContent]


class Response(BaseModel):
    id: str
    object: Literal["realtime.response"]
    status: str
    status_details: Optional[ResponseStatusDetails] = None
    output: List[ResponseOutput]
    usage: Optional[ResponseUsage]


class ResponseCreated(BaseEvent):
    type: Literal[ServerEventTypes.RESPONSE_CREATED] = ServerEventTypes.RESPONSE_CREATED
    response: Response


class ResponseDone(BaseEvent):
    type: Literal[ServerEventTypes.RESPONSE_DONE] = ServerEventTypes.RESPONSE_DONE
    response: Response


class ResponseOutputItemAdded(BaseEvent):
    type: Literal[ServerEventTypes.RESPONSE_OUTPUT_ITEM_ADDED] = (
        ServerEventTypes.RESPONSE_OUTPUT_ITEM_ADDED
    )
    response_id: str
    output_index: int
    item: ResponseOutputItem


class ResponseOutputItemDone(BaseEvent):
    type: Literal[ServerEventTypes.RESPONSE_OUTPUT_ITEM_DONE] = (
        ServerEventTypes.RESPONSE_OUTPUT_ITEM_DONE
    )
    response_id: str
    output_index: int
    item: ResponseOutputItem


class ResponseContentPartBase(BaseEvent):
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    part: ResponseContentPart


class ResponseContentPartAdded(ResponseContentPartBase):
    type: Literal[ServerEventTypes.RESPONSE_CONTENT_PART_ADDED] = (
        ServerEventTypes.RESPONSE_CONTENT_PART_ADDED
    )


class ResponseContentPartDone(ResponseContentPartBase):
    type: Literal[ServerEventTypes.RESPONSE_CONTENT_PART_DONE] = (
        ServerEventTypes.RESPONSE_CONTENT_PART_DONE
    )


#### Response Text
class ResponseTextBase(BaseEvent):
    response_id: str
    item_id: str
    output_index: int
    content_index: int


class ResponseTextDelta(ResponseTextBase):
    type: Literal[ServerEventTypes.RESPONSE_TEXT_DELTA] = (
        ServerEventTypes.RESPONSE_TEXT_DELTA
    )
    delta: str


class ResponseTextDone(ResponseTextBase):
    type: Literal[ServerEventTypes.RESPONSE_TEXT_DONE] = (
        ServerEventTypes.RESPONSE_TEXT_DONE
    )
    text: str


#### Response Audio
class ResponseAudioTranscriptDone(BaseEvent):
    type: Literal[ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DONE] = (
        ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DONE
    )
    transcript: str


class ResponseAudioTranscriptDelta(BaseEvent):
    type: Literal[ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DELTA] = (
        ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DELTA
    )
    delta: str


class ResponseAudioDelta(BaseEvent):
    type: Literal[ServerEventTypes.RESPONSE_AUDIO_DELTA] = (
        ServerEventTypes.RESPONSE_AUDIO_DELTA
    )
    response_id: str
    item_id: str
    delta: str


class ResponseAudioDone(BaseEvent):
    type: Literal[ServerEventTypes.RESPONSE_AUDIO_DONE] = (
        ServerEventTypes.RESPONSE_AUDIO_DONE
    )
    response_id: str
    item_id: str
    output_index: int
    content_index: int


class InputAudioBufferCommitted(BaseEvent):
    type: Literal[ServerEventTypes.INPUT_AUDIO_BUFFER_COMMITTED] = (
        ServerEventTypes.INPUT_AUDIO_BUFFER_COMMITTED
    )
    previous_item_id: Optional[str] = None
    item_id: Optional[str] = None
    event_id: Optional[str] = None


class InputAudioBufferCleared(BaseEvent):
    type: Literal[ServerEventTypes.INPUT_AUDIO_BUFFER_CLEARED] = (
        ServerEventTypes.INPUT_AUDIO_BUFFER_CLEARED
    )


class InputAudioBufferSpeechStarted(BaseEvent):
    type: Literal[ServerEventTypes.INPUT_AUDIO_BUFFER_SPEECH_STARTED] = (
        ServerEventTypes.INPUT_AUDIO_BUFFER_SPEECH_STARTED
    )
    audio_start_ms: int
    item_id: str


class InputAudioBufferSpeechStopped(BaseEvent):
    type: Literal[ServerEventTypes.INPUT_AUDIO_BUFFER_SPEECH_STOPPED] = (
        ServerEventTypes.INPUT_AUDIO_BUFFER_SPEECH_STOPPED
    )
    audio_end_ms: int
    item_id: str


#### Function Calls
class ResponseFunctionCallArgumentsDelta(BaseEvent):
    type: Literal[ServerEventTypes.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA] = (
        ServerEventTypes.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA
    )
    response_id: str
    item_id: str
    output_index: int
    call_id: str
    delta: str


class ResponseFunctionCallArgumentsDone(BaseEvent):
    type: Literal[ServerEventTypes.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE] = (
        ServerEventTypes.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE
    )
    response_id: str
    item_id: str
    output_index: int
    call_id: str
    arguments: str


#### Rate Limits
class RateLimit(BaseModel):
    name: str
    limit: int
    remaining: int
    reset_seconds: float


class RateLimitsUpdated(BaseEvent):
    type: Literal[ServerEventTypes.RATE_LIMITS_UPDATED] = (
        ServerEventTypes.RATE_LIMITS_UPDATED
    )
    rate_limits: List[RateLimit]


ServerEvent = Union[
    ErrorEvent,
    ConversationCreated,
    ResponseAudioTranscriptDone,
    ResponseAudioTranscriptDelta,
    ResponseAudioDelta,
    ResponseCreated,
    ResponseDone,
    ResponseOutputItemAdded,
    ResponseOutputItemDone,
    ResponseContentPartAdded,
    ResponseContentPartDone,
    ResponseTextDelta,
    ResponseTextDone,
    ResponseAudioDone,
    ConversationItemInputAudioTranscriptionCompleted,
    SessionCreated,
    SessionUpdated,
    InputAudioBufferCleared,
    InputAudioBufferSpeechStarted,
    InputAudioBufferSpeechStopped,
    ConversationItemCreated,
    ConversationItemInputAudioTranscriptionFailed,
    ConversationItemTruncated,
    ConversationItemDeleted,
    RateLimitsUpdated,
]

EVENT_TYPE_TO_MODEL = {
    ServerEventTypes.ERROR: ErrorEvent,
    ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DONE: ResponseAudioTranscriptDone,
    ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DELTA: ResponseAudioTranscriptDelta,
    ServerEventTypes.RESPONSE_AUDIO_DELTA: ResponseAudioDelta,
    ServerEventTypes.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED: ConversationItemInputAudioTranscriptionCompleted,
    ServerEventTypes.SESSION_CREATED: SessionCreated,
    ServerEventTypes.SESSION_UPDATED: SessionUpdated,
    ServerEventTypes.CONVERSATION_CREATED: ConversationCreated,
    ServerEventTypes.INPUT_AUDIO_BUFFER_COMMITTED: InputAudioBufferCommitted,
    ServerEventTypes.INPUT_AUDIO_BUFFER_CLEARED: InputAudioBufferCleared,
    ServerEventTypes.INPUT_AUDIO_BUFFER_SPEECH_STARTED: InputAudioBufferSpeechStarted,
    ServerEventTypes.INPUT_AUDIO_BUFFER_SPEECH_STOPPED: InputAudioBufferSpeechStopped,
    ServerEventTypes.CONVERSATION_ITEM_CREATED: ConversationItemCreated,
    ServerEventTypes.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED: ConversationItemInputAudioTranscriptionFailed,
    ServerEventTypes.CONVERSATION_ITEM_TRUNCATED: ConversationItemTruncated,
    ServerEventTypes.CONVERSATION_ITEM_DELETED: ConversationItemDeleted,
    ServerEventTypes.RESPONSE_CREATED: ResponseCreated,
    ServerEventTypes.RESPONSE_DONE: ResponseDone,
    ServerEventTypes.RESPONSE_OUTPUT_ITEM_ADDED: ResponseOutputItemAdded,
    ServerEventTypes.RESPONSE_OUTPUT_ITEM_DONE: ResponseOutputItemDone,
    ServerEventTypes.RESPONSE_CONTENT_PART_ADDED: ResponseContentPartAdded,
    ServerEventTypes.RESPONSE_CONTENT_PART_DONE: ResponseContentPartDone,
    ServerEventTypes.RESPONSE_TEXT_DELTA: ResponseTextDelta,
    ServerEventTypes.RESPONSE_TEXT_DONE: ResponseTextDone,
    ServerEventTypes.RESPONSE_AUDIO_DONE: ResponseAudioDone,
    ServerEventTypes.RATE_LIMITS_UPDATED: RateLimitsUpdated,
}


def parse_server_event(event_data: dict) -> ServerEvent:
    event_type = event_data.get("type")
    if not event_type:
        raise ValueError("Event data is missing 'type' field")

    model_class = EVENT_TYPE_TO_MODEL.get(event_type)
    if not model_class:
        raise ValueError(f"Unknown event type: {event_type}")

    try:
        return model_class(**event_data)
    except ValidationError as e:
        raise ValueError(f"Failed to parse event of type {event_type}: {str(e)}")
