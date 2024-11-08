from typing import Optional, Literal, Dict, Any, Union, List

from pydantic import BaseModel

from src.relay_service.models import ServerEventTypes
from src.relay_service.models.base import BaseEvent
from src.relay_service.models.client_events import Session


class ErrorDetails(BaseModel):
    type: Optional[str] = None
    code: Optional[str] = None
    message: Optional[str] = None
    param: Optional[str] = None


class ErrorEvent(BaseEvent):
    type: Literal[ServerEventTypes.ERROR] = ServerEventTypes.ERROR
    error: ErrorDetails


class SessionCreated(BaseEvent):
    type: Literal[ServerEventTypes.SESSION_CREATED] = ServerEventTypes.SESSION_CREATED
    session: Session


class SessionUpdated(BaseEvent):
    type: Literal[ServerEventTypes.SESSION_UPDATED] = ServerEventTypes.SESSION_UPDATED
    session: Session


class Conversation(BaseModel):
    id: str
    object: Literal["realtime.conversation"]


class ConversationCreated(BaseEvent):
    type: Literal[ServerEventTypes.CONVERSATION_CREATED] = (
        ServerEventTypes.CONVERSATION_CREATED
    )
    conversation: Conversation


class ConversationItemContent(BaseModel):
    type: Literal["text", "input_text", "input_audio"]
    text: Optional[str] = None
    audio: Optional[str] = None
    transcript: Optional[str] = None


class ConversationItem(BaseModel):
    id: str
    type: Optional[Literal["message", "function_call", "function_call_output"]] = None
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[List[ConversationItemContent]] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None


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


class TokenDetails(BaseModel):
    cached_tokens: Optional[int] = 0
    text_tokens: Optional[int] = 0
    audio_tokens: Optional[int] = 0


class ResponseUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int
    input_token_details: Optional[Dict[str, Union[int, TokenDetails]]] = None
    output_token_details: Optional[Dict[str, Union[int, TokenDetails]]] = None


class ResponseOutputContent(BaseModel):
    type: Literal["text", "audio"]
    text: Optional[str] = None
    audio: Optional[str] = None
    transcript: Optional[str] = None


class ResponseOutput(BaseModel):
    id: str
    object: Literal["realtime.item"]
    type: Literal["message", "function_call", "function_call_output"]
    status: Optional[str] = None
    role: Optional[str] = None
    content: Optional[List[ResponseOutputContent]] = None
    # Function call fields
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None


class ResponseContentPart(BaseModel):
    type: str
    text: Optional[str] = None


class ResponseStatusDetails(BaseModel):
    type: str
    reason: str


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


class ResponseOutputItemContent(BaseModel):
    type: Literal["audio", "text"]
    text: Optional[str] = None
    audio: Optional[str] = None
    transcript: Optional[str] = None


class ResponseOutputItem(BaseModel):
    id: str
    type: Literal["message", "function_call", "function_call_output"]
    status: Optional[Literal["completed", "incomplete", "in_progress"]] = None
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[List[ResponseOutputItemContent]] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None


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


class ResponseAudioTranscriptDone(BaseEvent):
    type: Literal[ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DONE] = (
        ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DONE
    )
    response_id: str
    item_id: str
    transcript: str


class ResponseAudioTranscriptDelta(BaseEvent):
    type: Literal[ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DELTA] = (
        ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DELTA
    )
    response_id: str
    item_id: str
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
    ResponseFunctionCallArgumentsDelta,
    ResponseFunctionCallArgumentsDone,
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
    ServerEventTypes.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA: ResponseFunctionCallArgumentsDelta,
    ServerEventTypes.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE: ResponseFunctionCallArgumentsDone,
}
