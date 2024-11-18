import base64
import io
import json
import wave

import weave

from src.relay_service.models import BaseEvent, ServerEventTypes, ClientEventTypes


class StreamingWavWriter:
    """Writes audio integer or byte array chunks to a WAV file."""

    wav_file = None
    buffer = None
    in_memory = False

    def __init__(
        self,
        channels,
        sample_width,
        framerate,
    ):
        self.buffer = io.BytesIO()
        self.wav_file = wave.open(self.buffer, "wb")
        self.wav_file.setnchannels(channels)
        self.wav_file.setsampwidth(sample_width)
        self.wav_file.setframerate(framerate)

    def append_int16_chunk(self, int16_data):
        if int16_data is not None:
            self.wav_file.writeframes(int16_data)

    def close(self):
        self.wav_file.close()

    def get_wav_buffer(self):
        return self.buffer


class WeaveLogger:
    def __init__(self, project: str, entity: str | None = None):
        self.project_name = project if entity is None else f"{entity}/{project}"
        self.weave_client = weave.init(self.project_name)
        self.call_map = {}
        self.inputs = {}
        self.outputs = {}
        self.response_audio_storage = {}
        self.attributes = {}
        self.has_function_call = False

    def update_attributes(self, attributes: dict):
        self.attributes.update(attributes)

    def reset(self):
        self.call_map = {}
        self.inputs = {}
        self.outputs = {}

    def log(self, reset=True):
        if not self.has_function_call:
            call = self.weave_client.create_call(
                op="conversation_turn", inputs=self.inputs, attributes=self.attributes
            )

            self.weave_client.finish_call(call, output=self.outputs)
            if reset:
                self.reset()

    def add_output_audio(self, event_data):

        if event_data.response_id not in self.response_audio_storage:
            self.response_audio_storage[event_data.response_id] = StreamingWavWriter(
                1, 2, 24000
            )
        self.response_audio_storage[event_data.response_id].append_int16_chunk(
            base64.b64decode(event_data.delta)
        )

    def commit_output_audio(self, event_data):
        if event_data.response_id in self.response_audio_storage:

            wav_stream = self.response_audio_storage[event_data.response_id]
            wav_stream.close()
            wav_stream.buffer.seek(0)
            assistant_outputs = self.outputs.get("Assistant", [])
            assistant_outputs.append(
                {"Audio": wave.open(wav_stream.get_wav_buffer(), "rb")}
            )
            self.outputs.update({"Assistant": assistant_outputs})
            del self.response_audio_storage[event_data.response_id]

    def log_input_audio(self, event_data):
        base64_pcm = event_data.audio
        pcm_audio = base64.b64decode(base64_pcm)
        wav_stream = StreamingWavWriter(1, 2, 24000)
        wav_stream.append_int16_chunk(pcm_audio)
        wav_stream.close()
        wav_stream.buffer.seek(0)
        user_inputs = self.inputs.get("User", {})
        user_inputs.update({"Audio": wave.open(wav_stream.get_wav_buffer(), "rb")})
        self.inputs.update({"User": user_inputs})


weave_logger = WeaveLogger("realtime-demo-dev", "parambharat")


def log_to_weave(event_data: BaseEvent):
    match event_data.type:
        case ServerEventTypes.SESSION_CREATED:
            weave_logger.update_attributes({"session": {"id": event_data.event_id}})
        case ServerEventTypes.SESSION_UPDATED:
            session_dict = weave_logger.attributes.get("session", {})
            session_dict["session"] = event_data.session
            weave_logger.update_attributes({"session": session_dict})
        case ServerEventTypes.CONVERSATION_ITEM_CREATED:
            if event_data.item.type == "function_call":
                weave_logger.has_function_call = True
            elif event_data.item.type == "function_call_output":
                weave_logger.has_function_call = False

        case ClientEventTypes.CONVERSATION_ITEM_CREATE:
            if (
                event_data.item.role == "user"
                and event_data.item.content[0].type == "input_text"
            ):
                user_input = weave_logger.inputs.get("User", {})
                user_input.update({"Text": event_data.item.content[0].text})
                weave_logger.inputs.update({"User": user_input})

            elif event_data.item.type == "function_call_output":
                assistant_output = weave_logger.outputs.get("Assistant", [])
                fn_call_dict = {
                    "Outputs": event_data.model_dump(mode="json", exclude_none=True)
                }
                assistant_output.append({"Function Call": fn_call_dict})
                weave_logger.outputs.update({"Assistant": assistant_output})

        case ClientEventTypes.INPUT_AUDIO_BUFFER_APPEND:
            weave_logger.log_input_audio(event_data)

        case ServerEventTypes.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
            user_inputs = weave_logger.inputs.get("User", {})
            user_inputs.update({"Transcript: ": event_data.transcript})
            weave_logger.inputs.update({"User": user_inputs})

        case ServerEventTypes.RESPONSE_AUDIO_DELTA:
            weave_logger.add_output_audio(event_data)

        case ServerEventTypes.RESPONSE_AUDIO_DONE:
            weave_logger.commit_output_audio(event_data)

        case ServerEventTypes.RESPONSE_AUDIO_TRANSCRIPT_DONE:
            assistant_outputs = weave_logger.outputs.get("Assistant", [])
            assistant_outputs.append({"Transcript": event_data.transcript})
            weave_logger.outputs.update({"Assistant": assistant_outputs})

        case ServerEventTypes.RESPONSE_DONE:
            assistant_outputs = weave_logger.outputs.get("Assistant", [])
            response_output = event_data.response.output
            for item in response_output:
                if item.type == "function_call":
                    fn_args = json.loads(item.arguments)
                    item_dict = item.model_dump(
                        mode="json", exclude={"arguments"}, exclude_none=True
                    )
                    item_dict["arguments"] = fn_args
                    fn_call_dict = {"Inputs": item_dict}
                    assistant_outputs.append({"Function Call": fn_call_dict})
                    weave_logger.has_function_call = True
            assistant_outputs.append(
                event_data.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude={"response": {"output": {"__all__": {"content"}}}},
                )
            )
            weave_logger.log()
