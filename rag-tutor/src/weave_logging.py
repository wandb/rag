import io
import wave
from datetime import datetime

import weave


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
        self.inputs = {}
        self.outputs = {}
        self.attributes = {}

    def update_attributes(self, attributes: dict):
        self.attributes.update(attributes)

    def reset(self):
        self.inputs = {}
        self.outputs = {}

    @weave.op(name=f"conversation_turn-{datetime.now()}")
    def conversation_turn(self, User):
        return {"Assistant": self.outputs}

    def log(self, inputs, outputs, reset=True):
        self.inputs = inputs
        self.outputs = outputs
        with weave.attributes(self.attributes):
            _ = self.conversation_turn(User=self.inputs)
        # call = self.weave_client.create_call(
        #     op=operation, inputs=self.inputs, attributes=self.attributes
        # )
        #
        # self.weave_client.finish_call(call, output=self.outputs)
        if reset:
            self.reset()


weave_logger = WeaveLogger("realtime-demo-dev", "parambharat")
