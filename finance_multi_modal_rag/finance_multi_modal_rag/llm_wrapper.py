import os
from typing import List, Optional, Union

import weave
from mistralai import Mistral
from openai import OpenAI

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
]


class MultiModalPredictor(weave.Model):
    model_name: str
    base_url: Optional[str] = None
    _llm_client: Union[OpenAI, Mistral] = None

    def __init__(self, model_name: str, base_url: Optional[str] = None):
        super().__init__(model_name=model_name, base_url=base_url)
        if self.model_name in OPENAI_MODELS:
            self._llm_client = OpenAI(
                base_url=self.base_url, api_key=os.environ["OPENAI_API_KEY"]
            )
        else:
            self._llm_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    @weave.op()
    def format_user_prompts(self, prompts: List[str]):
        content = []
        for prompt in prompts:
            if prompt.startswith("data:image/png;base64,") or prompt.startswith(
                "data:image/jpeg;base64,"
            ):
                if self.model_name in OPENAI_MODELS:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": prompt, "detail": "high"},
                        }
                    )
                else:
                    content.append({"type": "image_url", "image_url": prompt})
            else:
                content.append({"type": "text", "text": prompt})
        return content

    @weave.op()
    def predict(
        self, user_prompts: List[str], system_prompt: Optional[str] = None, **kwargs
    ):
        messages = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        user_prompt_contents = self.format_user_prompts(user_prompts)
        messages.append({"role": "user", "content": user_prompt_contents})
        if self.model_name in OPENAI_MODELS:
            return (
                self._llm_client.chat.completions.create(
                    model=self.model_name, messages=messages, **kwargs
                )
                .choices[0]
                .message.content
            )
        else:
            return (
                self._llm_client.chat.complete(
                    model=self.model_name, messages=messages, **kwargs
                )
                .choices[0]
                .message.content
            )
