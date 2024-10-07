# A collection of APIs to use for generating model responses
# OpenAI/Anthropic models are supported via their respective APIs below
# Other models are routed to LocalAPI, which assumes a VLLM instance running on localhost:8000
# Additional APIs (e.g., Google, Together, etc.) may need to be added
# Only asyncronous apis are supported
# Non-asnyc requests can be handled, but chat() should still be async, so include something like await asyncio.sleep(0)

from abc import ABC, abstractmethod
from typing import List, Dict
import os

import backoff
import openai
import anthropic


class ChatAPI(ABC):

    @abstractmethod
    def __init__(self, model):
        pass

    @abstractmethod
    async def chat(self, messages):
        pass


class OpenAIAPI(ChatAPI):

    def __init__(self, model: str):
        self.model = model
        self.client = openai.AsyncClient(
            api_key=os.environ.get("OPENAI_API_KEY"))

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        
        if self.model.startswith("o1"):
            if messages[0]["role"] == "system":
                system_message = messages.pop(0)["content"]
                user_message = messages[0]["content"]
                messages[0] = {
                    "role": "user",
                    "content": f"<|BEGIN_SYSTEM_MESSAGE|>\n{system_message.strip()}\n<|END_SYSTEM_MESSAGE|>\n\n{user_message}"
                }

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        
        else:   
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
        return response.choices[0].message.content


class AnthropicAPI(ChatAPI):

    def __init__(self, model: str):
        self.model = model
        self.client = anthropic.AsyncClient(
            api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @backoff.on_exception(backoff.fibo, anthropic.AnthropicError, max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        else:
            system_message = ""

        response = await self.client.messages.create(
            model=self.model,
            messages=messages,
            system=system_message,
            # max_tokens=8192, # 4096 for claude-3-*
            **kwargs,
        )
        
        return response.content[0].text


class GeminiAPI(OpenAIAPI):
    """Google Gemini API.
    
    Models include gemini-1.5-pro-001 etc.
    """
    def __init__(self, model: str):
        import google.auth
        import google.auth.transport.requests

        # Programmatically get an access token, need to setup your google cloud account properly,
        # and get `gcloud auth application-default login` to be run first
        creds, _project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        # Note: the credential lives for 1 hour by default (https://cloud.google.com/docs/authentication/token-types#at-lifetime); after expiration, it must be refreshed.

        project_id = creds.quota_project_id
        # Pass the Vertex endpoint and authentication to the OpenAI SDK
        self.model = f"google/{model}"
        self.client = openai.AsyncClient(
            base_url=f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/us-central1/endpoints/openapi",
            api_key=creds.token,
        )

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=10, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        message = response.choices[0].message
        if message is None:
            # This happens for Google Gemini under high concurrency
            raise openai.OpenAIError("No response from Google Gemini")
        return message.content


class TogetherAPI(ChatAPI):

    def __init__(self, model: str):
        self.model = model
        self.client = openai.AsyncClient(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1"
        )

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content
    

class LocalAPI(ChatAPI):

    def __init__(self, model: str):
        self.model = model
        self.client = openai.AsyncClient(base_url="http://localhost:8000/v1", api_key="EMPTY")

    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content
    
    @backoff.on_exception(backoff.fibo, (openai.OpenAIError), max_tries=5, max_value=30)
    async def complete(self, prompt: str, **kwargs) -> str:
        response = await self.client.completions.create(
            model=self.model,
            prompt=prompt,
            **kwargs,
        )
        return response.choices[0].text


def get_chat_api_from_model(model: str) -> ChatAPI:
    if model.startswith("gpt") or model.startswith("o1"):
        return OpenAIAPI(model)
    if model.startswith("claude"):
        return AnthropicAPI(model)
    if model.startswith("gemini"):
        return GeminiAPI(model)
    if model == "meta-llama/Meta-Llama-3.1-405B-Instruct":
        return TogetherAPI(model + "-turbo")
    return LocalAPI(model)
