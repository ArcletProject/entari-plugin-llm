import time
from typing import Literal, overload

import litellm
from arclet.entari import add_service
from launart import Launart, Service
from launart.status import Phase

from ._types import Message
from .config import get_model_config
from .log import log


class LLMService(Service):
    id = "entari_plugin_llm"

    def __init__(self):
        super().__init__()
        self.total_tokens = 0
        self.total_calls = 0
        self.start_time = 0

    @property
    def required(self) -> set[str]:
        return set()

    @property
    def stages(self) -> set[Phase]:
        return {"preparing", "blocking", "cleanup"}

    def _build_payload(
        self,
        messages: list[Message],
        stream: bool,
        system: str | None = None,
        model: str | None = None,
        **kwargs,
    ) -> dict:
        conf = get_model_config(model)

        if system or conf.prompt:
            messages.insert(0, {"role": "system", "content": system or conf.prompt})

        return {
            "model": conf.name,
            "messages": messages,
            "stream": stream,
            "base_url": conf.base_url,
            "api_key": conf.api_key,
            **conf.extra,
            **kwargs,
        }

    def _update_usage(self, response: litellm.ModelResponse | litellm.ModelResponseStream):
        usage: litellm.Usage | None = response.get("usage", None)
        if usage:
            self.total_tokens += usage.total_tokens

    async def _stream_wrapper(self, stream_resp: litellm.CustomStreamWrapper):
        async for chunk in stream_resp:
            usage = chunk.get("usage", None)
            if usage:
                self._update_usage(chunk)
            yield chunk

    @overload
    async def generate(
        self,
        message: str | list[Message],
        *,
        stream: Literal[False] = False,
        system: str | None = None,
        model: str | None = None,
        **kwargs,
    ) -> litellm.ModelResponse:
        ...

    @overload
    async def generate(
        self,
        message: str | list[Message],
        *,
        stream: Literal[True],
        system: str | None = None,
        model: str | None = None,
        **kwargs,
    ) -> litellm.CustomStreamWrapper:
        ...

    @overload
    async def generate(
        self,
        message: str | list[Message],
        *,
        stream: bool,
        system: str | None = None,
        model: str | None = None,
        **kwargs,
    ) -> litellm.ModelResponse | litellm.CustomStreamWrapper:
        ...

    async def generate(
        self,
        message: str | list[Message],
        *,
        stream: bool = False,
        system: str | None = None,
        model: str | None = None,
        **kwargs,
    ) ->  litellm.ModelResponse | litellm.CustomStreamWrapper:
        if isinstance(message, str):
            message = [{"role": "user", "content": message}]

        payload = self._build_payload(
            messages=message,
            stream=stream,
            system=system,
            model=model,
            **kwargs,
        )

        response = await litellm.acompletion(**payload)

        self.total_calls += 1

        if not stream:
            self._update_usage(response) # type: ignore
            return response
        else:
            return self._stream_wrapper(response) # type: ignore

    async def launch(self, manager: Launart):
        async with self.stage("preparing"):
            litellm.drop_params = True
            self.start_time = time.time()

        async with self.stage("blocking"):
            await manager.status.wait_for_sigexit()

        async with self.stage("cleanup"):
            uptime = time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time))
            log(
                "success",
                f"运行统计: 耗时 [ {uptime} ] | 总请求 [ {self.total_calls} ] | 预估总 Token [ {self.total_tokens} ]"
            )

llm = LLMService()

add_service(llm)
