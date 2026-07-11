import time
from typing import Any, Literal, TypeVar, overload

import litellm
from agno.agent import Agent
from agno.db.sqlite import AsyncSqliteDb
from agno.models.litellm import LiteLLM
from agno.models.message import Message
from agno.run.base import RunStatus
from arclet.entari import add_service, local_data
from arclet.letoderea.context import Contexts
from launart import Launart, Service
from launart.status import Phase

from ._callback import TokenUsageHandler
from .config import _conf, get_model_config
from .response import GenericResponse
from .sessions import AgnoSessionStore
from .tools.bridge import get_agno_tools

TOutput = TypeVar("TOutput")
OutputType = Literal["json_object"] | type[Any] | dict[str, Any]


class LLMService(Service):
    id = "entari_plugin_llm"

    def __init__(self) -> None:
        super().__init__()
        self.total_tokens = 0
        self.total_calls = 0
        self.start_time: float = 0.0
        self.usage_handler = TokenUsageHandler(self)
        self._db: AsyncSqliteDb | None = None
        self._sessions: AgnoSessionStore | None = None

    @property
    def required(self) -> set[str]:
        return set()

    @property
    def stages(self) -> set[Phase]:
        return {"preparing", "blocking", "cleanup"}

    @property
    def session_store(self) -> AgnoSessionStore:
        if self._sessions is None:
            raise RuntimeError("Agno session store is not initialized")
        return self._sessions

    @overload
    async def generate(
        self,
        message: str | list[Message],
        variables: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        system: str | None = None,
        model: str | None = None,
        output: type[TOutput],
        session_id: str | None = None,
        user_id: str | None = None,
        ignore_user_prompt: bool = False,
        ctx: Contexts | None = None,
        **kwargs: Any,
    ) -> GenericResponse[TOutput]: ...

    @overload
    async def generate(
        self,
        message: str | list[Message],
        variables: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        system: str | None = None,
        model: str | None = None,
        output: Literal["json_object"] | dict[str, Any],
        session_id: str | None = None,
        user_id: str | None = None,
        ignore_user_prompt: bool = False,
        ctx: Contexts | None = None,
        **kwargs: Any,
    ) -> GenericResponse[Any]: ...

    @overload
    async def generate(
        self,
        message: str | list[Message],
        variables: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        system: str | None = None,
        model: str | None = None,
        output: None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        ignore_user_prompt: bool = False,
        ctx: Contexts | None = None,
        **kwargs: Any,
    ) -> GenericResponse[None]: ...

    async def generate(
        self,
        message: str | list[Message],
        variables: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        system: str | None = None,
        model: str | None = None,
        output: OutputType | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        ignore_user_prompt: bool = False,
        ctx: Contexts | None = None,
        **kwargs: Any,
    ) -> GenericResponse[Any]:
        """Generate an LLM response through an Agno Agent."""
        if output is not None and stream:
            raise ValueError("output is not supported when stream=True")
        if session_id is not None and user_id is None:
            raise ValueError("user_id is required when session_id is provided")

        model_config = get_model_config(model)
        if system is not None:
            instructions = system
        elif ignore_user_prompt:
            instructions = None
        else:
            instructions = model_config.prompt or _conf.prompt or None

        if variables:
            variable_instructions = "下列是用以辅助你思考回答的变量：\n" + "\n".join(
                f"- **{key}**: {value!r}" for key, value in variables.items()
            )
            instructions = (
                f"{instructions}\n\n{variable_instructions}"
                if instructions
                else variable_instructions
            )

        request_params = {**model_config.extra, **kwargs}
        agno_model = LiteLLM(
            id=model_config.name,
            api_key=model_config.api_key,
            api_base=model_config.base_url,
            request_params=request_params or None,
        )

        agent_kwargs: dict[str, Any] = {
            "id": self.id,
            "model": agno_model,
            "instructions": instructions,
            "tools": get_agno_tools(ctx),
            "markdown": False,
        }
        if output is not None:
            agent_kwargs["output_schema"] = output if output != "json_object" else {"type": "object"}
            agent_kwargs["use_json_mode"] = True
        if self._db is not None:
            agent_kwargs["db"] = self._db
        if session_id is not None:
            agent_kwargs["session_id"] = session_id
            agent_kwargs["user_id"] = user_id
            agent_kwargs["add_history_to_context"] = True
            agent_kwargs["enable_session_summaries"] = True
            agent_kwargs["add_session_summary_to_context"] = False

        agent = Agent(**agent_kwargs)
        if stream:
            return GenericResponse.from_stream(agent.arun(message, stream=True))

        result = await agent.arun(message)
        if result.status is RunStatus.error:
            raise RuntimeError(str(result.content or "Agno agent run failed"))
        return GenericResponse.from_run_output(result, structured=output is not None)

    async def vision(
        self,
        image_url: str | dict[str, Any],
        system: str | None = None,
        model: str | None = None,
    ) -> GenericResponse[None]:
        """Describe an image through the standard generate path."""
        model_config = get_model_config(model)
        if not litellm.supports_vision(model_config.name):
            raise RuntimeError(f"Model {model_config.name} does not support vision input")

        if isinstance(image_url, str):
            url = image_url
        else:
            image = image_url.get("image_url", image_url)
            url = image["url"] if isinstance(image, dict) else str(image)
        message = Message(
            role="user",
            content=[
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        )
        return await self.generate([message], system=system, model=model_config.name)

    async def launch(self, manager: Launart) -> None:
        async with self.stage("preparing"):
            litellm.drop_params = True
            litellm.callbacks = [self.usage_handler]
            self.start_time = time.time()
            db_path = local_data.get_data_file("entari_plugin_llm", "agno.db")
            self._db = AsyncSqliteDb(db_file=str(db_path))
            self._sessions = AgnoSessionStore(self._db, self.id)

        async with self.stage("blocking"):
            await manager.status.wait_for_sigexit()

        async with self.stage("cleanup"):
            if self._db is not None:
                await self._db.db_engine.dispose()


llm = LLMService()
add_service(llm)
