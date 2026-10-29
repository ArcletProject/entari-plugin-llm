import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Any, Literal, TypeVar, cast, overload

from agno.agent import Agent
from agno.db.sqlite import AsyncSqliteDb
from agno.models.message import Message
from agno.run.agent import RunErrorEvent, RunOutput, RunOutputEvent
from agno.run.base import RunStatus
from arclet.entari import Session, add_service, local_data
from arclet.letoderea import Contexts
from launart import Launart, Service
from launart.status import Phase

from .capabilities import supports_vision
from .config import ScopedModel, _conf, get_model_config
from .history import strip_history_media
from .image_memory import remember_image
from .memory import build_memory_manager
from .model_factory import create_model
from .pricing import resolve_metrics_cost
from .response import GenericResponse
from .runtime import ToolExecutors, run_llm_tools
from .sessions import AgnoSessionStore, SessionInfo
from .skills import apply_skills, resolve_selections
from .tools import get_agno_tools

TOutput = TypeVar("TOutput")
OutputType = Literal["json_object"] | type[Any] | dict[str, Any]


class LLMService(Service):
    id = "entari_plugin_llm"

    def __init__(self) -> None:
        super().__init__()
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_calls = 0
        self.start_time: float = 0.0
        self._run_token_totals: dict[str, int] = {}
        self._run_cost_totals: dict[str, float] = {}
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
        ignore_user_prompt: bool = False,
        use_tools: bool = True,
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
        ignore_user_prompt: bool = False,
        use_tools: bool = True,
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
        ignore_user_prompt: bool = False,
        use_tools: bool = True,
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
        ignore_user_prompt: bool = False,
        use_tools: bool = True,
        **kwargs: Any,
    ) -> GenericResponse[Any]:
        """Generate a stateless LLM response through an Agno Agent."""
        if "session_id" in kwargs or "user_id" in kwargs:
            raise TypeError("session_id and user_id are reserved for Entari chat persistence")
        return await self._run_agent(
            message,
            variables,
            stream=stream,
            system=system,
            model=model,
            output=output,
            llm_session=None,
            ignore_user_prompt=ignore_user_prompt,
            use_tools=use_tools,
            request_params=kwargs,
        )

    async def _generate_for_session(
        self,
        message: str | list[Message],
        variables: dict[str, Any] | None = None,
        *,
        llm_session: SessionInfo,
        model: str | None = None,
        ctx: Contexts | None = None,
        session: Session | None = None,
        extra_instructions: list[str] | None = None,
        allow_silent: bool = False,
    ) -> GenericResponse[None]:
        """Generate a persisted response for Entari's chat session manager."""
        return await self._run_agent(
            message,
            variables,
            stream=False,
            system=None,
            model=model,
            output=None,
            llm_session=llm_session,
            ignore_user_prompt=False,
            request_params={},
            ctx=ctx,
            session=session,
            extra_instructions=extra_instructions,
            allow_silent=allow_silent,
        )

    async def _run_agent(
        self,
        message: str | list[Message],
        variables: dict[str, Any] | None,
        *,
        stream: bool,
        system: str | None,
        model: str | None,
        output: OutputType | None,
        llm_session: SessionInfo | None,
        ignore_user_prompt: bool,
        request_params: dict[str, Any],
        ctx: Contexts | None = None,
        session: Session | None = None,
        extra_instructions: list[str] | None = None,
        allow_silent: bool = False,
        use_tools: bool = True,
    ) -> GenericResponse[Any]:
        if output is not None and stream:
            raise ValueError("output is not supported when stream=True")

        model_config = get_model_config(model)
        if use_tools:
            tools, executors = get_agno_tools(exclude=None if allow_silent else {"no_reply"})
        else:
            tools, executors = [], {}

        if system is not None:
            instructions = system
        elif ignore_user_prompt:
            instructions = None
        else:
            instructions = model_config.prompt or _conf.prompt or None

        if extra_instructions:
            extra = "\n\n".join(item for item in extra_instructions if item)
            if extra:
                instructions = f"{instructions}\n\n{extra}" if instructions else extra

        if variables:
            variable_instructions = "下列是用以辅助你思考回答的变量：\n" + "\n".join(
                f"- **{key}**: {value!r}" for key, value in variables.items()
            )
            instructions = f"{instructions}\n\n{variable_instructions}" if instructions else variable_instructions

        instructions, agent_skills = apply_skills(instructions, resolve_selections(model_config, _conf.skills))

        model_request_params = request_params or None
        agno_model = create_model(model_config, request_params=model_request_params)

        agent_kwargs: dict[str, Any] = {
            "id": self.id,
            "model": agno_model,
            "instructions": instructions,
            "tools": tools,
            "markdown": False,
        }
        if agent_skills is not None:
            agent_kwargs["skills"] = agent_skills
        if output is not None:
            agent_kwargs["output_schema"] = output if output != "json_object" else {"type": "object"}
            agent_kwargs["use_json_mode"] = True
        if llm_session is not None:
            if self._db is not None:
                agent_kwargs["db"] = self._db
            agent_kwargs["session_id"] = llm_session.session_id
            agent_kwargs["user_id"] = str(llm_session.user_id)
            agent_kwargs["add_history_to_context"] = True
            agent_kwargs["enable_session_summaries"] = True
            agent_kwargs["add_session_summary_to_context"] = False
            if _conf.history_limit is not None:
                agent_kwargs["num_history_messages"] = _conf.history_limit
            if _conf.max_tool_calls_from_history is not None:
                agent_kwargs["max_tool_calls_from_history"] = _conf.max_tool_calls_from_history

            memory_manager = build_memory_manager(self._db, _conf.memory)
            if memory_manager is not None:
                agent_kwargs["memory_manager"] = memory_manager
                if _conf.memory.mode == "agentic":
                    agent_kwargs["enable_agentic_memory"] = True
                else:
                    agent_kwargs["update_memory_on_run"] = True

        if _conf.strip_history_media:
            agent_kwargs["store_media"] = False
            agent_kwargs["pre_hooks"] = [strip_history_media]

        agent = Agent(**agent_kwargs)
        if stream:
            return GenericResponse.from_stream(
                self._stream_agent(agent, message, executors, llm_session, session, ctx, model_config)
            )

        response = await agent.arun(message)
        while await self._should_continue(response, executors, llm_session, session, ctx, model_config):
            response = await agent.acontinue_run(run_response=response, requirements=response.requirements)
        return GenericResponse.from_run_output(
            response, structured=output is not None, stopped=response.is_paused or response.is_cancelled
        )

    async def _should_continue(
        self,
        response: RunOutput,
        executors: ToolExecutors,
        llm_session: SessionInfo | None,
        session: Session | None,
        ctx: Contexts | None,
        model_config: ScopedModel,
    ) -> bool:
        if response.status is RunStatus.error:
            raise RuntimeError(str(response.content or "Agno agent run failed"))
        self._track_usage(response, model_config)
        if not response.is_paused:
            return False
        return await run_llm_tools(response, executors, llm_session, session, ctx)

    def _track_usage(self, response: RunOutput, model_config: ScopedModel) -> None:
        """Accumulate tokens and cost from Agno's native ``RunOutput.metrics``.

        Agno keeps metrics cumulative across continuation segments of the same
        run, so only the delta relative to the previous segment is counted.
        """

        metrics = getattr(response, "metrics", None)
        if metrics is None:
            return

        run_id = getattr(response, "run_id", None) or f"anonymous-{id(response)}"
        total = metrics.get("total_tokens") if isinstance(metrics, dict) else getattr(metrics, "total_tokens", None)
        total_tokens = None
        if total is not None:
            try:
                total_tokens = int(total)
            except (TypeError, ValueError):
                total_tokens = None

        if total_tokens is not None:
            previous = self._run_token_totals.get(run_id, 0)
            if total_tokens > previous:
                self.total_tokens += total_tokens - previous
                self._run_token_totals[run_id] = total_tokens

        cost = resolve_metrics_cost(metrics, model_config)
        if cost is not None:
            previous_cost = self._run_cost_totals.get(run_id, 0.0)
            if cost > previous_cost:
                self.total_cost += cost - previous_cost
            self._run_cost_totals[run_id] = max(previous_cost, cost)

        self.total_calls += 1
        if not getattr(response, "is_paused", False):
            self._run_token_totals.pop(run_id, None)
            self._run_cost_totals.pop(run_id, None)

    async def _stream_agent(
        self,
        agent: Agent,
        message: str | list[Message],
        executors: ToolExecutors,
        llm_session: SessionInfo | None,
        session: Session | None,
        ctx: Contexts | None,
        model_config: ScopedModel,
    ) -> AsyncIterator[RunOutputEvent | RunOutput]:
        stream = cast(
            AsyncGenerator[RunOutputEvent | RunOutput, None],
            agent.arun(message, stream=True, yield_run_output=True),
        )
        while True:
            response: RunOutput | None = None
            error: RunErrorEvent | None = None
            async with aclosing(stream):
                async for event in stream:
                    if isinstance(event, RunOutput):
                        response = event
                    else:
                        if isinstance(event, RunErrorEvent):
                            error = event
                        yield event

            if error is not None:
                raise RuntimeError(str(error.content or "Agno agent run failed"))
            if response is None:
                raise RuntimeError("Agno stream ended without a run output")
            if not await self._should_continue(response, executors, llm_session, session, ctx, model_config):
                yield response
                return

            stream = cast(
                AsyncGenerator[RunOutputEvent | RunOutput, None],
                agent.acontinue_run(
                    run_response=response, requirements=response.requirements, stream=True, yield_run_output=True
                ),
            )

    async def vision(
        self,
        image_url: str | dict[str, Any],
        system: str | None = None,
        model: str | None = None,
    ) -> GenericResponse[None]:
        """Describe an image through the standard generate path."""
        model_config = get_model_config(model)
        if not supports_vision(model_config):
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
        response = await self.generate([message], system=system, model=model_config.name, use_tools=False)
        if isinstance(response.content, str) and response.content:
            remember_image(url, response.content)
        return response

    async def launch(self, manager: Launart) -> None:
        async with self.stage("preparing"):
            self.start_time = time.time()
            db_path = local_data.get_data_file("entari_plugin_llm", "agno.db")
            self._db = AsyncSqliteDb(db_file=str(db_path))
            # Agno 只在 OS app 中调用 _create_all_tables()；普通会话/run 表因为
            # 各自传了 create_table_if_not_found=True 会自动创建，但 memories
            # 表不会。缺失时 upsert_user_memory() 会静默丢弃写入，所以这里
            # 主动补齐（幂等）。
            await self._db._create_all_tables()
            self._sessions = AgnoSessionStore(self._db, self.id)

        async with self.stage("blocking"):
            await manager.status.wait_for_sigexit()

        async with self.stage("cleanup"):
            if self._db is not None:
                await self._db.db_engine.dispose()


llm = LLMService()
add_service(llm)
