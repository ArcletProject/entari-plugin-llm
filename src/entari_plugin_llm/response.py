import json
from collections.abc import AsyncIterator
from typing import Any, Generic, TypeVar, cast

from agno.run.agent import RunOutput

TOutput = TypeVar("TOutput")


class GenericResponse(Generic[TOutput]):

    def __init__(
        self,
        run_output: RunOutput | None = None,
        stream: AsyncIterator[Any] | None = None,
        structured: bool = False,
    ) -> None:
        self._run_output = run_output
        self._stream = stream
        self._structured = structured

    @property
    def content(self) -> str | TOutput | None:
        if self._run_output is not None:
            return self._run_output.content
        return None

    @property
    def output(self) -> TOutput | Any | None:
        if self._run_output is None or not self._structured:
            return None
        content = self._run_output.content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None
        return cast(TOutput, content)

    def __str__(self) -> str:
        content = self.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)

    def __repr__(self) -> str:
        return f"GenericResponse(content={self.content!r}, is_stream={self.is_stream})"

    @property
    def messages(self) -> list:
        if self._run_output is not None:
            return self._run_output.messages or []
        return []

    @property
    def metrics(self) -> Any:
        if self._run_output is not None:
            return self._run_output.metrics
        return None

    def stream(self) -> AsyncIterator[Any]:
        if self._stream is not None:
            return self._stream
        raise RuntimeError("stream() called on a non-streaming response")

    @property
    def is_stream(self) -> bool:
        return self._stream is not None

    @classmethod
    def from_run_output(
        cls,
        run_output: RunOutput,
        *,
        structured: bool = False,
    ) -> "GenericResponse[TOutput]":
        return cls(run_output=run_output, structured=structured)

    @classmethod
    def from_stream(
        cls,
        stream: AsyncIterator[Any],
    ) -> "GenericResponse[Any]":
        return cls(stream=stream)
