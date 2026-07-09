import json
from datetime import datetime
from typing import Literal, TypeAlias, cast

from entari_plugin_database import Base
from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TEXT, TypeDecorator

from ._types import Message

ROLE: TypeAlias = Literal["user", "assistant", "tool"]


class JSONText(TypeDecorator):
    impl = TEXT
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return json.loads(value)


class LLMSession(Base):
    __tablename__ = "entari_plugin_llm_session"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    topic: Mapped[str] = mapped_column(String(24))
    model: Mapped[str] = mapped_column(String(64))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)

    total_tokens: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    contexts = relationship("SessionContext", back_populates="session", cascade="all, delete-orphan")


class SessionContext(Base):
    __tablename__ = "entari_plugin_llm_context"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("entari_plugin_llm_session.session_id"), index=True)

    role: Mapped[ROLE] = mapped_column(String(16))
    content: Mapped[list] = mapped_column(JSONText, nullable=True)
    reasoning_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    name: Mapped[str | None] = mapped_column(String(64), nullable=True)

    tool_calls: Mapped[list | None] = mapped_column(JSONText, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    session = relationship("LLMSession", back_populates="contexts")

    @property
    def message(self) -> Message:
        msg: dict = {"role": self.role, "content": self.content}

        if self.role == "user":
            if self.name:
                msg["name"] = self.name

        elif self.role == "assistant":
            if self.reasoning_content:
                msg["reasoning_content"] = self.reasoning_content
            if self.tool_calls:
                msg["tool_calls"] = self.tool_calls
            msg["content"] = self.content

        elif self.role == "tool" and self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id

        elif self.role == "system":
            if self.name:
                msg["name"] = self.name

        return cast(Message, msg)
