from collections.abc import Sequence

from entari_plugin_user import UserSession, get_user  # entari: plugin
from satori import At

from ._jsondata import get_default_model
from .config import _conf
from .manager import LLMSessionManager
from .sessions import SessionInfo


def _parse_session_id(choice: str, rows: Sequence[SessionInfo]) -> str | None:
    text = choice.strip()
    if not text:
        return None
    if text.isdigit():
        index = int(text)
        if 1 <= index <= len(rows):
            return rows[index - 1].session_id
        return None
    for row in rows:
        if row.session_id == text:
            return row.session_id
    return None


def render_session_list(rows: Sequence[SessionInfo]) -> str:
    lines = [f"会话列表（共 {len(rows)} 个）"]
    for idx, row in enumerate(rows, 1):
        flag = " [当前]" if row.is_active else ""
        lines.append(f"{idx}. {row.topic}{flag} | ID: {row.session_id}")
    return "\n".join(lines)


def render_model_list(current_model: str | None = None, channel: str = "$default") -> str:
    default_model = get_default_model(channel)
    lines = [f"模型列表（共 {len(_conf.models)} 个）"]
    for model in _conf.models:
        alias = f" ({model.alias})" if model.alias else ""
        line = f"- {model.name}{alias}"
        if current_model and current_model == model.name:
            line += " [当前]"
        elif default_model and default_model == model.name:
            line += " [默认]"
        lines.append(line)
    return "\n".join(lines)


async def resolve_user_id(session: UserSession, target: str | At | None = None) -> int:
    if target is None:
        return session.user_id
    platform_id = target.id if isinstance(target, At) else target
    if platform_id == session.platform_id:
        return session.user_id
    user = await get_user(session.platform, await session.user_get(platform_id))
    return user.id


async def select_session(session: UserSession) -> str | None:
    rows = await LLMSessionManager.list_sessions(session.user_id)
    if not rows:
        await session.send("暂无会话")
        return None

    prompt_text = f"{render_session_list(rows)}\n请输入会话序号或ID："
    response = await session.prompt(prompt_text)
    if response is None:
        await session.send("等待超时")
        return None

    selected = _parse_session_id(response.extract_plain_text(), rows)
    if selected is None:
        await session.send("输入无效，请输入会话序号或ID")
        return None
    return selected
