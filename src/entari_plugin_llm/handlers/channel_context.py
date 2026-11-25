from arclet.entari import ChannelType, MessageCreatedEvent
from arclet.letoderea import on
from entari_plugin_user import UserSession  # entari: plugin

from ..config import _conf
from ..context import group_context


@on(MessageCreatedEvent, priority=2, label="群聊上下文缓存")
async def buffer_group_message(session: UserSession, event: MessageCreatedEvent):
    """记录群聊消息，供后续对话作为上下文使用"""
    if _conf.group_context_limit <= 0 or event.channel.type != ChannelType.TEXT:
        return

    sender_name = (
        (event.member.nick if event.member else None)
        or session.user_name
        or (event.user.nick or event.user.name if event.user else None)
    )
    sender = str(session.user_id) if not sender_name else f"{sender_name}({session.user_id})"
    if event.user.id == session.session.account.self_id:
        sender = f"我({session.user_id})"
    group_context.record(
        f"{session.platform}:{event.channel.id}",
        message_id=event.message.id,
        sender=sender,
        text=session.session.content,
    )
