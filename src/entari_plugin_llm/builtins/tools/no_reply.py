from arclet.letoderea import ExitState

from entari_plugin_llm import register_tool


@register_tool(stop_after_tool_call=True)
async def no_reply() -> ExitState:
    """
    结束本轮对话，不发送任何回复。

    当前消息没有明确叫到你，且你判断没有必要回应时使用。
    一旦调用，本轮不会向用户发送任何内容。明确 @ 你或回复你时不要调用。
    """
    return ExitState.stop
