from arclet.entari import Image, MessageChain, Session

from entari_plugin_llm import llm, register_tool


@register_tool
async def image_vision(session: Session, index: int = 0) -> str | None:
    """
    从会话中识别图片中的内容并返回描述
    如果有多张图片，请根据索引多次调用此工具（例如 index=0, index=1）。

    Args:
        session (Session): 当前会话对象
        index (int): 图片索引，默认为0，表示第一张图片
    Returns:
        图片内容的描述。为空表示工具执行已暂停
    """
    img_chain = MessageChain[Image]()

    if reply := session.reply:
        img_chain.extend(MessageChain(reply.origin.message).select(Image))

    if session._content and session.elements.has(Image):
        img_chain.extend(session.elements.get(Image))

    img_urls = img_chain.map(lambda x: x.src)

    if not img_urls:
        return "未找到图片"

    if index >= len(img_urls):
        return f"索引 {index} 超出图片数量 {len(img_urls)} 的范围"

    try:
        resp = await llm.vision(img_urls[index])
    except RuntimeError:
        return "无法识别图片内容"
    if resp.stopped:
        return
    return resp.content or "无法识别图片内容"
