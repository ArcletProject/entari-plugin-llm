from arclet.alconna import Alconna, Args, MultiVar, Option, Subcommand, store_true
from arclet.entari import MessageChain, Session, command, metadata
from arclet.entari.const import ITEM_MESSAGE_REPLY
from arclet.letoderea import BLOCK, Contexts

from .._jsondata import set_default_model
from ..config import get_model_config, get_model_list
from ..exception import ModelNotFoundError
from ..manager import LLMSessionManager
from ..utils import render_model_list, render_session_list, select_session


metadata(
    name="LLM 指令",
    author=[
        {"name": "RF-Tar-Railt", "email": "rf_tar_railt@qq.com"},
        {"name": "KomoriDev", "email": "mute231010@gmail.com"},
    ],
    version="0.1.0",
    description="LLM 工具箱插件的指令模块",
)


llm_alc = Alconna(
    "llm",
    Args["content?#内容", MultiVar(str)],
    Option("-m|--model", Args["model?#模型名称", str], dest="select_model", help_text="指定模型"),
    Option(
        "-n|--new",
        dest="new_opt",
        default=False,
        action=store_true,
        help_text="创建新会话",
    ),
    Subcommand("new", dest="new_cmd", help_text="创建新会话"),
    Subcommand("switch", Args["session_id?#会话ID", str], help_text="切换会话"),
    Subcommand("delete", Args["session_id?#会话ID", str], help_text="删除会话"),
    Subcommand(
        "session",
        Option("-l|--list", help_text="查看会话列表"),
        help_text="查看当前会话信息",
    ),
    Subcommand(
        "model",
        Args["model?#模型名称", str],
        Option("-l|--list", help_text="查看模型列表"),
        help_text="查看当前模型信息, 或者切换会话使用的模型",
    )
)

llm_alc.shortcut("ai", {"command": "llm", "fuzzy": True, "prefix": True})

llm_disp = command.mount(llm_alc, skip_for_unmatch=False)


@llm_disp.assign("new_cmd")
async def _(session: Session, model: command.Query[str] = command.Query("select_model.model")):
    conf = get_model_config(model.result if model.available else None, session.channel.id)
    if model.available:
        set_default_model(conf.name, session.channel.id)
    new_session = await LLMSessionManager.create_new_session(
        f"{session.account.platform}:{session.user.id}",
        model=conf.name
    )
    await session.send(f"以创建并切换到新会话\n会话ID: {new_session.session_id}")
    return BLOCK


@llm_disp.assign("switch")
async def _(session: Session, session_id: command.Match[str]):
    if not session_id.available:
        selected = await select_session(session)
        if selected is None:
            return BLOCK

        session_id.result = selected

    switched = await LLMSessionManager.switch(f"{session.account.platform}:{session.user.id}", session_id.result)
    await session.send("切换成功" if switched else "未找到对应会话")
    return BLOCK


@llm_disp.assign("delete")
async def _(session: Session, session_id: command.Match[str]):
    if not session_id.available:
        selected = await select_session(session)
        if selected is None:
            return BLOCK

        session_id.result = selected
    info = await LLMSessionManager.get_current_session_info(f"{session.account.platform}:{session.user.id}")
    deleted = await LLMSessionManager.delete(f"{session.account.platform}:{session.user.id}", session_id.result)
    if deleted:
        rows = await LLMSessionManager.list_sessions(f"{session.account.platform}:{session.user.id}")
        if not rows:
            await LLMSessionManager.create_new_session(f"{session.account.platform}:{session.user.id}")
            await session.send("删除成功，已自动创建新会话")
        elif info and info["session_id"] == session_id.result:
            switched = await LLMSessionManager.switch(
                f"{session.account.platform}:{session.user.id}", rows[0].session_id
            )
            await session.send("删除成功，已切换到最近的会话" if switched else "删除成功，但未找到对应会话")
        else:
            await session.send("删除成功，当前会话列表：\n" + render_session_list(rows))
    else:
        await session.send("未找到对应会话")
    return BLOCK


@llm_disp.assign("session.list")
async def _(session: Session):
    rows = await LLMSessionManager.list_sessions(f"{session.account.platform}:{session.user.id}")

    if not rows:
        await session.send("暂无会话")
        return BLOCK

    await session.send(render_session_list(rows))
    return BLOCK


@llm_disp.assign("session", priority=20)
async def _(session: Session):
    info = await LLMSessionManager.get_current_session_info(f"{session.account.platform}:{session.user.id}")
    if info is None:
        await session.send("当前没有活动会话")
        return BLOCK

    created_at = info["created_at"].strftime("%Y-%m-%d %H:%M:%S")
    await session.send(
        "\n".join(
            [
                f"会话ID: {info['session_id']}",
                f"话题: {info['topic']}",
                f"使用模型: {info['model']}",
                f"消息数: {info['message_count']}",
                f"累计 Token: {info['total_tokens']}",
                f"创建时间: {created_at}",
            ]
        )
    )
    return BLOCK


@llm_disp.assign("model.list")
async def _(session: Session):
    info = await LLMSessionManager.get_current_session_info(f"{session.account.platform}:{session.user.id}")
    current_model = info["model"] if info else None
    await session.send(render_model_list(current_model, session.channel.id))
    return BLOCK


@llm_disp.assign("model", priority=20)
async def _(session: Session, model: command.Match[str]):
    if model.available:
        if model.result not in get_model_list():
            info = await LLMSessionManager.get_current_session_info(f"{session.account.platform}:{session.user.id}")
            current_model = info["model"] if info else None
            await session.send(render_model_list(current_model, session.channel.id))
            return BLOCK

        conf = get_model_config(model.result, session.channel.id)
        set_default_model(conf.name, session.channel.id)

        if await LLMSessionManager.select_model(f"{session.account.platform}:{session.user.id}", conf.name):
            await session.send(f"已切换当前会话使用的模型为: {model.result}")
        else:
            await session.send(f"已切换默认模型为: {model.result}")
        return BLOCK
    info = await LLMSessionManager.get_current_session_info(f"{session.account.platform}:{session.user.id}")
    current_model = info["model"] if info else None
    await session.send(render_model_list(current_model, session.channel.id))
    return BLOCK


@llm_disp.handle(priority=25)
async def main_chat(
    ctx: Contexts,
    session: Session,
    content: command.Match[MessageChain],
    new_opt: command.Query[bool] = command.Query("new_opt.value"),
    model: command.Query[str] = command.Query("select_model.model"),
):
    user_prompt = MessageChain([])

    if reply := ctx.get(ITEM_MESSAGE_REPLY):
        user_prompt += reply.origin.message

    if content.available:
        user_prompt += content.result

    if not user_prompt:
        resp = await session.prompt("需要我为你做些什么？")
        if not resp:
            await session.send("等待超时")
            return BLOCK
        user_prompt = resp

    try:
        conf = get_model_config(model.result if model.available else None, session.channel.id)
        if model.available:
            set_default_model(conf.name, session.channel.id)
        answer = await LLMSessionManager.chat(
            user_prompt,
            session=session,
            ctx=ctx,
            model=conf.name,
            new=new_opt.result,
        )
        if answer != "[END_OF_RESPONSE]":
            await session.send(answer)
    except ModelNotFoundError as e:
        await session.send(MessageChain(str(e)))
    except Exception as e:
        await session.send(MessageChain(str(e)))

    return BLOCK
