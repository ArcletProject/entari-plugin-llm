from typing import Any, Literal

from arclet.entari import BasicConfModel, Session, plugin_config
from arclet.entari.config import model_field

from ._jsondata import get_default_model
from .exception import ModelNotFoundError
from .log import log


class SkillConfig(BasicConfModel):
    name: str
    """技能名称，或包含 SKILL.md 的目录路径"""
    mode: Literal["auto", "always"] = "auto"
    """auto: 由模型按需加载；always: 直接注入系统提示，适合输出格式约束"""


SkillRef = str | SkillConfig


def _coerce_skill_refs(value: list[SkillRef] | None) -> list[SkillRef] | None:
    if value is None:
        return None

    result: list[SkillRef] = []
    for item in value:
        if isinstance(item, SkillConfig):
            skill = item
        elif isinstance(item, str):
            result.append(item)
            continue
        elif isinstance(item, dict):
            skill = SkillConfig(**item)
        else:
            raise TypeError("skills entries must be a name, path, or {name, mode} object")

        if skill.mode not in {"auto", "always"}:
            raise ValueError(f"Skill {skill.name!r} has invalid mode {skill.mode!r}")
        result.append(skill)
    return result


class ScopedModel(BasicConfModel):
    name: str
    """模型名称（或 `provider/model` 形式的旧版 LiteLLM 写法）"""
    alias: str | None = None
    """模型的别名"""
    provider: str | None = None
    """模型提供方，如 openai、anthropic、google、deepseek、dashscope、bedrock 等。
    未设置时根据 base_url 与模型名自动推断"""
    vision: bool | None = None
    """该模型是否支持图片输入。未设置时根据提供方与模型名推断"""
    api_key: str | None = None
    """用于使用 OpenAI API 进行身份验证的 API 密钥。如果未设置，则回退到全局 api_key"""
    base_url: str = "https://api.openai.com/v1"
    """OpenAI API 的接口地址。如果未设置，则回退到全局 base_url"""
    prompt: str = ""
    """该模型使用的提示词。如果未设置，则回退到全局 prompt"""
    skills: list[SkillRef] | None = None
    """该模型启用的技能。None 表示继承全局配置，空列表表示显式关闭"""
    extra: dict[str, Any] = model_field(default_factory=dict)
    """传递给 LLM API 调用的额外参数"""

    def __post_init__(self):
        self.skills = _coerce_skill_refs(self.skills)


class Chance(BasicConfModel):
    value: float
    """概率值，范围为 0 到 1"""
    platform: str
    """适用的平台，如 "qq"、"wechat"、"discord" 等。"""
    channel: list[str]
    """适用的频道列表。['*'] 表示所有频道。"""


class Config(BasicConfModel, extra="allow"):
    api_key: str | None = None
    """用于使用 OpenAI API 进行身份验证的全局 API 密钥。用作没有特定键的模型的后备"""
    base_url: str = "https://api.openai.com/v1"
    """OpenAI API 的全局接口地址。用作没有特定接口地址的模型的后备"""
    prompt: str = ""
    """全局提示词。用作没有特定提示词的模型的后备"""
    skills: list[SkillRef] = model_field(default_factory=list)
    """全局启用的技能。模型未单独配置 skills 时继承此列表"""
    models: list[ScopedModel] = model_field(default_factory=list)
    """配置模型及其各自设置的列表"""
    tools: dict[str, dict[str, Any]] = model_field(default_factory=dict)
    """工具"""
    history_limit: int | None = None
    """限制每个用户的历史会话数量。如果未设置，则没有限制"""
    group_context_limit: int = 10
    """群聊中最多携带多少条最近消息作为上下文。群聊按频道共享会话，0 表示关闭"""
    max_tool_calls_from_history: int | None = None
    """限制历史上下文中的工具调用数量。如果未设置，则没有限制"""
    chance_of_full_chat: float | list[Chance] = 0.5
    """在群聊中，触发完整对话的概率。0 表示关闭，1 表示总是触发完整对话"""

    def _reload_tools(self):
        loaded_tools: dict[str, dict[str, Any]] = {}

        for key, value in self.tools.items():
            if key.startswith("$"):
                loaded_tools[key] = value
                continue

            tool_config = dict(value)
            new_key = key

            if key.startswith("~"):
                new_key = key[1:]
                if "$disable" not in tool_config or isinstance(tool_config["$disable"], bool):
                    tool_config["$disable"] = True
            elif key.startswith("?"):
                new_key = key[1:]
                tool_config["$optional"] = True

            if new_key.startswith("::agno."):
                new_key = new_key[2:]
            elif new_key.startswith("::"):
                new_key = new_key.replace("::", "entari_plugin_llm.builtins.tools.")
            if tool_config.get("$disable"):
                continue

            call_limit = tool_config.get("$call_limit")
            if call_limit is not None and (type(call_limit) is not int or call_limit < 0):
                raise ValueError(f"{new_key}: $call_limit must be a positive integer")

            loaded_tools[new_key] = tool_config

        self.tools = loaded_tools

    def __post_init__(self):
        if self.group_context_limit < 0:
            raise ValueError("group_context_limit must be a non-negative integer")
        self.skills = _coerce_skill_refs(self.skills) or []
        self._reload_tools()

    def get_chance_of(self, session: Session):
        if isinstance(self.chance_of_full_chat, (float, int)):
            return self.chance_of_full_chat
        if not session.event.channel:
            return 0.0
        chance = next((c for c in self.chance_of_full_chat if c.platform == session.account.platform), None)
        if not chance:
            return 0.0
        if "*" in chance.channel or session.event.channel.id in chance.channel:
            return chance.value
        return 0.0


_conf = plugin_config(Config)


def get_model_config(model_name: str | None = None, channel: str = "$default") -> ScopedModel:
    if model_name is None:
        if not _conf.models:
            raise ModelNotFoundError("No models configured.")

        model_name = get_default_model(channel)
        model = next((m for m in _conf.models if m.name == model_name or m.alias == model_name), None)
    else:
        model = next((m for m in _conf.models if m.name == model_name or m.alias == model_name), None)
        if not model:
            log("warning", f"Model {model_name} not found in config. Using default model instead.")
            model_name = get_default_model(channel)
            model = next((m for m in _conf.models if m.name == model_name or m.alias == model_name), None)

    if model:
        model_cp = ScopedModel(
            name=model.name,
            alias=model.alias,
            provider=model.provider,
            vision=model.vision,
            api_key=model.api_key,
            base_url=model.base_url,
            prompt=model.prompt,
            skills=model.skills,
            extra=model.extra,
        )
        if not model.api_key and _conf.api_key:
            model_cp.api_key = _conf.api_key
        if model.base_url == "https://api.openai.com/v1" and _conf.base_url != "https://api.openai.com/v1":
            model_cp.base_url = _conf.base_url
        if not model.prompt and _conf.prompt:
            model_cp.prompt = _conf.prompt
        return model_cp
    raise ModelNotFoundError(f"Model {model_name} not found in config.")


def get_model_list() -> set[str]:
    return {m.name for m in _conf.models} | {m.alias for m in _conf.models if m.alias}
