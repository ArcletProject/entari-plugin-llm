# entari-plugin-llm

entari-plugin-llm 是一个用于 Entari 框架的 LLM（大语言模型）插件，提供对话、函数调用（Tool Call）、会话管理、视觉输入以及若干内置实用工具（如图像识别、网页处理等）。

插件默认使用 [Agno](https://github.com/agno-agi/agno) 的原厂模型适配器：OpenAI 兼容端点走 `OpenAIChat` / `OpenAILike`，Anthropic、Gemini、DashScope、Bedrock 等走各自的原生 SDK。LiteLLM 仅作为长尾 provider 的可选兜底，不再是核心依赖。

## 主要特性

- 基于 Agno 原生适配器的聊天能力（支持流式与非流式）。
- 支持“函数调用”机制（Tool Call），可以把插件内的订阅函数自动注册为 LLM 可调用的工具。
- 会话与上下文持久化（使用 Agno 的 SQLite 会话存储）。
- 支持视觉（image）输入的模型调用（可通过 `vision` 显式声明）。
- 支持 Agno 原生 Skills：按模型或全局启用，`auto` 按需加载，`always` 直接注入格式约束。
- 兼容旧版 LiteLLM 风格的 `provider/model` 模型名，升级后配置无需立即改写。
- 可配置的模型/提示/工具调用策略，通过 `Config` 加载与热重载（ConfigReload）。

## 要求

- Python >= 3.11
- 依赖见 `pyproject.toml` 中的 `dependencies`（推荐使用 pdm 或 uv 等现代包管理器）。
- 核心依赖已包含 `openai`，因此 OpenAI 以及所有 OpenAI 兼容端点开箱可用。
- 其他原厂 SDK 按需安装：

  ```bash
  pip install "entari-plugin-llm[anthropic]"   # Claude
  pip install "entari-plugin-llm[google]"      # Gemini / Vertex AI
  pip install "entari-plugin-llm[aws-bedrock]" # AWS Bedrock
  pip install "entari-plugin-llm[mistral]"     # Mistral
  pip install "entari-plugin-llm[cohere]"      # Cohere
  pip install "entari-plugin-llm[groq]"        # Groq
  pip install "entari-plugin-llm[ollama]"      # Ollama
  pip install "entari-plugin-llm[cerebras]"    # Cerebras
  ```

- 费用估算（可选）：

  ```bash
  pip install "entari-plugin-llm[pricing]"
  ```

  该可选依赖组包含 [`genai-prices`](https://github.com/pydantic/genai-prices)。插件会优先使用 provider 返回的 `RunOutput.metrics.cost`；只有 provider 没有返回费用时，才用 `genai-prices` 按 token 数估算。未安装时仍会正常统计 token，费用显示为“未统计”。

- 长尾 provider 可以继续使用 LiteLLM 兜底：

  ```bash
  pip install "entari-plugin-llm[litellm]"
  ```

  该可选依赖组包含 `litellm` 与 `mcp`。未安装时插件仍可正常加载；只有实际用到未适配的 provider 时才会提示安装。

## 快速开始

1. 克隆仓库：

  ```powershell
  git clone https://github.com/ArcletProject/entari-plugin-llm.git
  cd entari-plugin-llm
  ```

2. 安装依赖（使用 pdm，或使用 pip 在虚拟环境中安装）：

  ```powershell
  pdm sync
  # 或者使用 uv：
  uv sync
  ```

3. 运行本地示例（运行 Entari 应用）：

  ```powershell
  python main.py
  ```

说明：`main.py` 通过 `Entari.load("")` 加载当前目录下的 Entari 配置并启动服务 —— 在实际部署时请提供合适的配置文件与环境变量（例如模型的 API key、base_url 等）。

## 配置模型

每个模型都可以单独设置 `provider`、`api_key`、`base_url`、`vision` 与 `extra`；未设置时回退到全局 `api_key` / `base_url`。

```yaml
entari_plugin_llm:
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  models:
    - name: gpt-4o

    - name: deepseek-chat
      provider: deepseek
      api_key: ${DEEPSEEK_API_KEY}

    - name: kimi-k2
      provider: moonshot
      api_key: ${MOONSHOT_API_KEY}

    - name: glm-4.5
      provider: zhipu
      api_key: ${ZHIPU_API_KEY}

    - name: qwen-plus
      provider: dashscope
      api_key: ${DASHSCOPE_API_KEY}

    - name: Qwen/Qwen2.5-VL-72B-Instruct
      provider: siliconflow
      api_key: ${SILICONFLOW_API_KEY}
      vision: true

    - name: claude-sonnet-4
      provider: anthropic
      api_key: ${ANTHROPIC_API_KEY}

    - name: gemini-2.5-pro
      provider: google
      api_key: ${GOOGLE_API_KEY}

    - name: my-gateway-model
      provider: openai-compatible
      base_url: https://gateway.example.com/v1
      api_key: ${GATEWAY_API_KEY}

    - name: local-model
      provider: vllm
      base_url: http://localhost:8000/v1

    - name: llama3.2
      provider: ollama
      base_url: http://localhost:11434
```

### provider 解析顺序

1. 显式配置的 `provider`。
2. 旧版 LiteLLM 风格的 `provider/model` 前缀（如 `deepseek/deepseek-chat`、`anthropic/claude-sonnet-4`）。
3. `base_url` 的域名（如 `api.deepseek.com`、`api.siliconflow.cn`）。
4. 模型名前缀（如 `gpt-`、`claude-`、`gemini-`、`qwen`、`deepseek`、`kimi`）。
5. 未识别且配置了自定义 `base_url` 时，使用 `OpenAILike` 作为通用 OpenAI 兼容端点。
6. 其余情况回退到 LiteLLM。

常用 provider 名称：`openai`、`openai-compatible`、`anthropic`、`google`、`vertexai`、`dashscope`、`deepseek`、`siliconflow`、`moonshot`、`groq`、`openrouter`、`xai`、`together`、`fireworks`、`perplexity`、`sambanova`、`cerebras`、`nebius`、`mistral`、`cohere`、`ollama`、`vllm`、`lmstudio`、`zhipu`、`minimax`、`volcengine`、`hunyuan`、`stepfun`、`yi`、`baichuan`、`sensenova`、`qianfan`、`spark`、`aws-bedrock`、`litellm`。

### 旧配置迁移

旧的 LiteLLM 模型名可以直接保留，插件会在工厂中解析前缀并映射到原生适配器：

```yaml
models:
  - name: deepseek/deepseek-chat   # 等价于 provider: deepseek + name: deepseek-chat
  - name: anthropic/claude-sonnet-4
```

如果某个 provider 没有原生适配器，或原生 SDK 未安装但 LiteLLM 已安装，插件会自动降级到 LiteLLM。

### 视觉能力

Agno 没有统一的能力查询接口，因此插件使用以下规则：

- `vision: true` / `vision: false` 优先，适合显式覆盖。
- 未设置时根据 provider 与模型名推断；无法识别时按不支持处理。
- 常见视觉模型（`gpt-4o`、`claude-3/4`、`gemini`、`*-vl`、`glm-4v`、`kimi-latest` 等）会被自动识别。

### 技能（Skills）

插件使用 Agno 原生的 `Skills` / `LocalSkills`。技能是独立的 `SKILL.md` 目录，不需要把大段提示词复制进 `entari.yml`。

```yaml
entari_plugin_llm:
  # 全局默认技能，所有未单独配置 skills 的模型都会继承
  skills:
    - xml-output

  models:
    - name: gpt-4o
      # 覆盖全局技能；空列表 [] 表示该模型关闭所有技能
      skills:
        - name: xml-output
          mode: always
        - ./my-skills
```

支持两种模式：

- `auto`（默认）：Agno 只在系统提示中放入技能名称与描述，模型需要时通过 `get_skill_instructions` 工具加载完整指令；同时提供 reference 与 script 工具。
- `always`：插件把该技能的完整指令直接拼进系统提示，适合“每次回答都必须遵守”的输出格式约束。

内置技能：

- `xml-output`：要求模型返回单个、转义正确的 XML 文档，默认根节点为 `<result>`。

自定义技能可以指向包含 `SKILL.md` 的单个技能目录，或包含多个技能子目录的父目录。技能不需要额外依赖；未启用时不会向 Agent 注入任何 skill 工具或提示。

注意：`always` 只是把指令放进系统提示，仍然是提示词约束，不是 XML Schema 校验。如果 XML 需要被程序严格解析，建议继续使用 structured output 生成数据，再自行序列化为 XML，或对模型输出做校验与重试。

### 群聊会话

在群聊（`ChannelType.TEXT`）中，会话所有者是 `平台:频道ID` 而不是单个用户，因此同一频道的所有成员共享同一个会话、同一个模型设置和同一份历史。私聊仍然按用户隔离。

```yaml
entari_plugin_llm:
  # 作为上下文携带的最近群聊消息条数，0 关闭
  group_context_limit: 10
```

- 同一频道的消息按到达顺序串行处理（followup）：前一条消息跑完 Agent 后才开始下一条。
- 群聊上下文只保存在内存中，进程重启后清空；已经进入会话历史的消息不会重复注入。
- 群聊中的 `/llm session @某人` 会忽略被 @ 的用户，始终显示当前频道的会话。
- 受适配器限制：QQ 官方接口只推送被 @ 的消息，OneBot 等适配器才能拿到完整群聊记录。

### 扩展接口

其他插件可以订阅对话流程事件来追加提示词或改写回复：

```python
from arclet.entari import plugin
from entari_plugin_llm import LLMCollectVariableEvent, LLMResponseEvent

collect = plugin.dispatch(LLMCollectVariableEvent)
response = plugin.dispatch(LLMResponseEvent)


@collect.on()
async def add_instruction(event: LLMCollectVariableEvent):
    # 直接修改事件字段并返回 None 即可，返回 dict 仍会作为变量注入
    event.instructions.append("回答时优先使用短句。")


@response.on()
async def replace_reply(event: LLMResponseEvent):
    # 返回 MessageChain 替换回复，返回 False 阻止发送，返回 None 保持原样
    return event.message
```

- `LLMCollectVariableEvent` 提供 `session`、`llm_session`、`user_message`，其中 `instructions` 会拼接到本次系统提示词中。
- `LLMResponseEvent.message` 是即将发送的 `MessageChain`；该事件在普通对话与 `/llm ask` 两条路径上都会触发。
- 包顶层还导出了 `supports_vision`、`get_model_config`、`resolve_session_owner`、`SessionInfo`、`SessionLocks` 与 `owner_lock_key`，供外部插件判断视觉能力、读取模型配置或复用会话锁。

### 额外参数

`extra` 中的键如果与适配器的构造字段同名（如 `temperature`、`max_tokens`、`reasoning_effort`），会作为构造参数传入；其余键会合并进 `request_params`。运行时通过 `llm.generate(..., temperature=0.3)` 传入的参数优先级最高。

```yaml
models:
  - name: gpt-4o
    extra:
      temperature: 0.2
      max_tokens: 4096
      request_params:
        seed: 42
```

对于 GPT-5 / o 系列等不接受采样参数的模型，插件会自动忽略 `temperature`、`top_p`、`frequency_penalty`、`presence_penalty` 并输出一条警告。

## 基本用法（示例）

作为插件使用时，包会在导入时通过 `metadata()` 注册插件信息，并在运行时加载配置、工具与服务。你可以在代码中直接引用导出的服务：

```python
from entari_plugin_llm import llm

# 在异步上下文中调用
resp = await llm.generate("Hello world")
```

### 工具与函数调用（Tool）

- 插件将符合 arclet.letoderea 订阅器规范的函数自动注册为可被 LLM 调用的工具。
- 工具的参数与文档由函数的 docstring 与类型注解自动生成 JSON Schema，以便 LLM 在函数调用时进行参数填充。

### 配置与热重载

插件使用 `Config` 对象管理模型、提示词、上下文长度以及工具调用的最大循环步数。修改 Entari 插件配置并触发 `ConfigReload` 事件可以热重载这些设置。

### Token 与费用统计

插件直接读取 Agno 原生 `RunOutput.metrics`：

- `total_tokens`：全局与会话统计保持一致，不依赖 LiteLLM 的 callback。
- `cost`：优先使用 provider 返回的费用（例如 OpenRouter、部分 OpenAI 兼容端点）。如果 provider 没有返回，且安装了 `pricing` 可选依赖，则通过 `genai-prices` 根据模型名与 input/output/cache token 数估算。
- 未知模型、`genai-prices` 未安装或价格数据不匹配时，费用显示为“未统计”，不会影响正常对话。
- 费用以 USD 显示（provider 返回值或 `genai-prices` 估算）；`genai-prices` 使用的是其内置价格快照，实际账单请以 provider 为准。

## 项目结构（重要文件与目录）

- `src/entari_plugin_llm/` - 插件实现代码
  - `__init__.py` - 插件元信息与自动注册
  - `config.py` - 模型与插件配置
  - `model_factory.py` - provider → Agno 原生适配器工厂，含 LiteLLM 兜底
  - `capabilities.py` - 视觉能力推断
  - `pricing.py` - provider cost 读取与 `genai-prices` 可选费用估算
  - `skills.py` - Agno Skills 的解析、加载与 `auto` / `always` 模式
  - `_litellm.py` - LiteLLM 可选依赖的懒加载入口
  - `skills/` - 内置技能目录（如 `skills/xml-output/SKILL.md`）
  - `service.py` - LLM 服务实现（Agent 调用、工具调用处理、vision、token/费用统计等）
  - `handlers/` - Entari 事件处理器（chat、command、check 等）
  - `tools/` - 插件提供的工具注册逻辑与内置工具（如 image_vision、webpage_processor）
- `main.py` - 一个简单的启动示例
- `pyproject.toml` - 项目与依赖配置

## 开发与调试

在开发过程中你可以编辑 `entari.yml` 或插件配置，并使用 Entari 的配置热重载来应用更改。

运行单元测试：

```bash
python -m unittest discover -s tests -v
```

## 贡献

欢迎提交 issue 与 PR。请遵循仓库的编码风格与测试规范，尽量在 PR 中包含说明与复现步骤。

## 许可证

本项目遵循 MIT 许可证（见 `pyproject.toml` 中的 license 字段）。

## 联系方式

作者: RF-Tar-Railt, KomoriDev（详见 `src/entari_plugin_llm/__init__.py` 中的作者信息）

## 更多信息

请参阅 `pyproject.toml` 中的依赖与可选依赖（如浏览器、各 provider SDK、`genai-prices` 费用估算、LiteLLM 兜底等）以获取额外功能支持。
