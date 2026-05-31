# entari-plugin-llm

entari-plugin-llm 是一个用于 Entari 框架的 LLM（大语言模型）插件，提供基于 litellm 的对话能力、函数调用（Tool Call）支持、会话管理以及若干内置实用工具（如图像识别、网页处理等）。

插件目标是为基于 Arclet Entari 的机器人/服务提供一个可配置、可拓展的 LLM 工具箱，便于在对话中调用函数以完成复杂任务，并支持结构化 JSON 输出、视觉识别等能力。

主要特性
- 基于 litellm 的聊天能力（支持流式与非流式）。
- 支持“函数调用”机制（Tool Call），可以把插件内的订阅函数自动注册为 LLM 可调用的工具。
- 会话与上下文持久化（使用 `entari-plugin-database` 提供的数据库模型）。
- 支持视觉（image）输入的模型调用（当模型支持 vision 时）。
- 可配置的模型/提示/工具调用策略，通过 `Config` 加载与热重载（ConfigReload）。

要求
- Python >= 3.10
- 依赖见 `pyproject.toml` 中的 `dependencies`（推荐使用 pdm 或 uv 等现代包管理器）。

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

## 项目结构（重要文件与目录）

- `src/entari_plugin_llm/` - 插件实现代码
  - `__init__.py` - 插件元信息与自动注册
  - `service.py` - LLM 服务实现（封装了 litellm 的调用、工具调用处理、vision 等）
  - `model.py` - 数据库 ORM 模型（会话与上下文）
  - `handlers/` - Entari 事件处理器（chat、command、check 等）
  - `tools/` - 插件提供的工具注册逻辑与内置工具（如 image_vision、webpage_processor）
- `main.py` - 一个简单的启动示例
- `pyproject.toml` - 项目与依赖配置

## 开发与调试

在开发过程中你可以编辑 `entari.yml` 或插件配置，并使用 Entari 的配置热重载来应用更改。

## 贡献

欢迎提交 issue 与 PR。请遵循仓库的编码风格与测试规范，尽量在 PR 中包含说明与复现步骤。

## 许可证

本项目遵循 MIT 许可证（见 `pyproject.toml` 中的 license 字段）。

## 联系方式

作者: RF-Tar-Railt, KomoriDev（详见 `src/entari_plugin_llm/__init__.py` 中的作者信息）

## 更多信息

请参阅 `pyproject.toml` 中的依赖与可选依赖（如浏览器、Google provider 等）以获取额外功能支持。

