import sys
from pathlib import Path

from arclet.entari import Session, local_data, plugin
from arclet.entari.config import BasicConfModel

from entari_plugin_llm import register_tool
from entari_plugin_llm.builtins.tools.generic import detect_superuser


class Config(BasicConfModel):
    plugin_dir: str | None = None
    """插件存放目录，默认为 None，表示使用默认的缓存目录"""


_conf = plugin.plugin_config(Config)
if _conf.plugin_dir:
    _dir = Path.cwd().joinpath(_conf.plugin_dir).absolute()
    _dir.mkdir(parents=True, exist_ok=True)
else:
    _dir = local_data.get_cache_dir("llm")


@register_tool
async def list_plugins() -> dict[str, dict[str, str]]:
    """
    列出当前已加载的插件名称列表。

    元信息包括插件名称 (name)，插件定位 (role) 和插件描述 (description)。

    Returns:
        已加载的插件 ID 与其部分元信息的映射
    """

    return {
        p.id: {
            "name": p.metadata.name,
            "role": p.metadata.role.value,
            "description": p.metadata.description or "No description available",
        }
        if p.metadata
        else {"name": p.id, "role": "normal", "description": "No description available"}
        for p in plugin.get_plugins()
    }


@register_tool
async def load_plugin(session: Session, import_path: str, config: dict | None = None) -> str:
    """
    加载新的插件。

    **该工具需要先检测超级用户权限。**

    Args:
        session (Session): 当前会话对象
        import_path: 插件的导入路径，例如 "entari_plugin_example"
        config: 可选的插件配置字典，如果插件需要配置，可以传入相应的配置

    Returns:
        str: 加载结果信息
    """
    if not await detect_superuser(session):
        return "当前用户没有权限使用此工具"
    sys.path.insert(0, str(_dir))
    try:
        result = plugin.load_plugin(import_path, config)
        if result.error:
            return f"加载插件 {import_path} 失败: {result.error!r}"
        return f"插件 {import_path} 已成功加载"
    except Exception as e:
        return f"加载插件 {import_path} 失败: {e!r}"
    finally:
        sys.path.remove(str(_dir))


@register_tool
async def unload_plugin(session: Session, plugin_id: str) -> str:
    """
    卸载已加载的插件。

    **该工具需要先检测超级用户权限。**

    Args:
        session (Session): 当前会话对象
        plugin_id: 已加载插件的 ID (插件的导入路径)

    Returns:
        str: 卸载结果信息
    """
    if not await detect_superuser(session):
        return "当前用户没有权限使用此工具"
    try:
        result = await plugin.unload_plugin_async(plugin_id)
        if not result:
            return f"未能卸载插件 {plugin_id}，请检查插件 ID 是否正确"
        return f"插件 {plugin_id} 已成功卸载"
    except Exception as e:
        return f"卸载插件 {plugin_id} 失败: {e!r}"


@register_tool
async def reload_plugin(session: Session, plugin_id: str) -> str:
    """
    重新加载已加载的插件。

    **该工具需要先检测超级用户权限。**

    Args:
        session (Session): 当前会话对象
        plugin_id: 已加载插件的 ID (插件的导入路径)

    Returns:
        str: 重新加载结果信息
    """
    if not await detect_superuser(session):
        return "当前用户没有权限使用此工具"
    sys.path.insert(0, str(_dir))
    try:
        result = await plugin.reload_plugin(plugin_id)
        if not result:
            return f"未能重新加载插件 {plugin_id}，请检查插件 ID 是否正确"
        return f"插件 {plugin_id} 已成功重新加载"
    except Exception as e:
        return f"重新加载插件 {plugin_id} 失败: {e!r}"
    finally:
        sys.path.remove(str(_dir))


@register_tool
async def write_plugin(session: Session, plugin_name: str, content: str, config: dict | None = None) -> str:
    """
    写一个新的（或更新）单文件插件并加载。注意该工具会限制只在特定目录下写入文件，以防止写入敏感目录。

    **该工具需要先检测超级用户权限。**

    Args:
        session (Session): 当前会话对象
        plugin_name (str): 插件名称
        content (str): 要写入文件的内容
        config (dict | None): 可选的插件配置字典，如果插件需要配置，可以传入相应的配置

    Returns:
        str: 写入结果信息，如果写入失败，则返回错误信息
    """
    if not await detect_superuser(session):
        return "当前用户没有权限使用此工具"
    sys.path.insert(0, str(_dir))
    path = _dir / f"{plugin_name}.py"
    try:
        exists = path.exists()
        try:
            with open(path, "w+", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            return f"写入插件 {path} 时发生错误: {e!r}"
        if exists:
            try:
                result = await plugin.reload_plugin(plugin_name)
                if not result:
                    return f"插件 {plugin_name} 已成功写入，但重新加载失败，请检查插件 ID 是否正确"
                return f"插件 {plugin_name} 已成功写入并重新加载"
            except Exception as e:
                return f"插件 {plugin_name} 已成功写入，但重新加载失败: {e!r}"
        try:
            result = plugin.load_plugin(plugin_name, config)
            if result.error:
                return f"写入插件 {plugin_name} 成功，但加载失败: {result.error!r}"
            return f"插件 {plugin_name} 已成功写入并加载"
        except Exception as e:
            return f"写入插件 {plugin_name} 成功，但加载失败: {e!r}"
    finally:
        sys.path.remove(str(_dir))


@register_tool
async def visit_plugin(session: Session, plugin_id: str) -> str:
    """
    打开插件的源代码文件。

    **该工具需要先检测超级用户权限。**

    Args:
        session (Session): 当前会话对象
        plugin_id: 已加载插件的 ID (插件的导入路径)
    """
    if not await detect_superuser(session):
        return "当前用户没有权限使用此工具"
    plg = plugin.find_plugin(plugin_id)
    if not plg:
        return f"插件 {plugin_id} 未找到，请检查插件 ID 是否正确"
    try:
        path = Path(plg.module.__file__).absolute()  # type: ignore
        if not path.exists():
            return f"插件 {plugin_id} 的源代码文件未找到"
        with open(path, encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        return f"读取插件 {plugin_id} 的源代码文件时发生错误: {e!r}"
