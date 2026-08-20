import sys

from arclet.entari import Session, local_data, plugin

from entari_plugin_llm import register_tool
from entari_plugin_llm.builtins.tools.generic import detect_superuser


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
    sys.path.insert(0, str(local_data.get_cache_dir("llm")))
    try:
        plg = plugin.load_plugin(import_path, config)
        if plg is None:
            return f"未能加载插件 {import_path}，请检查导入路径是否正确"
        return f"插件 {import_path} 已成功加载"
    except Exception as e:
        return f"加载插件 {import_path} 失败: {e}"
    finally:
        sys.path.remove(str(local_data.get_cache_dir("llm")))


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
        return f"卸载插件 {plugin_id} 失败: {e}"


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
    sys.path.insert(0, str(local_data.get_cache_dir("llm")))
    try:
        result = await plugin.reload_plugin(plugin_id)
        if not result:
            return f"未能重新加载插件 {plugin_id}，请检查插件 ID 是否正确"
        return f"插件 {plugin_id} 已成功重新加载"
    except Exception as e:
        return f"重新加载插件 {plugin_id} 失败: {e}"
    finally:
        sys.path.remove(str(local_data.get_cache_dir("llm")))
