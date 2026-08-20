import asyncio
import platform

from arclet.entari import Session, local_data
from arclet.entari.filter import superusers

from entari_plugin_llm import register_tool

superusers_check = superusers().check


@register_tool
async def detect_platform() -> str:
    """
    检测当前运行环境的平台类型。

    Returns:
        str: 平台类型，例如 "Linux", "Windows", "Darwin" 等
    """
    return platform.system()


@register_tool
async def detect_superuser(session: Session) -> bool:
    """
    检测当前用户是否为超级用户。

    Args:
        session (Session): 当前会话对象

    Returns:
        bool: 如果当前用户是超级用户，则返回 True，否则返回 False
    """
    return (await superusers_check(session)) is None


@register_tool
async def run_command(session: Session, command: str) -> str:
    """
    执行指定的 shell 命令并返回输出结果。

    **该工具需要先检测超级用户权限。**

    Args:
        session (Session): 当前会话对象
        command (str): 要执行的 shell 命令

    Returns:
        str: 命令输出结果，如果命令执行失败，则返回错误信息
    """
    if not await detect_superuser(session):
        return "当前用户没有权限使用此工具"
    try:
        process = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            return f"命令执行失败，错误信息: {stderr.decode().strip()}"
        return stdout.decode().strip()
    except Exception as e:
        return f"执行命令时发生错误: {e}"


@register_tool
async def read_file(session: Session, file_path: str) -> str:
    """
    读取指定路径的文件内容并返回。

    **该工具需要先检测超级用户权限。**

    Args:
        session (Session): 当前会话对象
        file_path (str): 文件的绝对路径或相对路径

    Returns:
        str: 文件内容，如果文件不存在或无法读取，则返回错误信息
    """
    if not await detect_superuser(session):
        return "当前用户没有权限使用此工具"
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"文件 {file_path} 未找到"
    except Exception as e:
        return f"读取文件 {file_path} 时发生错误: {e}"


@register_tool
async def write_file(session: Session, file_name: str, content: str) -> str:
    """
    将指定内容写入指定路径的文件。注意该工具会限制只在特定目录下写入文件，以防止写入敏感目录。

    **该工具需要先检测超级用户权限。**

    Args:
        session (Session): 当前会话对象
        file_name (str): 文件名
        content (str): 要写入文件的内容

    Returns:
        str: 写入结果信息，如果写入失败，则返回错误信息
    """
    if not await detect_superuser(session):
        return "当前用户没有权限使用此工具"
    path = local_data.get_cache_file("llm", file_name)
    try:
        with open(path, "w+", encoding="utf-8") as f:
            f.write(content)
        return f"文件 {path} 已成功写入"
    except Exception as e:
        return f"写入文件 {path} 时发生错误: {e}"
