import inspect
from urllib.parse import quote

import httpx
from arclet.entari import Session, command, plugin
from arclet.letoderea.core import add_task

from entari_plugin_llm import LLMToolEvent

tools = plugin.dispatch(LLMToolEvent)
client = httpx.AsyncClient(timeout=30)


@plugin.collect_disposes
def dispose_client():
    add_task(client.aclose())


@tools
async def ask_user_for_argument(session: Session, prompt: str, timeout: int = 120):
    """
    向用户询问参数并等待

    Args:
        session (Session): 当前会话对象
        prompt (str): 询问提示语
        timeout (int): 超时时间，单位秒
    Returns:
        str: 用户输入的参数
        null: 等待超时或用户未输入
    """
    resp = await session.prompt(prompt, timeout=timeout)
    if not resp:
        return "未提供必要的信息，无法继续操作"
    return resp.extract_plain_text()


API = "https://wttr.in/{city}?format=j1&lang=zh"


async def _get_weather(city: str) -> dict:
    """
    获取指定城市的天气信息

    如果用户未给出指定城市名称，需要询问用户

    Args:
        city (str): 城市名称
    """
    url = API.format(city=quote(city))
    response = await client.get(url)
    if response.status_code != 200:
        return {"error": {"code": response.status_code, "message": response.text}}
    data = response.json()
    current_condition = data.get("current_condition", [])
    if current_condition:
        condition = current_condition[0]
        temperature = condition["temp_C"] + "°C"
        weather_desc = condition["lang_zh"][0]["value"]
        feels_like = condition["FeelsLikeC"] + "°C"
        humidity = condition["humidity"] + "%"
        wind_speed = condition["windspeedKmph"] + " km/h"
        uv_index = condition["uvIndex"]
        visibility = condition["visibility"] + " km"
        current_time = condition["localObsDateTime"]
        precip = condition["precipMM"] + " mm"
        return {
            "format": "* {type}: {content}",
            "temperature": temperature,
            "condition": weather_desc,
            "feels_like": feels_like,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "uv_index": uv_index,
            "visibility": visibility,
            "precipitation": precip,
            "observation_time": current_time,
        }
    return {"error": {"code": 404, "message": "未找到该城市的天气信息"}}


tools.register(_get_weather)


@command.on("weather {city?}")
async def get_weather(session: Session, city: str = ""):
    """
    获取指定城市的天气信息

    如果用户未给出指定城市名称，需要询问用户

    Args:
        session (Session): 当前会话对象
        city (str): 城市名称
    """
    if not city:
        resp = await session.prompt("你想查询哪个城市的天气？")
        if not resp:
            return "未提供城市名称，无法查询天气"
        city = resp.extract_plain_text()
    weather_info = await _get_weather(city)
    if "error" in weather_info:
        return f"获取 {city} 天气信息失败，状态码：{weather_info['code']}，错误信息：{weather_info['message']}。"
    return inspect.cleandoc(f"""
        {city} 当前天气信息（{weather_info["observation_time"]}）：
        天气状况：{weather_info["condition"]}
        温度：{weather_info["temperature"]}，体感温度：{weather_info["feels_like"]}
        湿度：{weather_info["humidity"]}
        风速：{weather_info["wind_speed"]}
        紫外线指数：{weather_info["uv_index"]}
        能见度：{weather_info["visibility"]}
        降雨量: {weather_info["precipitation"]}
    """)
