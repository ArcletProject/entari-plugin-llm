import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from arclet.entari import local_data


@dataclass(slots=True)
class LLMState:
    default_model: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMState":
        value = data.get("default_model")
        default_model = value if isinstance(value, str) and value else None
        return cls(default_model=default_model)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _state_path() -> Path:
    return local_data.get_data_file("entari_plugin_llm", "state.json")


def _read_state(channel: str) -> LLMState:
    path = _state_path()
    if not path.exists():
        return LLMState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return LLMState()
    if not isinstance(data, dict):
        return LLMState()
    return LLMState.from_dict(data.get(channel, {}))


def _write_state(data: LLMState, channel: str) -> None:
    path = _state_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({channel: data.to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        try:
            existing_data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_data = {}
        if not isinstance(existing_data, dict):
            existing_data = {}
        existing_data[channel] = data.to_dict()
        path.write_text(json.dumps(existing_data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_default_model(channel: str = "$default") -> str | None:
    return _read_state(channel).default_model


def set_default_model(model_name: str | None, channel: str = "$default") -> None:
    state = _read_state(channel)
    state.default_model = model_name if model_name else None
    _write_state(state, channel)
