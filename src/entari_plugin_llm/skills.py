from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from agno.skills import LocalSkills, Skills

SkillMode = Literal["auto", "always"]
_SKILL_MODES = {"auto", "always"}
_BUILTIN_SKILLS_DIR = Path(__file__).with_name("skills")


class SkillError(RuntimeError):
    """Raised when a configured skill cannot be resolved or loaded."""


@dataclass(frozen=True)
class SkillSelection:
    """A resolved skill reference with its activation mode.

    ``auto`` lets Agno expose the skill metadata and load its instructions on
    demand through the skill tools. ``always`` preloads the instructions into
    the system prompt, which is useful for output-format contracts.
    """

    name: str
    mode: SkillMode = "auto"


def normalize_skills(items: Iterable[Any] | Any | None) -> list[SkillSelection]:
    """Normalize config values into unique ``SkillSelection`` entries.

    Strings are accepted as a shortcut for ``mode="auto"``. If the same skill
    is configured both ways, ``always`` wins.
    """

    if items is None:
        return []
    if isinstance(items, (str, SkillSelection)) or not isinstance(items, Iterable):
        items = [items]

    modes: dict[str, SkillMode] = {}
    order: list[str] = []
    for item in items:
        if isinstance(item, str):
            name, raw_mode = item, "auto"
        elif isinstance(item, SkillSelection):
            name, raw_mode = item.name, item.mode
        elif isinstance(item, dict):
            name = item.get("name")
            raw_mode = item.get("mode", "auto")
        else:
            name = getattr(item, "name", None)
            raw_mode = getattr(item, "mode", None) or "auto"

        if not isinstance(name, str) or not name.strip():
            raise SkillError("Skill entries must be a name, path, or object with a non-empty 'name'.")
        name = name.strip()
        mode = str(raw_mode)
        if mode not in _SKILL_MODES:
            raise SkillError(f"Skill {name!r} has invalid mode {mode!r}; expected 'auto' or 'always'.")

        if name not in modes:
            order.append(name)
            modes[name] = mode  # type: ignore[assignment]
        elif mode == "always":
            modes[name] = "always"

    return [SkillSelection(name, modes[name]) for name in order]


def resolve_selections(model_config: Any, global_skills: Sequence[Any]) -> list[SkillSelection]:
    """Resolve a model's skills, falling back to the global configuration.

    ``None`` means inherit; an empty list explicitly disables skills for that
    model.
    """

    raw = getattr(model_config, "skills", None)
    return normalize_skills(global_skills if raw is None else raw)


def resolve_skill_path(name: str) -> Path:
    """Resolve a built-in skill name or a user-provided skill path."""

    builtin = _BUILTIN_SKILLS_DIR / name
    if builtin.is_dir() and (builtin / "SKILL.md").is_file():
        return builtin.resolve()

    path = Path(name).expanduser()
    if path.is_file() and path.name == "SKILL.md":
        return path.parent.resolve()
    if path.is_dir():
        if (path / "SKILL.md").is_file():
            return path.resolve()
        if any((child / "SKILL.md").is_file() for child in path.iterdir() if child.is_dir()):
            return path.resolve()

    raise SkillError(f"Skill {name!r} not found. Use a built-in skill name or a path containing SKILL.md.")


@lru_cache(maxsize=16)
def _load_skills(paths: tuple[str, ...]) -> Skills:
    return Skills(loaders=[LocalSkills(path) for path in paths])


def _load_selected_skills(selections: Sequence[SkillSelection]) -> Skills:
    paths: list[str] = []
    for selection in selections:
        path = str(resolve_skill_path(selection.name))
        if path not in paths:
            paths.append(path)
    return _load_skills(tuple(paths))


def build_skills(selections: Sequence[SkillSelection]) -> Skills | None:
    """Build the Agno ``Skills`` object for auto-loaded skills."""

    if not selections:
        return None
    return _load_selected_skills(selections)


def preload_skill_instructions(selections: Sequence[SkillSelection]) -> str:
    """Return instructions for skills that must always be active."""

    always = [selection for selection in selections if selection.mode == "always"]
    if not always:
        return ""

    skills = _load_selected_skills(always)
    sections = [
        f"## Skill: {skill.name}\n\n{skill.instructions.strip()}"
        for skill in skills.get_all_skills()
        if skill.instructions.strip()
    ]
    return "\n\n".join(sections)


def apply_skills(
    instructions: str | None,
    selections: Sequence[SkillSelection],
) -> tuple[str | None, Skills | None]:
    """Apply skill selections to an agent's instructions and tools."""

    preload = preload_skill_instructions(selections)
    if preload:
        instructions = f"{instructions}\n\n{preload}" if instructions else preload

    auto = [selection for selection in selections if selection.mode == "auto"]
    return instructions, build_skills(auto)


__all__ = [
    "SkillError",
    "SkillMode",
    "SkillSelection",
    "apply_skills",
    "build_skills",
    "normalize_skills",
    "preload_skill_instructions",
    "resolve_selections",
    "resolve_skill_path",
]
