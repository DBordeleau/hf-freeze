"""Typed project configuration and root discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from hf_freeze.models import RepoType

_CONFIG_KEYS = frozenset({"include", "exclude", "dependencies", "bindings"})
_DEPENDENCY_KEYS = frozenset({"repo_id", "repo_type", "revision"})
_BINDING_KEYS = frozenset({"environment"})
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class ConfigError(ValueError):
    """A path-specific project configuration error."""


@dataclass(frozen=True)
class DeclaredDependency:
    """A named, committed Hub dependency declaration."""

    name: str
    repo_id: str
    repo_type: RepoType
    revision: str


@dataclass(frozen=True)
class ProjectConfig:
    """Immutable normalized values from ``[tool.hf-freeze]``."""

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    dependencies: tuple[DeclaredDependency, ...] = ()
    environment_bindings: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ProjectContext:
    """The stable root and configuration resolved for one command."""

    root: Path
    config_path: Path | None
    config: ProjectConfig


def resolve_project_context(path: str | Path = ".") -> ProjectContext:
    """Resolve the nearest configured project without merging ancestors."""

    requested = Path(path)
    normalized = requested.resolve(strict=False)
    start = normalized.parent if normalized.is_file() else normalized

    for directory in (start, *start.parents):
        config_path = directory / "pyproject.toml"
        if not config_path.is_file():
            continue
        document = _read_toml(config_path)
        configured = _configured_table(document, config_path)
        if configured is not None:
            return ProjectContext(
                root=directory,
                config_path=config_path,
                config=_parse_config(configured, config_path),
            )

    fallback = normalized if normalized.is_dir() else normalized.parent
    return ProjectContext(fallback, None, ProjectConfig())


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(
            f"invalid configuration {path}: malformed TOML: {error}"
        ) from error
    except OSError as error:
        raise ConfigError(
            f"invalid configuration {path}: cannot read file: {error}"
        ) from error
    return document


def _configured_table(document: dict[str, Any], path: Path) -> dict[str, Any] | None:
    tool = document.get("tool")
    if tool is None:
        return None
    if not isinstance(tool, dict):
        return None
    configured = tool.get("hf-freeze")
    if configured is None:
        return None
    if not isinstance(configured, dict):
        raise ConfigError(
            f"invalid configuration {path}: [tool.hf-freeze] must be a table"
        )
    return configured


def _parse_config(table: dict[str, Any], path: Path) -> ProjectConfig:
    _reject_unknown(table, _CONFIG_KEYS, "[tool.hf-freeze]", path)
    include = _string_tuple(table.get("include", []), "include", path)
    exclude = _string_tuple(table.get("exclude", []), "exclude", path)
    dependencies = _parse_dependencies(table.get("dependencies", {}), path)
    bindings = _parse_bindings(table.get("bindings", {}), dependencies, path)
    return ProjectConfig(include, exclude, dependencies, bindings)


def _string_tuple(value: Any, field: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(
            f"invalid configuration {path}: {field} must be an array of strings"
        )
    return tuple(value)


def _parse_dependencies(value: Any, path: Path) -> tuple[DeclaredDependency, ...]:
    if not isinstance(value, dict):
        raise ConfigError(f"invalid configuration {path}: dependencies must be a table")
    parsed: list[DeclaredDependency] = []
    for name in sorted(value):
        if not isinstance(name, str) or _NAME_PATTERN.fullmatch(name) is None:
            raise ConfigError(
                f"invalid configuration {path}: dependency name {name!r} must contain "
                "only letters, digits, '_' or '-'"
            )
        declaration = value[name]
        if not isinstance(declaration, dict):
            raise ConfigError(
                f"invalid configuration {path}: dependency {name!r} must be a table"
            )
        location = f"dependency {name!r}"
        _reject_unknown(declaration, _DEPENDENCY_KEYS, location, path)
        repo_id = _required_string(declaration, "repo_id", location, path)
        revision = _required_string(declaration, "revision", location, path)
        repo_type_value = _required_string(declaration, "repo_type", location, path)
        try:
            repo_type = RepoType(repo_type_value)
        except ValueError as error:
            raise ConfigError(
                f"invalid configuration {path}: {location}.repo_type must be "
                "'model' or 'dataset'"
            ) from error
        parsed.append(DeclaredDependency(name, repo_id, repo_type, revision))
    return tuple(parsed)


def _parse_bindings(
    value: Any,
    dependencies: tuple[DeclaredDependency, ...],
    path: Path,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        raise ConfigError(f"invalid configuration {path}: bindings must be a table")
    _reject_unknown(value, _BINDING_KEYS, "bindings", path)
    environment = value.get("environment", {})
    if not isinstance(environment, dict):
        raise ConfigError(
            f"invalid configuration {path}: bindings.environment must be a table"
        )
    dependency_names = {item.name for item in dependencies}
    parsed: list[tuple[str, str]] = []
    for variable in sorted(environment):
        target = environment[variable]
        if not isinstance(target, str) or not target:
            raise ConfigError(
                f"invalid configuration {path}: environment binding {variable!r} "
                "must name a dependency"
            )
        if target not in dependency_names:
            raise ConfigError(
                f"invalid configuration {path}: environment binding {variable!r} "
                f"references unknown dependency {target!r}"
            )
        parsed.append((variable, target))
    return tuple(parsed)


def _required_string(
    table: dict[str, Any], field: str, location: str, path: Path
) -> str:
    value = table.get(field)
    if not isinstance(value, str) or not value:
        raise ConfigError(
            f"invalid configuration {path}: {location}.{field} must be a "
            "non-empty string"
        )
    return value


def _reject_unknown(
    table: dict[str, Any], allowed: frozenset[str], location: str, path: Path
) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        rendered = ", ".join(repr(item) for item in unknown)
        raise ConfigError(
            f"invalid configuration {path}: unknown key(s) in {location}: {rendered}"
        )
