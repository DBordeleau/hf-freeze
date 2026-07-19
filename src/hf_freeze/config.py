"""Typed project configuration and root discovery."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pathspec import GitIgnoreSpec
from pathspec.patterns.gitignore import GitIgnorePatternError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from hf_freeze.models import RepoType

_CONFIG_KEYS = frozenset({"include", "exclude", "dependencies", "bindings"})
_DEPENDENCY_KEYS = frozenset({"repo_id", "repo_type", "revision"})
_BINDING_KEYS = frozenset({"environment"})
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

DEFAULT_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "env",
        "node_modules",
        "venv",
    }
)


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


def iter_scoped_python_files(
    context: ProjectContext, requested_path: str | Path
) -> tuple[tuple[Path, str], ...]:
    """Return deterministic Python candidates with their source display paths."""

    requested = Path(requested_path).resolve(strict=False)
    if not requested.exists():
        raise ValueError(f"scan path does not exist: {requested_path}")
    if requested.is_file() and requested.suffix != ".py":
        raise ValueError(f"scan path is not a Python file: {requested_path}")

    if context.config_path is None:
        display_root = requested.parent if requested.is_file() else requested
    else:
        try:
            requested.relative_to(context.root)
        except ValueError as error:
            raise ValueError(
                f"scan path is outside configured project root {context.root}: "
                f"{requested}"
            ) from error
        display_root = context.root

    include = _compile_patterns(context.config.include)
    exclude = _compile_patterns(context.config.exclude)
    selected: list[tuple[Path, str]] = []
    for source_path in _candidate_python_files(requested):
        display_path = source_path.relative_to(display_root).as_posix()
        relative_parts = source_path.relative_to(display_root).parts[:-1]
        if any(part in DEFAULT_EXCLUDED_DIRECTORIES for part in relative_parts):
            continue
        if context.config_path is not None:
            included = not context.config.include or include.match_file(display_path)
            if not included or exclude.match_file(display_path):
                continue
        selected.append((source_path, display_path))
    return tuple(sorted(selected, key=lambda item: item[1]))


def _candidate_python_files(requested: Path) -> tuple[Path, ...]:
    if requested.is_file():
        return (requested,)

    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(requested):
        directory_names[:] = sorted(
            name for name in directory_names if name not in DEFAULT_EXCLUDED_DIRECTORIES
        )
        base = Path(directory)
        files.extend(base / name for name in sorted(file_names) if name.endswith(".py"))
    return tuple(files)


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
    _validate_patterns(include, "include", path)
    _validate_patterns(exclude, "exclude", path)
    dependencies = _parse_dependencies(table.get("dependencies", {}), path)
    bindings = _parse_bindings(table.get("bindings", {}), dependencies, path)
    return ProjectConfig(include, exclude, dependencies, bindings)


def _validate_patterns(patterns: tuple[str, ...], field: str, path: Path) -> None:
    for index, pattern in enumerate(patterns):
        try:
            _compile_patterns((pattern,))
        except GitIgnorePatternError as error:
            raise ConfigError(
                f"invalid configuration {path}: {field}[{index}] is not a valid "
                f"gitignore pattern: {error}"
            ) from error


def _compile_patterns(patterns: tuple[str, ...]) -> GitIgnoreSpec:
    return GitIgnoreSpec.from_lines(patterns)


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
