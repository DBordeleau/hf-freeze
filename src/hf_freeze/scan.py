"""Static, no-network discovery of supported Hugging Face Hub calls."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider, ScopeProvider

from hf_freeze.models import (
    CallKind,
    DependencyFinding,
    RepoType,
    ScanDiagnostic,
    ScanResult,
    SourceLocation,
)

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


@dataclass(frozen=True)
class _CallSpec:
    kind: CallKind
    repo_type: RepoType
    keyword_names: tuple[str, ...]


class _ConstantCollector(cst.CSTVisitor):
    """Record literal values for simple binding nodes."""

    def __init__(self) -> None:
        self.values: dict[int, str | None] = {}

    def visit_Assign(self, node: cst.Assign) -> None:
        value = _literal_string(node.value)
        for target in node.targets:
            if isinstance(target.target, cst.Name):
                self.values[id(target.target)] = value

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        if isinstance(node.target, cst.Name):
            value = _literal_string(node.value) if node.value is not None else None
            self.values[id(node.target)] = value

    def visit_AugAssign(self, node: cst.AugAssign) -> None:
        if isinstance(node.target, cst.Name):
            self.values[id(node.target)] = None


class _FindingVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider)

    def __init__(
        self, display_path: str, binding_values: dict[int, str | None]
    ) -> None:
        self.display_path = display_path
        self.binding_values = binding_values
        self.findings: list[DependencyFinding] = []

    def visit_Call(self, node: cst.Call) -> None:
        spec = _match_call(node.func)
        if spec is None:
            return

        repo_expression = _find_argument(node, spec.keyword_names)
        repo_id, unresolved_reason = _resolve_repo_id(
            repo_expression, self._resolve_name
        )
        if repo_id is not None and _is_obvious_local_path(repo_id):
            return

        revision_expression = _find_argument(node, ("revision",), positional_index=None)
        requested_revision, revision_unresolved_reason = _resolve_revision(
            revision_expression, self._resolve_name
        )
        position = self.get_metadata(PositionProvider, node).start
        self.findings.append(
            DependencyFinding(
                repo_id=repo_id,
                repo_type=_repo_type(node, spec.repo_type, self._resolve_name),
                call_kind=spec.kind,
                requested_revision=requested_revision,
                source=SourceLocation(
                    path=self.display_path,
                    line=position.line,
                    column=position.column,
                ),
                unresolved_reason=unresolved_reason,
                revision_unresolved_reason=revision_unresolved_reason,
            )
        )

    def _resolve_name(self, expression: cst.Name) -> str | None:
        scope = self.get_metadata(ScopeProvider, expression)
        assignments = scope[expression.value]
        if len(assignments) != 1:
            return None
        assignment = next(iter(assignments))
        return self.binding_values.get(id(assignment.node))


def scan_path(path: str | Path = ".") -> ScanResult:
    """Scan one Python file or a directory tree without importing project code."""

    target = Path(path)
    if not target.exists():
        raise ValueError(f"scan path does not exist: {target}")
    if target.is_file() and target.suffix != ".py":
        raise ValueError(f"scan path is not a Python file: {target}")

    findings: list[DependencyFinding] = []
    diagnostics: list[ScanDiagnostic] = []
    for source_path, display_path in _python_files(target):
        file_findings, diagnostic = _scan_file(source_path, display_path)
        findings.extend(file_findings)
        if diagnostic is not None:
            diagnostics.append(diagnostic)

    def location_key(item: DependencyFinding | ScanDiagnostic) -> tuple[str, int, int]:
        return (item.source.path, item.source.line, item.source.column)

    return ScanResult(
        findings=tuple(sorted(findings, key=location_key)),
        diagnostics=tuple(sorted(diagnostics, key=location_key)),
    )


def _python_files(target: Path) -> list[tuple[Path, str]]:
    if target.is_file():
        return [(target, target.name)]

    files: list[tuple[Path, str]] = []
    for directory, directory_names, file_names in os.walk(target):
        directory_names[:] = sorted(
            name for name in directory_names if name not in DEFAULT_EXCLUDED_DIRECTORIES
        )
        base = Path(directory)
        for file_name in sorted(file_names):
            if file_name.endswith(".py"):
                source_path = base / file_name
                files.append((source_path, source_path.relative_to(target).as_posix()))
    return sorted(files, key=lambda item: item[1])


def _scan_file(
    source_path: Path, display_path: str
) -> tuple[list[DependencyFinding], ScanDiagnostic | None]:
    try:
        module = cst.parse_module(source_path.read_bytes())
    except cst.ParserSyntaxError as error:
        message = " ".join(error.message.split())
        return [], ScanDiagnostic(
            source=SourceLocation(
                path=display_path,
                line=error.raw_line,
                column=error.raw_column,
            ),
            message=f"parse error: {message}",
        )
    except OSError as error:
        return [], ScanDiagnostic(
            source=SourceLocation(path=display_path, line=1, column=0),
            message=f"read error: {error}",
        )

    wrapper = MetadataWrapper(module)
    collector = _ConstantCollector()
    wrapper.module.visit(collector)
    visitor = _FindingVisitor(display_path, collector.values)
    wrapper.visit(visitor)
    return visitor.findings, None


def _match_call(function: cst.BaseExpression) -> _CallSpec | None:
    if isinstance(function, cst.Attribute):
        name = function.attr.value
    elif isinstance(function, cst.Name):
        name = function.value
    else:
        return None

    if name == "load_dataset":
        return _CallSpec(CallKind.LOAD_DATASET, RepoType.DATASET, ("path",))
    if name == "hf_hub_download":
        return _CallSpec(CallKind.HF_HUB_DOWNLOAD, RepoType.MODEL, ("repo_id",))
    if name == "snapshot_download":
        return _CallSpec(CallKind.SNAPSHOT_DOWNLOAD, RepoType.MODEL, ("repo_id",))
    if name == "from_pretrained" and isinstance(function, cst.Attribute):
        return _CallSpec(
            CallKind.FROM_PRETRAINED,
            RepoType.MODEL,
            ("pretrained_model_name_or_path", "model_name_or_path"),
        )
    return None


def _find_argument(
    call: cst.Call,
    keyword_names: tuple[str, ...],
    positional_index: int | None = 0,
) -> cst.BaseExpression | None:
    for argument in call.args:
        if argument.keyword is not None and argument.keyword.value in keyword_names:
            return argument.value
    if positional_index is None:
        return None
    positional = [
        argument.value
        for argument in call.args
        if argument.keyword is None and not argument.star
    ]
    if len(positional) > positional_index:
        return positional[positional_index]
    return None


def _literal_string(expression: cst.BaseExpression | None) -> str | None:
    if not isinstance(expression, cst.SimpleString):
        return None
    try:
        value = expression.evaluated_value
    except ValueError:
        return None
    return value if isinstance(value, str) else None


def _resolve_optional_string(
    expression: cst.BaseExpression | None,
    resolve_name: Callable[[cst.Name], str | None],
) -> str | None:
    if expression is None:
        return None
    literal = _literal_string(expression)
    if literal is not None:
        return literal
    if isinstance(expression, cst.Name):
        return resolve_name(expression)
    return None


def _resolve_revision(
    expression: cst.BaseExpression | None,
    resolve_name: Callable[[cst.Name], str | None],
) -> tuple[str | None, str | None]:
    if expression is None:
        return None, None
    literal = _literal_string(expression)
    if literal is not None:
        if literal.strip():
            return literal, None
        return None, "revision is empty or whitespace-only"
    if isinstance(expression, cst.Name):
        value = resolve_name(expression)
        if value is not None:
            if value.strip():
                return value, None
            return None, "revision is empty or whitespace-only"
        return (
            None,
            f"revision name '{expression.value}' does not have one "
            "unambiguous string assignment",
        )
    if isinstance(expression, cst.Subscript):
        reason = "revision is a subscript expression"
    elif isinstance(expression, cst.FormattedString):
        reason = "revision is an interpolated string"
    elif isinstance(expression, cst.Call):
        reason = "revision is returned by a function call"
    elif isinstance(expression, cst.IfExp):
        reason = "revision is a conditional expression"
    elif isinstance(expression, cst.Attribute):
        reason = "revision is an attribute expression"
    else:
        reason = "revision is a dynamic expression"
    return None, reason


def _resolve_repo_id(
    expression: cst.BaseExpression | None,
    resolve_name: Callable[[cst.Name], str | None],
) -> tuple[str | None, str | None]:
    if expression is None:
        return None, "repository ID argument is missing"
    literal = _literal_string(expression)
    if literal is not None:
        return literal, None
    if isinstance(expression, cst.Name):
        value = resolve_name(expression)
        if value is not None:
            return value, None
        return (
            None,
            f"repository ID name '{expression.value}' does not have one "
            "unambiguous string assignment",
        )
    if isinstance(expression, cst.Subscript):
        reason = "repository ID is a subscript expression"
    elif isinstance(expression, cst.FormattedString):
        reason = "repository ID is an interpolated string"
    elif isinstance(expression, cst.Call):
        reason = "repository ID is returned by a function call"
    elif isinstance(expression, cst.IfExp):
        reason = "repository ID is a conditional expression"
    elif isinstance(expression, cst.Attribute):
        reason = "repository ID is an attribute expression"
    else:
        reason = "repository ID is a dynamic expression"
    return None, reason


def _repo_type(
    call: cst.Call,
    default: RepoType,
    resolve_name: Callable[[cst.Name], str | None],
) -> RepoType | None:
    expression = _find_argument(call, ("repo_type",), positional_index=None)
    if expression is None:
        return default
    value = _resolve_optional_string(expression, resolve_name)
    if value == RepoType.MODEL.value:
        return RepoType.MODEL
    if value == RepoType.DATASET.value:
        return RepoType.DATASET
    return None


def _is_obvious_local_path(repo_id: str) -> bool:
    normalized = repo_id.strip()
    lowered = normalized.lower()
    return (
        normalized in {".", "..", "~"}
        or normalized.startswith(("./", "../", ".\\", "..\\", "/", "\\", "~/", "~\\"))
        or lowered.startswith("file:")
        or re.match(r"^[a-zA-Z]:(?:[\\/]|[^/\\])", normalized) is not None
    )
