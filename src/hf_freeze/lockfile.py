"""Schema-v1 lockfile creation, parsing, and atomic persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from hf_freeze.hub import HubResolver
from hf_freeze.models import (
    CallKind,
    DependencyFinding,
    DependencyKind,
    LockedDependency,
    LockedSource,
    Lockfile,
    RepoType,
    ScanResult,
)

SCHEMA_VERSION = 1


class LockError(Exception):
    """Base class for expected lock creation and parsing failures."""


class LockValidationError(LockError):
    """Scan findings cannot safely produce a complete lockfile."""


class LockfileFormatError(LockError):
    """Lockfile input is malformed or uses an unsupported version."""


_KINDS = {
    CallKind.FROM_PRETRAINED: DependencyKind.MODEL,
    CallKind.LOAD_DATASET: DependencyKind.DATASET,
    CallKind.HF_HUB_DOWNLOAD: DependencyKind.DIRECT_FILE,
    CallKind.SNAPSHOT_DOWNLOAD: DependencyKind.SNAPSHOT,
}


def resolve_lockfile(scan_result: ScanResult, resolver: HubResolver) -> Lockfile:
    """Validate findings, deduplicate resolution calls, and build a lockfile."""

    if scan_result.diagnostics:
        paths = ", ".join(
            diagnostic.source.path for diagnostic in scan_result.diagnostics
        )
        raise LockValidationError(f"scan failed for: {paths}")

    grouped: dict[
        tuple[RepoType, str, DependencyKind],
        tuple[str, set[LockedSource]],
    ] = {}
    unresolved: list[DependencyFinding] = []
    for finding in scan_result.findings:
        if (
            finding.repo_id is None
            or finding.repo_type is None
            or finding.revision_unresolved_reason is not None
        ):
            unresolved.append(finding)
            continue
        kind = _KINDS[finding.call_kind]
        if (
            finding.requested_revision is not None
            and not finding.requested_revision.strip()
        ):
            unresolved.append(finding)
            continue
        revision = (
            "main" if finding.requested_revision is None else finding.requested_revision
        )
        key = (finding.repo_type, finding.repo_id, kind)
        source = LockedSource(
            path=_source_path(finding.source.path, "source path"),
            line=finding.source.line,
            call=finding.call_kind,
        )
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = (revision, {source})
        elif previous[0] != revision:
            raise LockValidationError(
                f"conflicting revisions for {finding.repo_type.value} repository "
                f"'{finding.repo_id}' ({kind.value}): '{previous[0]}' and '{revision}'"
            )
        else:
            previous[1].add(source)

    if unresolved:
        locations = ", ".join(
            f"{finding.source.path}:{finding.source.line}" for finding in unresolved
        )
        raise LockValidationError(f"unresolved Hub dependencies at: {locations}")

    resolved: dict[tuple[RepoType, str, str], str] = {}
    dependencies: list[LockedDependency] = []
    for (repo_type, repo_id, kind), (revision, sources) in sorted(
        grouped.items(),
        key=lambda item: (item[0][0].value, item[0][1], item[0][2].value),
    ):
        resolution_key = (repo_type, repo_id, revision)
        sha = resolved.get(resolution_key)
        if sha is None:
            sha = resolver.resolve(repo_id, repo_type, revision)
            resolved[resolution_key] = sha
        dependencies.append(
            LockedDependency(
                repo_id=repo_id,
                repo_type=repo_type,
                kind=kind,
                requested_revision=revision,
                sha=sha,
                sources=tuple(sorted(sources, key=_source_key)),
            )
        )
    return Lockfile(version=SCHEMA_VERSION, dependencies=tuple(dependencies))


def lockfile_to_dict(lockfile: Lockfile) -> dict[str, object]:
    """Convert a lockfile to JSON-compatible values with explicit key order."""

    return {
        "version": lockfile.version,
        "dependencies": [
            {
                "repo_id": dependency.repo_id,
                "repo_type": dependency.repo_type.value,
                "kind": dependency.kind.value,
                "requested_revision": dependency.requested_revision,
                "sha": dependency.sha,
                "sources": [
                    {
                        "path": _source_path(source.path, "source path"),
                        "line": source.line,
                        "call": source.call.value,
                    }
                    for source in sorted(dependency.sources, key=_source_key)
                ],
            }
            for dependency in sorted(lockfile.dependencies, key=_dependency_key)
        ],
    }


def serialize_lockfile(lockfile: Lockfile) -> str:
    """Return canonical UTF-8-compatible schema-v1 JSON text."""

    return json.dumps(lockfile_to_dict(lockfile), indent=2, ensure_ascii=False) + "\n"


def parse_lockfile(text: str) -> Lockfile:
    """Parse and validate schema-v1 JSON text."""

    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise LockfileFormatError(f"malformed lockfile JSON: {error.msg}") from error
    return lockfile_from_dict(value)


def read_lockfile(path: str | Path) -> Lockfile:
    """Read and parse an existing UTF-8 lockfile without modifying it."""

    try:
        text = Path(path).read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise LockfileFormatError("lockfile is not valid UTF-8") from error
    return parse_lockfile(text)


def lockfile_from_dict(value: object) -> Lockfile:
    """Validate JSON-compatible values and return domain objects."""

    root = _object(value, "lockfile", {"version", "dependencies"})
    version = root["version"]
    if type(version) is not int:
        raise LockfileFormatError("lockfile version must be an integer")
    if version != SCHEMA_VERSION:
        raise LockfileFormatError(f"unsupported lockfile version: {version}")
    raw_dependencies = _list(root["dependencies"], "dependencies")
    dependencies = tuple(
        _dependency(item, f"dependencies[{index}]")
        for index, item in enumerate(raw_dependencies)
    )
    return Lockfile(version=version, dependencies=dependencies)


def write_lockfile(path: str | Path, lockfile: Lockfile) -> None:
    """Atomically replace a lockfile after serializing it fully in memory."""

    destination = Path(path)
    content = serialize_lockfile(lockfile)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _dependency(value: object, label: str) -> LockedDependency:
    item = _object(
        value,
        label,
        {"repo_id", "repo_type", "kind", "requested_revision", "sha", "sources"},
    )
    raw_sources = _list(item["sources"], f"{label}.sources")
    return LockedDependency(
        repo_id=_string(item["repo_id"], f"{label}.repo_id"),
        repo_type=_enum(RepoType, item["repo_type"], f"{label}.repo_type"),
        kind=_enum(DependencyKind, item["kind"], f"{label}.kind"),
        requested_revision=_string(
            item["requested_revision"], f"{label}.requested_revision"
        ),
        sha=_string(item["sha"], f"{label}.sha"),
        sources=tuple(
            _source(source, f"{label}.sources[{index}]")
            for index, source in enumerate(raw_sources)
        ),
    )


def _source(value: object, label: str) -> LockedSource:
    item = _object(value, label, {"path", "line", "call"})
    line = item["line"]
    if type(line) is not int or line < 1:
        raise LockfileFormatError(f"{label}.line must be a positive integer")
    return LockedSource(
        path=_source_path(item["path"], f"{label}.path"),
        line=line,
        call=_enum(CallKind, item["call"], f"{label}.call"),
    )


def _object(value: object, label: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise LockfileFormatError(
            f"{label} must contain exactly: {', '.join(sorted(keys))}"
        )
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise LockfileFormatError(f"{label} must be a list")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LockfileFormatError(f"{label} must be a non-empty string")
    return value


def _source_path(value: object, label: str) -> str:
    path = _string(value, label)
    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if (
        "\\" in path
        or posix_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or path != posix_path.as_posix()
    ):
        raise LockfileFormatError(
            f"{label} must be a canonical project-relative POSIX path"
        )
    return path


def _enum(enum_type: Any, value: object, label: str) -> Any:
    text = _string(value, label)
    try:
        return enum_type(text)
    except ValueError as error:
        raise LockfileFormatError(f"invalid {label}: {text}") from error


def _source_key(source: LockedSource) -> tuple[str, int, str]:
    return (source.path, source.line, source.call.value)


def _dependency_key(dependency: LockedDependency) -> tuple[str, str, str]:
    return (dependency.repo_type.value, dependency.repo_id, dependency.kind.value)
