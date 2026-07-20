"""Schema-v1 lockfile creation, parsing, and atomic persistence."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from hf_freeze.models import (
    CALL_KIND_TO_DEPENDENCY_KIND,
    CallKind,
    DependencyFinding,
    DependencyKind,
    DiagnosticSeverity,
    LockedDependency,
    LockedSource,
    Lockfile,
    RepoType,
    ScanResult,
)

if TYPE_CHECKING:
    from hf_freeze.hub import HubResolver

SCHEMA_VERSION = 1
_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
_Identity = tuple[RepoType, str, DependencyKind]


class LockError(Exception):
    """Base class for expected lock creation and parsing failures."""


class LockValidationError(LockError):
    """Scan findings cannot safely produce a complete lockfile."""


class LockfileFormatError(LockError):
    """Lockfile input is malformed or uses an unsupported version."""


class LockSelectionError(LockError):
    """A repository cannot be selected safely for an update."""


def select_repository_entries(
    lockfile: Lockfile, repo_id: str
) -> tuple[LockedDependency, ...]:
    """Return exact compatible matches, allowing different kinds and sources."""

    matches = tuple(item for item in lockfile.dependencies if item.repo_id == repo_id)
    if not matches:
        raise LockSelectionError(f"repository '{repo_id}' is not present in hf.lock")
    identities: set[_Identity] = set()
    for item in matches:
        identity = (item.repo_type, item.repo_id, item.kind)
        if identity in identities:
            raise LockSelectionError(
                f"hf.lock contains multiple entries for {item.repo_type.value} "
                f"repository '{repo_id}' ({item.kind.value}); remove the duplicates "
                "before updating"
            )
        identities.add(identity)
    states = {(item.repo_type, item.sha, item.requested_revision) for item in matches}
    if len(states) != 1:
        raise LockSelectionError(
            f"repository '{repo_id}' has ambiguous/incompatible hf.lock entries "
            "with different types, SHAs, or requested revisions"
        )
    return matches


def update_repository_entries(
    lockfile: Lockfile,
    repo_id: str,
    requested_revision: str,
    sha: str,
) -> Lockfile:
    """Return a deterministic lockfile updating every compatible exact match."""

    select_repository_entries(lockfile, repo_id)
    dependencies = (
        replace(item, requested_revision=requested_revision, sha=sha)
        if item.repo_id == repo_id
        else item
        for item in lockfile.dependencies
    )
    return replace(
        lockfile, dependencies=tuple(sorted(dependencies, key=_dependency_key))
    )


def resolve_lockfile(
    scan_result: ScanResult,
    resolver: HubResolver,
    existing_lockfile: Lockfile | None = None,
) -> Lockfile:
    """Validate findings, deduplicate resolution calls, and build a lockfile."""

    fatal_diagnostics = tuple(
        diagnostic
        for diagnostic in scan_result.diagnostics
        if diagnostic.severity is DiagnosticSeverity.ERROR
    )
    if fatal_diagnostics:
        paths = ", ".join(
            f"{diagnostic.source.path}:{diagnostic.source.line}: {diagnostic.message}"
            for diagnostic in fatal_diagnostics
        )
        raise LockValidationError(f"scan failed: {paths}")

    grouped: dict[_Identity, tuple[str, set[LockedSource]]] = {}
    unresolved: list[DependencyFinding] = []
    for finding in scan_result.findings:
        if (
            finding.repo_id is None
            or finding.repo_type is None
            or finding.revision_unresolved_reason is not None
        ):
            unresolved.append(finding)
            continue
        kind = CALL_KIND_TO_DEPENDENCY_KIND[finding.call_kind]
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

    existing = _existing_dependency_lookup(existing_lockfile)
    preserved: dict[_Identity, LockedDependency] = {}
    for identity, (revision, _) in grouped.items():
        if not _COMMIT_SHA.fullmatch(revision):
            continue
        previous = existing.get(identity)
        if previous is None:
            continue
        if revision != previous.sha:
            repo_type, repo_id, kind = identity
            raise LockValidationError(
                f"pinned SHA mismatch for {repo_type.value} repository '{repo_id}' "
                f"({kind.value}): source has '{revision}' but hf.lock has "
                f"'{previous.sha}'"
            )
        preserved[identity] = previous

    resolved: dict[tuple[RepoType, str, str], str] = {}
    dependencies: list[LockedDependency] = []
    for identity, (revision, sources) in sorted(
        grouped.items(),
        key=lambda item: (item[0][0].value, item[0][1], item[0][2].value),
    ):
        repo_type, repo_id, kind = identity
        previous = preserved.get(identity)
        if previous is None:
            requested_revision = revision
            resolution_key = (repo_type, repo_id, revision)
            sha = resolved.get(resolution_key)
            if sha is None:
                sha = resolver.resolve(repo_id, repo_type, revision)
                resolved[resolution_key] = sha
        else:
            requested_revision = previous.requested_revision
            sha = previous.sha
        dependencies.append(
            LockedDependency(
                repo_id=repo_id,
                repo_type=repo_type,
                kind=kind,
                requested_revision=requested_revision,
                sha=sha,
                sources=tuple(sorted(sources, key=_source_key)),
            )
        )
    return Lockfile(version=SCHEMA_VERSION, dependencies=tuple(dependencies))


def _existing_dependency_lookup(
    lockfile: Lockfile | None,
) -> dict[_Identity, LockedDependency]:
    """Return an unambiguous identity lookup for an existing lockfile."""

    existing: dict[_Identity, LockedDependency] = {}
    if lockfile is None:
        return existing
    for dependency in sorted(lockfile.dependencies, key=_dependency_key):
        identity = (dependency.repo_type, dependency.repo_id, dependency.kind)
        if identity in existing:
            repo_type, repo_id, kind = identity
            raise LockValidationError(
                f"hf.lock contains multiple entries for {repo_type.value} repository "
                f"'{repo_id}' ({kind.value}); remove the duplicates before re-locking"
            )
        existing[identity] = dependency
    return existing


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
