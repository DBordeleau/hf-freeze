"""Pure, deterministic repository-tree comparison for update review."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import PurePosixPath

from hf_freeze.models import LockedDependency, Lockfile

SEMANTIC_FILES = {"config.json", "tokenizer_config.json", "adapter_config.json"}
SEMANTIC_SIZE_LIMIT = 64 * 1024


class DiffError(Exception):
    """An expected failure to prepare a repository diff."""


class ArtifactCategory(str, Enum):
    ADAPTER = "adapter"
    TOKENIZER = "tokenizer"
    PROCESSOR = "processor"
    CONFIG = "config"
    REMOTE_CODE = "remote code"
    WEIGHTS = "weights"
    DATASET = "dataset"
    LICENSE = "license"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class ChangeState(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"
    METADATA_UNAVAILABLE = "metadata unavailable"


class IdentityKind(str, Enum):
    LFS_SHA256 = "LFS SHA-256"
    XET_HASH = "Xet hash"
    GIT_BLOB = "Git blob ID"
    SIZE = "size"


@dataclass(frozen=True)
class RepositoryFile:
    path: str
    size: int | None
    lfs_sha256: str | None = None
    xet_hash: str | None = None
    blob_id: str | None = None


@dataclass(frozen=True)
class FileChange:
    path: str
    category: ArtifactCategory
    state: ChangeState
    old: RepositoryFile | None
    new: RepositoryFile | None
    identity: IdentityKind | None = None
    semantic_paths: tuple[str, ...] | None = None
    semantic_unavailable: bool = False


@dataclass(frozen=True)
class DiffResult:
    files: tuple[FileChange, ...]
    changed_bytes: int
    unknown_changed_sizes: int


def select_locked_dependency(lockfile: Lockfile, repo_id: str) -> LockedDependency:
    """Select an exact repository match, rejecting incompatible duplicates."""

    matches = [item for item in lockfile.dependencies if item.repo_id == repo_id]
    if not matches:
        raise DiffError(f"repository '{repo_id}' is not present in hf.lock")
    identities = {
        (item.repo_type, item.sha, item.requested_revision) for item in matches
    }
    if len(identities) != 1:
        raise DiffError(
            f"repository '{repo_id}' has ambiguous hf.lock entries with different "
            "types, SHAs, or requested revisions"
        )
    return matches[0]


def categorize(path: str) -> ArtifactCategory:
    """Assign a path using a small, ordered artifact ruleset."""

    lower = path.lower()
    name = PurePosixPath(lower).name
    suffix = PurePosixPath(lower).suffix
    if name.startswith("adapter_") or "/adapters/" in f"/{lower}/":
        return ArtifactCategory.ADAPTER
    token_name = any(part in name for part in ("tokenizer", "vocab", "sentencepiece"))
    if token_name or name in {
        "merges.txt",
        "spiece.model",
        "special_tokens_map.json",
    }:
        return ArtifactCategory.TOKENIZER
    if any(part in name for part in ("processor", "feature_extractor")):
        return ArtifactCategory.PROCESSOR
    if name.endswith("config.json") or name in {"config.yaml", "config.yml"}:
        return ArtifactCategory.CONFIG
    if suffix == ".py":
        return ArtifactCategory.REMOTE_CODE
    if suffix in {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5"}:
        return ArtifactCategory.WEIGHTS
    if suffix in {".parquet", ".arrow", ".csv", ".tsv", ".jsonl"}:
        return ArtifactCategory.DATASET
    if name.startswith(("license", "licence", "copying")):
        return ArtifactCategory.LICENSE
    if name.startswith("readme") or suffix in {".md", ".rst"}:
        return ArtifactCategory.DOCUMENTATION
    return ArtifactCategory.OTHER


def compare_trees(
    old_files: tuple[RepositoryFile, ...], new_files: tuple[RepositoryFile, ...]
) -> DiffResult:
    """Compare paths using the strongest identity available on both sides."""

    old = _by_path(old_files)
    new = _by_path(new_files)
    changes: list[FileChange] = []
    changed_bytes = 0
    unknown_sizes = 0
    for path in sorted(old.keys() | new.keys()):
        before, after = old.get(path), new.get(path)
        identity: IdentityKind | None = None
        if before is None:
            state = ChangeState.ADDED
        elif after is None:
            state = ChangeState.REMOVED
        else:
            state, identity = _compare_file(before, after)
        change = FileChange(path, categorize(path), state, before, after, identity)
        changes.append(change)
        if state in {ChangeState.ADDED, ChangeState.CHANGED}:
            if after is None or after.size is None:
                unknown_sizes += 1
            else:
                changed_bytes += after.size
    return DiffResult(tuple(changes), changed_bytes, unknown_sizes)


def with_semantic_diff(
    result: DiffResult, path: str, old_text: str | None, new_text: str | None
) -> DiffResult:
    """Attach a bounded JSON key diff, retaining metadata on any parse failure."""

    try:
        if old_text is None or new_text is None:
            raise ValueError
        old = json.loads(old_text)
        new = json.loads(new_text)
        paths = _json_paths(old, new)
        unavailable = False
    except (json.JSONDecodeError, ValueError, TypeError):
        paths = None
        unavailable = True
    files = tuple(
        replace(item, semantic_paths=paths, semantic_unavailable=unavailable)
        if item.path == path
        else item
        for item in result.files
    )
    return replace(result, files=files)


def semantic_eligible(change: FileChange) -> bool:
    return (
        change.state is ChangeState.CHANGED
        and PurePosixPath(change.path).name in SEMANTIC_FILES
        and change.old is not None
        and change.new is not None
        and change.old.size is not None
        and change.new.size is not None
        and change.old.size <= SEMANTIC_SIZE_LIMIT
        and change.new.size <= SEMANTIC_SIZE_LIMIT
    )


def render_diff(repo_id: str, old_sha: str, new_sha: str, result: DiffResult) -> str:
    """Render stable human-readable output while hiding proven unchanged files."""

    lines = [repo_id, f"{old_sha} -> {new_sha}"]
    visible = [item for item in result.files if item.state is not ChangeState.UNCHANGED]
    for state in (ChangeState.ADDED, ChangeState.REMOVED, ChangeState.CHANGED):
        selected = [item for item in visible if item.state is state]
        if selected:
            lines.extend(["", state.value.capitalize()])
            _render_grouped(lines, selected)
    unavailable = [
        item for item in visible if item.state is ChangeState.METADATA_UNAVAILABLE
    ]
    if unavailable:
        lines.extend(["", "Metadata unavailable"])
        _render_grouped(lines, unavailable)
    if not visible:
        lines.extend(["", "No file changes detected."])

    remote = [
        item for item in result.files if item.category is ArtifactCategory.REMOTE_CODE
    ]
    remote_changed = any(
        item.state in {ChangeState.ADDED, ChangeState.REMOVED, ChangeState.CHANGED}
        for item in remote
    )
    remote_unavailable = any(
        item.state is ChangeState.METADATA_UNAVAILABLE for item in remote
    )
    if remote_changed and remote_unavailable:
        indicator = (
            "Remote Python code changed; additional remote-code metadata was "
            "unavailable."
        )
    elif remote_changed:
        indicator = "Remote Python code changed."
    elif remote_unavailable:
        indicator = "Remote Python code comparison unavailable."
    else:
        indicator = "No remote Python code changed."
    lines.extend(["", indicator, f"Estimated changed bytes: {result.changed_bytes}"])
    if result.unknown_changed_sizes:
        lines.append(f"Changed files with unknown size: {result.unknown_changed_sizes}")
    return "\n".join(lines)


def _by_path(files: tuple[RepositoryFile, ...]) -> dict[str, RepositoryFile]:
    result: dict[str, RepositoryFile] = {}
    for item in files:
        if item.path in result:
            raise DiffError(f"repository tree contains duplicate path '{item.path}'")
        result[item.path] = item
    return result


def _compare_file(
    old: RepositoryFile, new: RepositoryFile
) -> tuple[ChangeState, IdentityKind | None]:
    for attribute, identity in (
        ("lfs_sha256", IdentityKind.LFS_SHA256),
        ("xet_hash", IdentityKind.XET_HASH),
        ("blob_id", IdentityKind.GIT_BLOB),
    ):
        before, after = getattr(old, attribute), getattr(new, attribute)
        if before is not None and after is not None:
            state = ChangeState.UNCHANGED if before == after else ChangeState.CHANGED
            return state, identity
    if old.size is not None and new.size is not None and old.size != new.size:
        return ChangeState.CHANGED, IdentityKind.SIZE
    return ChangeState.METADATA_UNAVAILABLE, None


def _json_paths(old: object, new: object) -> tuple[str, ...]:
    paths: list[str] = []

    def walk(before: object, after: object, parts: tuple[str, ...]) -> None:
        if len(paths) >= 20 or before == after:
            return
        if isinstance(before, dict) and isinstance(after, dict) and len(parts) < 3:
            for key in sorted(before.keys() | after.keys()):
                if len(paths) >= 20:
                    break
                if key not in before or key not in after:
                    paths.append(".".join((*parts, str(key))))
                else:
                    walk(before[key], after[key], (*parts, str(key)))
        else:
            paths.append(".".join(parts) or "<root>")

    walk(old, new, ())
    return tuple(paths)


def _render_grouped(lines: list[str], changes: list[FileChange]) -> None:
    for category in ArtifactCategory:
        for item in changes:
            if item.category is not category:
                continue
            detail = ""
            if item.semantic_unavailable:
                detail = " (semantic comparison unavailable)"
            elif item.semantic_paths is not None:
                detail = f" (JSON keys: {', '.join(item.semantic_paths) or 'none'})"
            lines.append(f"  {category.value.upper():<13} {item.path}{detail}")
