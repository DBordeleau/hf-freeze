import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from hf_freeze.lockfile import (
    LockfileFormatError,
    LockValidationError,
    lockfile_to_dict,
    parse_lockfile,
    resolve_lockfile,
    serialize_lockfile,
    write_lockfile,
)
from hf_freeze.models import (
    CallKind,
    DependencyFinding,
    RepoType,
    ScanResult,
    SourceLocation,
)


class FakeResolver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, RepoType, str]] = []

    def resolve(self, repo_id: str, repo_type: RepoType, revision: str) -> str:
        self.calls.append((repo_id, repo_type, revision))
        return f"sha-{repo_id}-{revision}"


def finding(
    repo_id: str | None,
    call: CallKind,
    *,
    path: str = "src/nested/app.py",
    line: int = 1,
    revision: str | None = None,
    repo_type: RepoType | None = RepoType.MODEL,
) -> DependencyFinding:
    return DependencyFinding(
        repo_id=repo_id,
        repo_type=repo_type,
        call_kind=call,
        requested_revision=revision,
        source=SourceLocation(path, line, 0),
        unresolved_reason="dynamic" if repo_id is None else None,
    )


def test_resolution_merges_sources_and_deduplicates_across_kinds() -> None:
    resolver = FakeResolver()
    result = ScanResult(
        findings=(
            finding("org/repo", CallKind.SNAPSHOT_DOWNLOAD, path="z.py", line=4),
            finding("org/repo", CallKind.FROM_PRETRAINED, path="a.py", line=2),
            finding("org/repo", CallKind.FROM_PRETRAINED, path="a.py", line=1),
            finding("org/data", CallKind.LOAD_DATASET, repo_type=RepoType.DATASET),
        ),
        diagnostics=(),
    )

    lockfile = resolve_lockfile(result, resolver)

    assert [(item.repo_id, item.kind.value) for item in lockfile.dependencies] == [
        ("org/data", "dataset"),
        ("org/repo", "model"),
        ("org/repo", "snapshot"),
    ]
    assert [source.line for source in lockfile.dependencies[1].sources] == [1, 2]
    assert resolver.calls.count(("org/repo", RepoType.MODEL, "main")) == 1


@pytest.mark.parametrize(
    "findings",
    [
        (finding(None, CallKind.FROM_PRETRAINED),),
        (finding("org/repo", CallKind.HF_HUB_DOWNLOAD, repo_type=None),),
        (
            finding("org/repo", CallKind.FROM_PRETRAINED, revision="v1"),
            finding("org/repo", CallKind.FROM_PRETRAINED, revision="v2"),
        ),
    ],
)
def test_invalid_findings_abort_before_resolution(
    findings: tuple[DependencyFinding, ...],
) -> None:
    resolver = FakeResolver()

    with pytest.raises(LockValidationError):
        resolve_lockfile(ScanResult(findings=findings, diagnostics=()), resolver)

    assert resolver.calls == []


def test_serialization_is_canonical_and_round_trips() -> None:
    lockfile = resolve_lockfile(
        ScanResult(
            findings=(finding("org/model", CallKind.FROM_PRETRAINED),),
            diagnostics=(),
        ),
        FakeResolver(),
    )

    text = serialize_lockfile(lockfile)

    assert text.endswith("\n")
    assert '  "version": 1' in text
    assert text.index('"repo_id"') < text.index('"repo_type"') < text.index('"kind"')
    assert "src/nested/app.py" in text
    assert "\\\\" not in text
    assert parse_lockfile(text) == lockfile
    assert serialize_lockfile(parse_lockfile(text)) == text


@pytest.mark.parametrize(
    "path",
    [
        "C:\\Users\\name\\app.py",
        "C:relative\\app.py",
        "\\\\server\\share\\app.py",
        "/home/name/app.py",
        "../outside.py",
    ],
)
def test_rejects_non_project_relative_source_paths(path: str) -> None:
    lockfile = resolve_lockfile(
        ScanResult(
            findings=(finding("org/model", CallKind.FROM_PRETRAINED),),
            diagnostics=(),
        ),
        FakeResolver(),
    )
    value = lockfile_to_dict(lockfile)
    value["dependencies"][0]["sources"][0]["path"] = path  # type: ignore[index]

    with pytest.raises(LockfileFormatError, match="project-relative POSIX path"):
        parse_lockfile(json.dumps(value))


def test_serialization_rejects_invalid_source_path() -> None:
    lockfile = resolve_lockfile(
        ScanResult(
            findings=(finding("org/model", CallKind.FROM_PRETRAINED),),
            diagnostics=(),
        ),
        FakeResolver(),
    )
    dependency = lockfile.dependencies[0]
    invalid_source = replace(dependency.sources[0], path="../outside.py")
    invalid_lockfile = replace(
        lockfile,
        dependencies=(replace(dependency, sources=(invalid_source,)),),
    )

    with pytest.raises(LockfileFormatError, match="project-relative POSIX path"):
        serialize_lockfile(invalid_lockfile)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not json", "malformed lockfile JSON"),
        ('{"version": 2, "dependencies": []}', "unsupported lockfile version: 2"),
        ('{"version": 1}', "must contain exactly"),
    ],
)
def test_rejects_malformed_or_unsupported_input(text: str, message: str) -> None:
    with pytest.raises(LockfileFormatError, match=message):
        parse_lockfile(text)


def test_atomic_write_replaces_destination_and_cleans_up_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "hf.lock"
    destination.write_text("old\n", encoding="utf-8")
    lockfile = resolve_lockfile(
        ScanResult(
            findings=(finding("org/model", CallKind.FROM_PRETRAINED),),
            diagnostics=(),
        ),
        FakeResolver(),
    )
    real_replace = os.replace
    monkeypatch.setattr(
        "hf_freeze.lockfile.os.replace",
        lambda *_: (_ for _ in ()).throw(OSError("stop")),
    )

    with pytest.raises(OSError, match="stop"):
        write_lockfile(destination, lockfile)
    assert destination.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".hf.lock.*.tmp")) == []

    monkeypatch.setattr("hf_freeze.lockfile.os.replace", real_replace)
    write_lockfile(destination, lockfile)
    assert destination.read_text(encoding="utf-8") == serialize_lockfile(lockfile)
