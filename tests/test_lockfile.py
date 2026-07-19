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
    DependencyKind,
    LockedDependency,
    LockedSource,
    Lockfile,
    RepoType,
    ScanResult,
    SourceLocation,
)

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


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


def locked_dependency(
    *,
    requested_revision: str = "main",
    sha: str = SHA,
    sources: tuple[LockedSource, ...] = (),
) -> LockedDependency:
    return LockedDependency(
        "org/repo",
        RepoType.MODEL,
        DependencyKind.MODEL,
        requested_revision,
        sha,
        sources,
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


def test_new_call_kinds_round_trip_in_schema_v1_with_adapter_classification() -> None:
    result = ScanResult(
        findings=(
            finding("org/pipeline", CallKind.PIPELINE),
            finding("org/sentence", CallKind.SENTENCE_TRANSFORMER),
            finding("org/adapter", CallKind.PEFT_FROM_PRETRAINED),
        ),
        diagnostics=(),
    )

    lockfile = resolve_lockfile(result, FakeResolver())
    reparsed = parse_lockfile(serialize_lockfile(lockfile))

    assert reparsed == lockfile
    assert [(item.repo_type, item.kind) for item in lockfile.dependencies] == [
        (RepoType.MODEL, DependencyKind.ADAPTER),
        (RepoType.MODEL, DependencyKind.MODEL),
        (RepoType.MODEL, DependencyKind.MODEL),
    ]
    calls = sorted(
        source.call for item in reparsed.dependencies for source in item.sources
    )
    assert calls == [
        CallKind.PEFT_FROM_PRETRAINED,
        CallKind.PIPELINE,
        CallKind.SENTENCE_TRANSFORMER,
    ]


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


def test_matching_pinned_sha_preserves_tracking_revision_and_rebuilds_sources() -> None:
    existing = Lockfile(
        1,
        (
            locked_dependency(
                sources=(LockedSource("old.py", 20, CallKind.FROM_PRETRAINED),),
            ),
        ),
    )
    result = ScanResult(
        findings=(
            finding(
                "org/repo",
                CallKind.FROM_PRETRAINED,
                path="new.py",
                line=2,
                revision=SHA,
            ),
            finding(
                "org/repo",
                CallKind.PIPELINE,
                path="new.py",
                line=8,
                revision=SHA,
            ),
        ),
        diagnostics=(),
    )
    resolver = FakeResolver()

    lockfile = resolve_lockfile(result, resolver, existing_lockfile=existing)

    assert lockfile.dependencies[0].requested_revision == "main"
    assert lockfile.dependencies[0].sha == SHA
    assert lockfile.dependencies[0].sources == (
        LockedSource("new.py", 2, CallKind.FROM_PRETRAINED),
        LockedSource("new.py", 8, CallKind.PIPELINE),
    )
    assert resolver.calls == []


def test_different_pinned_sha_fails_before_resolution() -> None:
    existing = Lockfile(1, (locked_dependency(),))
    resolver = FakeResolver()

    with pytest.raises(
        LockValidationError,
        match=rf"org/repo.*{OTHER_SHA}.*{SHA}",
    ):
        resolve_lockfile(
            ScanResult(
                findings=(
                    finding(
                        "org/repo",
                        CallKind.FROM_PRETRAINED,
                        revision=OTHER_SHA,
                    ),
                ),
                diagnostics=(),
            ),
            resolver,
            existing_lockfile=existing,
        )

    assert resolver.calls == []


def test_duplicate_existing_identity_fails_instead_of_choosing_an_entry() -> None:
    existing = Lockfile(
        1,
        (locked_dependency(), locked_dependency(requested_revision="v1")),
    )
    resolver = FakeResolver()

    with pytest.raises(LockValidationError, match="multiple entries.*org/repo"):
        resolve_lockfile(
            ScanResult(
                findings=(
                    finding(
                        "org/repo",
                        CallKind.FROM_PRETRAINED,
                        revision=SHA,
                    ),
                ),
                diagnostics=(),
            ),
            resolver,
            existing_lockfile=existing,
        )

    assert resolver.calls == []


def test_first_exact_sha_lock_retains_normal_resolution_behavior() -> None:
    resolver = FakeResolver()

    lockfile = resolve_lockfile(
        ScanResult(
            findings=(
                finding(
                    "org/repo",
                    CallKind.FROM_PRETRAINED,
                    revision=SHA,
                ),
            ),
            diagnostics=(),
        ),
        resolver,
    )

    assert resolver.calls == [("org/repo", RepoType.MODEL, SHA)]
    assert lockfile.dependencies[0].requested_revision == SHA
    assert lockfile.dependencies[0].sha == f"sha-org/repo-{SHA}"


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
