from pathlib import Path

import pytest
from typer.testing import CliRunner

import hf_freeze.cli
from hf_freeze.check import IssueSeverity, check_lockfile
from hf_freeze.cli import app
from hf_freeze.lockfile import serialize_lockfile
from hf_freeze.models import (
    CallKind,
    DependencyFinding,
    DependencyKind,
    LockedDependency,
    LockedSource,
    Lockfile,
    RepoType,
    ScanDiagnostic,
    ScanResult,
    SourceLocation,
)

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


def finding(
    *,
    revision: str | None = SHA,
    repo_id: str | None = "org/model",
    repo_type: RepoType | None = RepoType.MODEL,
    line: int = 2,
    trust: bool = False,
    trust_error: str | None = None,
    revision_error: str | None = None,
    call: CallKind = CallKind.FROM_PRETRAINED,
) -> DependencyFinding:
    return DependencyFinding(
        repo_id=repo_id,
        repo_type=repo_type,
        call_kind=call,
        requested_revision=revision,
        source=SourceLocation("app.py", line, 4),
        unresolved_reason="dynamic repository ID" if repo_id is None else None,
        revision_unresolved_reason=revision_error,
        trust_remote_code=trust,
        trust_remote_code_unresolved_reason=trust_error,
    )


def dependency(
    *,
    sha: str = SHA,
    repo_id: str = "org/model",
    kind: DependencyKind = DependencyKind.MODEL,
    call: CallKind = CallKind.FROM_PRETRAINED,
) -> LockedDependency:
    return LockedDependency(
        repo_id=repo_id,
        repo_type=RepoType.MODEL,
        kind=kind,
        requested_revision="main",
        sha=sha,
        sources=(LockedSource("app.py", 2, call),),
    )


def checked(
    findings: tuple[DependencyFinding, ...] = (finding(),),
    dependencies: tuple[LockedDependency, ...] = (dependency(),),
    diagnostics: tuple[ScanDiagnostic, ...] = (),
):
    return check_lockfile(
        ScanResult(findings=findings, diagnostics=diagnostics),
        Lockfile(version=1, dependencies=dependencies),
    )


def test_matching_immutable_source_and_lock_is_clean_including_remote_code() -> None:
    assert checked(findings=(finding(trust=True),)) == ()


@pytest.mark.parametrize(
    ("call", "kind"),
    [
        (CallKind.PIPELINE, DependencyKind.MODEL),
        (CallKind.SENTENCE_TRANSFORMER, DependencyKind.MODEL),
        (CallKind.PEFT_FROM_PRETRAINED, DependencyKind.ADAPTER),
    ],
)
def test_new_call_kinds_have_offline_frozen_coverage(
    call: CallKind, kind: DependencyKind
) -> None:
    assert (
        checked(
            findings=(finding(call=call),),
            dependencies=(dependency(call=call, kind=kind),),
        )
        == ()
    )


@pytest.mark.parametrize(
    ("item", "dependencies", "code", "generic_code"),
    [
        (finding(trust=True), (), "UNPINNED_REMOTE_CODE", "MISSING_LOCK_ENTRY"),
        (
            finding(trust_error="dynamic trust"),
            (),
            "DYNAMIC_TRUST_REMOTE_CODE",
            "MISSING_LOCK_ENTRY",
        ),
        (
            finding(repo_id=None, trust=True),
            (dependency(),),
            "UNPINNED_REMOTE_CODE",
            "UNRESOLVED_DEPENDENCY",
        ),
        (
            finding(trust=True),
            (dependency(), dependency()),
            "UNPINNED_REMOTE_CODE",
            "DUPLICATE_LOCK_IDENTITY",
        ),
    ],
)
def test_remote_code_policy_is_not_masked_by_coverage_failures(
    item: DependencyFinding,
    dependencies: tuple[LockedDependency, ...],
    code: str,
    generic_code: str,
) -> None:
    issues = checked(findings=(item,), dependencies=dependencies)

    assert [issue.code for issue in issues].count(code) == 1
    assert generic_code in {issue.code for issue in issues}
    policy_issue = next(issue for issue in issues if issue.code == code)
    assert policy_issue.source == item.source


@pytest.mark.parametrize(
    ("findings", "dependencies", "diagnostics", "code"),
    [
        (
            (),
            (dependency(),),
            (ScanDiagnostic(SourceLocation("bad.py", 3, 1), "parse error"),),
            "SCAN_DIAGNOSTIC",
        ),
        ((finding(repo_id=None),), (dependency(),), (), "UNRESOLVED_DEPENDENCY"),
        ((finding(repo_type=None),), (dependency(),), (), "UNRESOLVED_DEPENDENCY"),
        (
            (finding(revision=None, revision_error="dynamic revision"),),
            (dependency(),),
            (),
            "UNRESOLVED_DEPENDENCY",
        ),
        ((finding(),), (), (), "MISSING_LOCK_ENTRY"),
        ((finding(),), (dependency(), dependency()), (), "DUPLICATE_LOCK_IDENTITY"),
        (
            (finding(), finding(revision=OTHER_SHA, line=3)),
            (dependency(),),
            (),
            "CONFLICTING_REVISIONS",
        ),
        ((finding(revision=None),), (dependency(),), (), "FLOATING_REVISION"),
        ((finding(revision="main"),), (dependency(),), (), "FLOATING_REVISION"),
        ((finding(revision=OTHER_SHA),), (dependency(),), (), "SHA_MISMATCH"),
        ((finding(),), (dependency(sha="short"),), (), "INVALID_LOCKED_SHA"),
        (
            (finding(revision=None, trust=True),),
            (dependency(),),
            (),
            "UNPINNED_REMOTE_CODE",
        ),
        (
            (finding(trust_error="dynamic trust"),),
            (dependency(),),
            (),
            "DYNAMIC_TRUST_REMOTE_CODE",
        ),
    ],
)
def test_required_failure_families_are_actionable_and_deterministic(
    findings: tuple[DependencyFinding, ...],
    dependencies: tuple[LockedDependency, ...],
    diagnostics: tuple[ScanDiagnostic, ...],
    code: str,
) -> None:
    issues = checked(findings, dependencies, diagnostics)

    assert code in {issue.code for issue in issues}
    assert issues == tuple(
        sorted(
            issues,
            key=lambda issue: (
                issue.source.path if issue.source else "",
                issue.source.line if issue.source else 0,
                issue.source.column if issue.source else 0,
                issue.severity.value,
                issue.code,
                issue.message,
            ),
        )
    )
    assert all(issue.remediation for issue in issues)
    assert all(
        issue.source == SourceLocation("app.py", issue.source.line, 4)
        for issue in issues
        if issue.source is not None and issue.source.path == "app.py"
    )


def test_unused_lock_entries_are_sorted_nonfatal_warnings() -> None:
    issues = checked(
        findings=(),
        dependencies=(dependency(repo_id="org/z"), dependency(repo_id="org/a")),
    )

    assert [issue.severity for issue in issues] == [
        IssueSeverity.WARNING,
        IssueSeverity.WARNING,
    ]
    assert [issue.code for issue in issues] == ["UNUSED_LOCK_ENTRY"] * 2
    assert "org/a" in issues[0].message
    assert "org/z" in issues[1].message


def write_cli_project(tmp_path: Path, source: str, lockfile: Lockfile | None) -> None:
    (tmp_path / "app.py").write_text(source, encoding="utf-8")
    if lockfile is not None:
        (tmp_path / "hf.lock").write_text(
            serialize_lockfile(lockfile), encoding="utf-8"
        )


def test_check_cli_succeeds_offline_and_renders_source_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_cli_project(
        tmp_path,
        f'AutoModel.from_pretrained("org/model", revision="{SHA}")\n',
        Lockfile(version=1, dependencies=(dependency(),)),
    )
    monkeypatch.setattr(
        hf_freeze.cli,
        "HfHubResolver",
        lambda: pytest.fail("frozen check must not create a network resolver"),
    )

    success = CliRunner().invoke(app, ["check", str(tmp_path), "--frozen"])
    (tmp_path / "app.py").write_text(
        'AutoModel.from_pretrained("org/model")\n', encoding="utf-8"
    )
    failure = CliRunner().invoke(app, ["check", str(tmp_path), "--frozen"])

    assert success.exit_code == 0
    assert success.stdout == ""
    assert failure.exit_code == 1
    assert "app.py:1:1  ERROR FLOATING_REVISION" in failure.stdout
    assert "Fix:" in failure.stdout


def test_check_cli_renders_unpinned_remote_code_when_lock_entry_is_missing(
    tmp_path: Path,
) -> None:
    write_cli_project(
        tmp_path,
        f'AutoModel.from_pretrained("org/model", revision="{SHA}", '
        "trust_remote_code=True)\n",
        Lockfile(version=1, dependencies=()),
    )

    result = CliRunner().invoke(app, ["check", str(tmp_path), "--frozen"])

    assert result.exit_code == 1
    assert "app.py:1:1  ERROR UNPINNED_REMOTE_CODE" in result.stdout
    assert "ERROR MISSING_LOCK_ENTRY" in result.stdout


def test_check_cli_warning_only_success_and_deterministic_output(
    tmp_path: Path,
) -> None:
    write_cli_project(
        tmp_path,
        "value = 1\n",
        Lockfile(
            version=1,
            dependencies=(dependency(repo_id="org/z"), dependency(repo_id="org/a")),
        ),
    )

    first = CliRunner().invoke(app, ["check", str(tmp_path), "--frozen"])
    second = CliRunner().invoke(app, ["check", str(tmp_path), "--frozen"])

    assert first.exit_code == second.exit_code == 0
    assert first.stdout == second.stdout
    assert first.stdout.index("org/a") < first.stdout.index("org/z")
    assert first.stdout.count("WARNING UNUSED_LOCK_ENTRY") == 2


@pytest.mark.parametrize(
    ("contents", "code"),
    [
        (None, "MISSING_LOCKFILE"),
        ("not json", "MALFORMED_LOCKFILE"),
        ('{"version": 2, "dependencies": []}', "UNSUPPORTED_LOCK_VERSION"),
    ],
)
def test_check_cli_lockfile_failures_exit_one(
    contents: str | None, code: str, tmp_path: Path
) -> None:
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    if contents is not None:
        (tmp_path / "hf.lock").write_text(contents, encoding="utf-8")

    result = CliRunner().invoke(app, ["check", str(tmp_path), "--frozen"])

    assert result.exit_code == 1
    assert code in result.stdout
    assert "Fix:" in result.stdout


@pytest.mark.parametrize(
    ("contents", "lock_code"),
    [
        (None, "MISSING_LOCKFILE"),
        ("not json", "MALFORMED_LOCKFILE"),
        ('{"version": 2, "dependencies": []}', "UNSUPPORTED_LOCK_VERSION"),
        ("read error", "LOCKFILE_READ_ERROR"),
    ],
)
@pytest.mark.parametrize(
    ("trust_value", "policy_code"),
    [
        ("True", "UNPINNED_REMOTE_CODE"),
        ("get_policy()", "DYNAMIC_TRUST_REMOTE_CODE"),
    ],
)
def test_unusable_lock_does_not_mask_remote_code_policy(
    contents: str | None,
    lock_code: str,
    trust_value: str,
    policy_code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "app.py").write_text(
        f'AutoModel.from_pretrained("org/model", revision="{SHA}", '
        f"trust_remote_code={trust_value})\n",
        encoding="utf-8",
    )
    if contents is not None:
        (tmp_path / "hf.lock").write_text(contents, encoding="utf-8")
    if lock_code == "LOCKFILE_READ_ERROR":
        monkeypatch.setattr(
            hf_freeze.cli,
            "read_lockfile",
            lambda *_: (_ for _ in ()).throw(OSError("cannot read lock")),
        )

    result = CliRunner().invoke(app, ["check", str(tmp_path), "--frozen"])

    assert result.exit_code == 1
    assert lock_code in result.stdout
    assert result.stdout.count(f"app.py:1:1  ERROR {policy_code}") == 1
    assert "MISSING_LOCK_ENTRY" not in result.stdout
    assert "FLOATING_REVISION" not in result.stdout
    assert "SHA_MISMATCH" not in result.stdout


def test_check_cli_requires_frozen_and_valid_path(tmp_path: Path) -> None:
    without_frozen = CliRunner().invoke(app, ["check", str(tmp_path)])
    missing_path = CliRunner().invoke(
        app, ["check", str(tmp_path / "missing"), "--frozen"]
    )

    assert without_frozen.exit_code == 2
    assert missing_path.exit_code == 2
