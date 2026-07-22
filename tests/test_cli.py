from pathlib import Path

import pytest
from typer.testing import CliRunner

import hf_freeze.cli
from hf_freeze.cli import app
from hf_freeze.config import ConfigError
from hf_freeze.diff import RepositoryFile
from hf_freeze.hub import HubContentError, HubResolutionError
from hf_freeze.lockfile import read_lockfile, write_lockfile
from hf_freeze.models import (
    CallKind,
    DependencyKind,
    LockedDependency,
    LockedSource,
    Lockfile,
    RepoType,
)

OLD_SHA = "a" * 40
NEW_SHA = "b" * 40


def locked_dependency(
    *,
    repo_id: str = "org/repo",
    repo_type: RepoType = RepoType.MODEL,
    kind: DependencyKind = DependencyKind.MODEL,
    revision: str = "main",
    sha: str = OLD_SHA,
    path: str = "app.py",
) -> LockedDependency:
    return LockedDependency(
        repo_id,
        repo_type,
        kind,
        revision,
        sha,
        (LockedSource(path, 1, CallKind.FROM_PRETRAINED),),
    )


class UpdateResolver:
    def __init__(self, candidate_sha: str = NEW_SHA) -> None:
        self.candidate_sha = candidate_sha
        self.resolutions: list[tuple[str, RepoType, str]] = []

    def resolve(self, repo_id: str, repo_type: RepoType, revision: str) -> str:
        self.resolutions.append((repo_id, repo_type, revision))
        return self.candidate_sha

    def tree(self, *args: object) -> tuple[RepositoryFile, ...]:
        identity = "old" if args[2] == OLD_SHA else "new"
        return (RepositoryFile("config.json", 10, blob_id=identity),)

    def read_small_file(self, *args: object) -> str:
        return '{"value": 1}' if args[2] == OLD_SHA else '{"value": 2}'


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout == "hf-freeze 0.1.0\n"


def test_source_diff_retries_with_utf8_after_legacy_console_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: list[tuple[str, bool]] = []
    encodings: list[str] = []

    def echo(value: str, *, nl: bool) -> None:
        if not rendered:
            rendered.append((value, nl))
            raise UnicodeEncodeError("charmap", value, 0, 1, "unsupported")
        rendered.append((value, nl))

    class LegacyStdout:
        def reconfigure(self, *, encoding: str) -> None:
            encodings.append(encoding)

    monkeypatch.setattr(hf_freeze.cli.typer, "echo", echo)
    monkeypatch.setattr(hf_freeze.cli.sys, "stdout", LegacyStdout())

    hf_freeze.cli._echo_source_diff("+ print('✅')\n")

    assert encodings == ["utf-8"]
    assert rendered == [("+ print('✅')\n", False)] * 2


@pytest.mark.parametrize(
    "arguments",
    [
        ["scan"],
        ["lock"],
        ["check", "--frozen"],
        ["pin"],
        ["diff", "org/repo"],
        ["update", "org/repo"],
    ],
)
def test_commands_validate_shared_project_context_before_work(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = Path.cwd() / "pyproject.toml"
    monkeypatch.setattr(
        "hf_freeze.cli.resolve_project_context",
        lambda _path: (_ for _ in ()).throw(
            ConfigError(f"invalid configuration {config_path}: test failure")
        ),
    )
    monkeypatch.setattr(
        "hf_freeze.cli.scan_path",
        lambda *_: pytest.fail("scanning started before configuration validation"),
    )
    monkeypatch.setattr(
        "hf_freeze.cli.HfHubResolver",
        lambda: pytest.fail("network access started before configuration validation"),
    )
    monkeypatch.setattr(
        "hf_freeze.cli.read_lockfile",
        lambda *_: pytest.fail(
            "lockfile access started before configuration validation"
        ),
    )

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 1
    assert str(config_path) in result.stderr
    assert "test failure" in result.stderr


def test_diff_command_uses_candidate_tree_and_semantic_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_sha, new_sha = "a" * 40, "b" * 40
    locked = LockedDependency(
        "org/repo",
        RepoType.MODEL,
        DependencyKind.MODEL,
        "main",
        old_sha,
        (LockedSource("app.py", 1, CallKind.FROM_PRETRAINED),),
    )
    write_lockfile(tmp_path / "hf.lock", Lockfile(1, (locked,)))

    class FakeResolver:
        candidate_sha = new_sha
        content_available = True

        def resolve(self, *args: object) -> str:
            assert args == ("org/repo", RepoType.MODEL, "v2")
            return self.candidate_sha

        def tree(self, *args: object) -> tuple[RepositoryFile, ...]:
            assert self.candidate_sha != old_sha, "tree must be skipped for same SHA"
            sha = args[2]
            identity = "old" if sha == old_sha else "new"
            return (RepositoryFile("config.json", 10, blob_id=identity),)

        def read_small_file(self, *args: object) -> str:
            if not self.content_available:
                raise HubContentError("unavailable")
            return '{"value": 1}' if args[2] == old_sha else '{"value": 2}'

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hf_freeze.cli.HfHubResolver", FakeResolver)

    result = CliRunner().invoke(app, ["diff", "org/repo", "--revision", "v2"])

    assert result.exit_code == 0
    assert f"{old_sha} -> {new_sha}" in result.stdout
    assert "CONFIG        config.json (JSON keys: value)" in result.stdout
    assert "No remote Python code changed." in result.stdout

    FakeResolver.content_available = False
    unavailable = CliRunner().invoke(app, ["diff", "org/repo", "--revision", "v2"])
    assert unavailable.exit_code == 0
    assert "semantic comparison unavailable" in unavailable.stdout

    FakeResolver.candidate_sha = old_sha
    no_change = CliRunner().invoke(app, ["diff", "org/repo", "--revision", "v2"])
    assert no_change.exit_code == 0
    assert "candidate resolves to the locked commit" in no_change.stdout


def test_diff_command_missing_lock_is_expected_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["diff", "org/repo"])

    assert result.exit_code == 1
    assert "Error:" in result.stderr


def test_diff_and_update_use_configured_root_lock_from_nested_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.hf-freeze]\n", encoding="utf-8")
    write_lockfile(tmp_path / "hf.lock", Lockfile(1, (locked_dependency(),)))
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    resolver = UpdateResolver(OLD_SHA)
    monkeypatch.setattr("hf_freeze.cli.HfHubResolver", lambda: resolver)

    diff = CliRunner().invoke(app, ["diff", "org/repo"])
    update = CliRunner().invoke(app, ["update", "org/repo"])

    assert diff.exit_code == update.exit_code == 0
    assert "hf.lock is already current" in update.stdout


def test_update_dry_run_and_write_share_diff_and_update_all_compatible_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "app.py"
    source.write_text("unchanged = True\n", encoding="utf-8")
    model = locked_dependency()
    snapshot = locked_dependency(kind=DependencyKind.SNAPSHOT, path="download.py")
    unrelated = locked_dependency(
        repo_id="org/data",
        repo_type=RepoType.DATASET,
        kind=DependencyKind.DATASET,
        sha="c" * 40,
        path="data.py",
    )
    destination = tmp_path / "hf.lock"
    write_lockfile(destination, Lockfile(1, (snapshot, unrelated, model)))
    original_lock = destination.read_bytes()
    original_source = source.read_bytes()
    resolver = UpdateResolver()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hf_freeze.cli.HfHubResolver", lambda: resolver)

    diff = CliRunner().invoke(app, ["diff", "org/repo"])
    dry_run = CliRunner().invoke(app, ["update", "org/repo"])

    assert diff.exit_code == dry_run.exit_code == 0
    assert diff.stdout.strip() in dry_run.stdout
    assert "requested_revision: main -> main" in dry_run.stdout
    assert f"sha: {OLD_SHA} -> {NEW_SHA}" in dry_run.stdout
    assert "Dry run; pass --write" in dry_run.stdout
    assert destination.read_bytes() == original_lock
    assert source.read_bytes() == original_source
    assert resolver.resolutions == [
        ("org/repo", RepoType.MODEL, "main"),
        ("org/repo", RepoType.MODEL, "main"),
    ]

    written = CliRunner().invoke(app, ["update", "org/repo", "--write"])

    assert written.exit_code == 0
    updated = read_lockfile(destination)
    selected = [item for item in updated.dependencies if item.repo_id == "org/repo"]
    assert [(item.kind, item.sha) for item in selected] == [
        (DependencyKind.MODEL, NEW_SHA),
        (DependencyKind.SNAPSHOT, NEW_SHA),
    ]
    assert all(item.requested_revision == "main" for item in selected)
    assert selected[0].sources == model.sources
    assert selected[1].sources == snapshot.sources
    assert (
        next(item for item in updated.dependencies if item.repo_id == "org/data")
        == unrelated
    )
    assert source.read_bytes() == original_source
    assert "hf-freeze pin --write" in written.stdout
    assert "hf-freeze check --frozen" in written.stdout


def test_update_writes_explicit_tracking_change_at_same_sha_then_skips_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "hf.lock"
    write_lockfile(destination, Lockfile(1, (locked_dependency(),)))
    resolver = UpdateResolver(OLD_SHA)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hf_freeze.cli.HfHubResolver", lambda: resolver)

    changed = CliRunner().invoke(
        app, ["update", "org/repo", "--revision", "stable", "--write"]
    )

    assert changed.exit_code == 0
    assert "requested_revision: main -> stable" in changed.stdout
    assert "Wrote hf.lock" in changed.stdout
    assert read_lockfile(destination).dependencies[0].requested_revision == "stable"

    current_bytes = destination.read_bytes()
    monkeypatch.setattr(
        "hf_freeze.cli.write_lockfile",
        lambda *_: pytest.fail("no-effective-change update replaced hf.lock"),
    )
    current = CliRunner().invoke(app, ["update", "org/repo", "--write"])

    assert current.exit_code == 0
    assert "already current; no file was replaced" in current.stdout
    assert destination.read_bytes() == current_bytes
    assert resolver.resolutions[-1] == ("org/repo", RepoType.MODEL, "stable")


@pytest.mark.parametrize(
    "failure", ["missing", "incompatible", "duplicate", "malformed"]
)
def test_update_selection_failures_happen_before_network_and_preserve_bytes(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "hf.lock"
    if failure == "missing":
        write_lockfile(destination, Lockfile(1, ()))
    elif failure == "incompatible":
        write_lockfile(
            destination,
            Lockfile(
                1,
                (
                    locked_dependency(),
                    locked_dependency(kind=DependencyKind.SNAPSHOT, sha="c" * 40),
                ),
            ),
        )
    elif failure == "duplicate":
        write_lockfile(
            destination,
            Lockfile(1, (locked_dependency(), locked_dependency(path="other.py"))),
        )
    else:
        destination.write_bytes(b"{not json\n")
    original = destination.read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "hf_freeze.cli.HfHubResolver",
        lambda: pytest.fail("network resolver constructed before lock validation"),
    )

    result = CliRunner().invoke(app, ["update", "org/repo", "--write"])

    assert result.exit_code == 1
    assert "Error:" in result.stderr
    if failure == "duplicate":
        assert "remove the duplicates before updating" in result.stderr
    assert destination.read_bytes() == original


@pytest.mark.parametrize("failure", ["resolve", "tree"])
def test_update_hub_preview_failures_preserve_lock(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "hf.lock"
    write_lockfile(destination, Lockfile(1, (locked_dependency(),)))
    original = destination.read_bytes()

    class FailingResolver(UpdateResolver):
        def resolve(self, *args: object) -> str:
            if failure == "resolve":
                raise HubResolutionError("repository is not accessible")
            return NEW_SHA

        def tree(self, *args: object) -> tuple[RepositoryFile, ...]:
            raise HubResolutionError("repository tree is unavailable")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hf_freeze.cli.HfHubResolver", FailingResolver)

    result = CliRunner().invoke(app, ["update", "org/repo", "--write"])

    assert result.exit_code == 1
    assert "Error:" in result.stderr
    assert destination.read_bytes() == original


def test_update_atomic_write_failure_preserves_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "hf.lock"
    write_lockfile(destination, Lockfile(1, (locked_dependency(),)))
    original = destination.read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hf_freeze.cli.HfHubResolver", UpdateResolver)
    monkeypatch.setattr(
        "hf_freeze.cli.write_lockfile",
        lambda *_: (_ for _ in ()).throw(OSError("atomic replace failed")),
    )

    result = CliRunner().invoke(app, ["update", "org/repo", "--write"])

    assert result.exit_code == 1
    assert "atomic replace failed" in result.stderr
    assert destination.read_bytes() == original
