from pathlib import Path

import pytest
from typer.testing import CliRunner

from hf_freeze.cli import app
from hf_freeze.diff import RepositoryFile
from hf_freeze.hub import HubContentError
from hf_freeze.lockfile import write_lockfile
from hf_freeze.models import (
    CallKind,
    DependencyKind,
    LockedDependency,
    LockedSource,
    Lockfile,
    RepoType,
)


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout == "hf-freeze 0.1.0\n"


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
