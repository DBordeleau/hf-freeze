from pathlib import Path

import pytest
from typer.testing import CliRunner

import hf_freeze.cli
from hf_freeze.cli import app
from hf_freeze.diff import RepositoryFile
from hf_freeze.lockfile import read_lockfile
from hf_freeze.models import RepoType

INITIAL_SHA = "a" * 40
UPDATED_SHA = "b" * 40


class LifecycleResolver:
    def __init__(self) -> None:
        self.candidate_sha = INITIAL_SHA
        self.resolutions: list[tuple[str, RepoType, str]] = []

    def resolve(self, repo_id: str, repo_type: RepoType, revision: str) -> str:
        self.resolutions.append((repo_id, repo_type, revision))
        return self.candidate_sha

    def tree(
        self, _repo_id: str, _repo_type: RepoType, revision: str
    ) -> tuple[RepositoryFile, ...]:
        value = "initial" if revision == INITIAL_SHA else "updated"
        return (RepositoryFile("config.json", 20, blob_id=value),)

    def read_small_file(
        self,
        _repo_id: str,
        _repo_type: RepoType,
        revision: str,
        _path: str,
        _expected_size: int | None,
    ) -> str:
        value = "initial" if revision == INITIAL_SHA else "updated"
        return f'{{"revision": "{value}"}}'


def test_complete_cli_lifecycle_from_tracking_lock_to_reviewed_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the supported lifecycle without network, model imports, or weights."""

    source = tmp_path / "app.py"
    source.write_text(
        'AutoModel.from_pretrained("org/model", revision="main")\n', encoding="utf-8"
    )
    resolver = LifecycleResolver()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)
    runner = CliRunner()

    scan = runner.invoke(app, ["scan"])
    initial_lock = runner.invoke(app, ["lock"])
    initial_lock_bytes = (tmp_path / "hf.lock").read_bytes()
    initial_pin = runner.invoke(app, ["pin", "--write"])
    initial_check = runner.invoke(app, ["check", "--frozen"])

    assert scan.exit_code == initial_lock.exit_code == initial_pin.exit_code == 0
    assert initial_check.exit_code == 0
    assert "org/model  revision=main" in scan.stdout
    assert INITIAL_SHA in source.read_text(encoding="utf-8")

    resolver.candidate_sha = UPDATED_SHA
    source_after_initial_pin = source.read_bytes()
    diff = runner.invoke(app, ["diff", "org/model"])
    dry_run = runner.invoke(app, ["update", "org/model"])

    assert diff.exit_code == dry_run.exit_code == 0
    assert f"{INITIAL_SHA} -> {UPDATED_SHA}" in diff.stdout
    assert "Dry run; pass --write to update hf.lock." in dry_run.stdout
    assert (tmp_path / "hf.lock").read_bytes() == initial_lock_bytes
    assert source.read_bytes() == source_after_initial_pin

    accepted = runner.invoke(app, ["update", "org/model", "--write"])

    assert accepted.exit_code == 0
    assert (tmp_path / "hf.lock").read_bytes() != initial_lock_bytes
    assert source.read_bytes() == source_after_initial_pin

    final_pin = runner.invoke(app, ["pin", "--write"])
    final_check = runner.invoke(app, ["check", "--frozen"])
    final_lock = read_lockfile(tmp_path / "hf.lock")

    assert final_pin.exit_code == final_check.exit_code == 0
    assert UPDATED_SHA in source.read_text(encoding="utf-8")
    assert [
        (item.requested_revision, item.sha) for item in final_lock.dependencies
    ] == [("main", UPDATED_SHA)]
    assert resolver.resolutions == [
        ("org/model", RepoType.MODEL, "main"),
        ("org/model", RepoType.MODEL, "main"),
        ("org/model", RepoType.MODEL, "main"),
        ("org/model", RepoType.MODEL, "main"),
    ]
