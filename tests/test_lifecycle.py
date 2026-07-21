from pathlib import Path

import pytest
from typer.testing import CliRunner

import hf_freeze.cli
from hf_freeze.cli import app
from hf_freeze.diff import RepositoryFile
from hf_freeze.lockfile import read_lockfile, write_lockfile
from hf_freeze.models import (
    CallKind,
    DependencyKind,
    LockedDependency,
    LockedSource,
    Lockfile,
    RepoType,
)

REPOSITORIES = ("org/static", "org/environment", "org/annotation")
TRACKING_REVISIONS = {
    "org/static": "static-track",
    "org/environment": "environment-track",
    "org/annotation": "annotation-track",
}
INITIAL_SHAS = {
    "org/static": "a" * 40,
    "org/environment": "b" * 40,
    "org/annotation": "c" * 40,
}
UPDATED_SHAS = {
    "org/static": "d" * 40,
    "org/environment": "e" * 40,
    "org/annotation": "f" * 40,
}


class LifecycleResolver:
    def __init__(self) -> None:
        self.candidates = INITIAL_SHAS.copy()
        self.resolutions: list[tuple[str, RepoType, str]] = []
        self.tree_requests: list[tuple[str, str]] = []

    def resolve(self, repo_id: str, repo_type: RepoType, revision: str) -> str:
        self.resolutions.append((repo_id, repo_type, revision))
        return self.candidates[repo_id]

    def tree(
        self, repo_id: str, _repo_type: RepoType, revision: str
    ) -> tuple[RepositoryFile, ...]:
        self.tree_requests.append((repo_id, revision))
        return (RepositoryFile("config.json", 20, blob_id=revision),)

    def read_small_file(
        self,
        repo_id: str,
        _repo_type: RepoType,
        revision: str,
        path: str,
        _expected_size: int | None,
    ) -> str:
        assert repo_id in REPOSITORIES
        assert path == "config.json"
        return f'{{"revision": "{revision}"}}'


def test_all_coverage_categories_complete_review_first_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise only fake metadata; target code, network, and weights stay unused."""

    (tmp_path / "pyproject.toml").write_text(
        """\
[tool.hf-freeze]
include = ["app.py"]

[tool.hf-freeze.dependencies.environment-model]
repo_id = "org/environment"
repo_type = "model"
revision = "environment-track"

[tool.hf-freeze.dependencies.annotation-model]
repo_id = "org/annotation"
repo_type = "model"
revision = "annotation-track"

[tool.hf-freeze.bindings.environment]
MODEL_ID = "environment-model"
""",
        encoding="utf-8",
    )
    source = tmp_path / "app.py"
    acknowledged = (
        b"# hf-freeze: ignore=runtime-user-selected-model\n"
        b"dynamic = AutoModel.from_pretrained(args.model)\n"
    )
    source.write_bytes(
        b"# static literal dependency\n"
        b'static = AutoModel.from_pretrained("org/static", revision="static-track")\n'
        b"# committed environment expression remains authoritative\n"
        b'model_id = os.getenv("MODEL_ID")\n'
        b"environment = AutoModel.from_pretrained(model_id)  # keep env comment\n"
        b"# keep annotation comment\n"
        b"# hf-freeze: dependency=annotation-model\n"
        b"annotation = AutoModel.from_pretrained(settings.model)\n" + acknowledged
    )
    resolver = LifecycleResolver()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)
    runner = CliRunner()

    scanned = runner.invoke(app, ["scan"])
    locked = runner.invoke(app, ["lock"])
    initial_lock_bytes = (tmp_path / "hf.lock").read_bytes()
    initial_pin_preview = runner.invoke(app, ["pin"])
    before_initial_write = source.read_bytes()
    initial_pin_write = runner.invoke(app, ["pin", "--write"])
    initial_check = runner.invoke(app, ["check", "--frozen"])

    assert [
        item.exit_code
        for item in (
            scanned,
            locked,
            initial_pin_preview,
            initial_pin_write,
            initial_check,
        )
    ] == [0, 0, 0, 0, 0]
    for category in (
        "LOCKED_STATIC",
        "LOCKED_ENV_BINDING",
        "LOCKED_ANNOTATION",
        "ACKNOWLEDGED_DYNAMIC",
    ):
        assert f"{category}: 1" in scanned.stdout
        assert f"{category}: 1" in initial_check.stdout
    assert "UNRESOLVED: 0" in scanned.stdout
    assert "Frozen verification: SUCCEEDED." in initial_check.stdout
    assert "outside the frozen guarantee" in initial_check.stdout
    assert "does not imply full runtime reproducibility" in initial_check.stdout
    assert source.read_bytes() == before_initial_write.replace(
        b'revision="static-track"',
        f'revision="{INITIAL_SHAS["org/static"]}"'.encode(),
    ).replace(
        b"model_id)  # keep env comment",
        f'model_id, revision="{INITIAL_SHAS["org/environment"]}")  '
        "# keep env comment".encode(),
    ).replace(
        b"settings.model)",
        f'settings.model, revision="{INITIAL_SHAS["org/annotation"]}")'.encode(),
    )
    assert source.read_bytes().endswith(acknowledged)
    assert all(sha in initial_pin_preview.stdout for sha in INITIAL_SHAS.values())
    initial_lock = read_lockfile(tmp_path / "hf.lock")
    assert {item.repo_id for item in initial_lock.dependencies} == set(REPOSITORIES)
    assert "runtime-user-selected-model" not in initial_lock_bytes.decode()

    resolver.candidates = UPDATED_SHAS.copy()
    source_after_initial_pin = source.read_bytes()
    for repo_id in REPOSITORIES:
        before_preview = (tmp_path / "hf.lock").read_bytes()
        diffed = runner.invoke(app, ["diff", repo_id])
        update_preview = runner.invoke(app, ["update", repo_id])
        assert diffed.exit_code == update_preview.exit_code == 0
        assert f"{INITIAL_SHAS[repo_id]} -> {UPDATED_SHAS[repo_id]}" in diffed.stdout
        assert "Dry run; pass --write to update hf.lock." in update_preview.stdout
        assert (tmp_path / "hf.lock").read_bytes() == before_preview
        assert source.read_bytes() == source_after_initial_pin

        update_write = runner.invoke(app, ["update", repo_id, "--write"])
        assert update_write.exit_code == 0

    updated_lock_bytes = (tmp_path / "hf.lock").read_bytes()
    assert updated_lock_bytes != initial_lock_bytes
    assert source.read_bytes() == source_after_initial_pin
    final_pin_preview = runner.invoke(app, ["pin"])
    assert final_pin_preview.exit_code == 0
    assert source.read_bytes() == source_after_initial_pin
    assert all(sha in final_pin_preview.stdout for sha in UPDATED_SHAS.values())

    final_pin_write = runner.invoke(app, ["pin", "--write"])
    final_check = runner.invoke(app, ["check", "--frozen"])
    final_lock = read_lockfile(tmp_path / "hf.lock")

    assert final_pin_write.exit_code == final_check.exit_code == 0
    assert "Frozen verification: SUCCEEDED." in final_check.stdout
    final_source = source.read_bytes()
    assert final_source.endswith(acknowledged)
    assert b'os.getenv("MODEL_ID")' in final_source
    assert b"model_id, revision=" in final_source
    assert b"# keep env comment" in final_source
    assert b"# hf-freeze: dependency=annotation-model" in final_source
    assert b"settings.model, revision=" in final_source
    assert {item.repo_id: item.sha for item in final_lock.dependencies} == UPDATED_SHAS
    assert {
        item.repo_id: item.requested_revision for item in final_lock.dependencies
    } == TRACKING_REVISIONS
    assert {repo_id for repo_id, _, _ in resolver.resolutions} == set(REPOSITORIES)
    assert resolver.tree_requests


def test_unacknowledged_unresolved_refuses_writes_and_fails_frozen_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "app.py"
    source.write_bytes(b"AutoModel.from_pretrained(select_repository())\n")
    resolver = LifecycleResolver()
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)
    runner = CliRunner()

    scanned = runner.invoke(app, ["scan", str(tmp_path)])
    absent_lock = runner.invoke(app, ["lock", str(tmp_path)])

    assert scanned.exit_code == 0
    assert "coverage=UNRESOLVED" in scanned.stdout
    assert "UNRESOLVED: 1" in scanned.stdout
    assert absent_lock.exit_code == 1
    assert not (tmp_path / "hf.lock").exists()
    assert resolver.resolutions == []

    write_lockfile(
        tmp_path / "hf.lock",
        Lockfile(
            1,
            (
                LockedDependency(
                    "org/static",
                    RepoType.MODEL,
                    DependencyKind.MODEL,
                    "main",
                    INITIAL_SHAS["org/static"],
                    (LockedSource("other.py", 1, CallKind.FROM_PRETRAINED),),
                ),
            ),
        ),
    )
    lock_bytes = (tmp_path / "hf.lock").read_bytes()
    source_bytes = source.read_bytes()

    existing_lock = runner.invoke(app, ["lock", str(tmp_path)])
    checked = runner.invoke(app, ["check", str(tmp_path), "--frozen"])

    assert existing_lock.exit_code == checked.exit_code == 1
    assert "Frozen verification: FAILED." in checked.stdout
    assert "UNRESOLVED: 1" in checked.stdout
    assert "ERROR UNRESOLVED_DEPENDENCY" in checked.stdout
    assert (tmp_path / "hf.lock").read_bytes() == lock_bytes
    assert source.read_bytes() == source_bytes
    assert resolver.resolutions == []
