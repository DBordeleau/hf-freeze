from pathlib import Path

import pytest
from typer.testing import CliRunner

import hf_freeze.cli
from hf_freeze.cli import app
from hf_freeze.hub import HubResolutionError
from hf_freeze.models import RepoType


class FakeResolver:
    calls: list[tuple[str, RepoType, str]]

    def __init__(self) -> None:
        self.calls = []

    def resolve(self, repo_id: str, repo_type: RepoType, revision: str) -> str:
        self.calls.append((repo_id, repo_type, revision))
        return "0123456789abcdef"


def test_lock_cli_writes_stable_lockfile_with_fake_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app.py").write_text(
        'REVISION = "constant-v2"\n'
        'AutoModel.from_pretrained("org/model")\n'
        'snapshot_download("org/model")\n'
        'load_dataset("org/data", revision="literal-v1")\n'
        'AutoModel.from_pretrained("org/constant", revision=REVISION)\n',
        encoding="utf-8",
    )
    resolver = FakeResolver()
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)

    first = CliRunner().invoke(app, ["lock", str(tmp_path)])
    first_bytes = (tmp_path / "hf.lock").read_bytes()
    second = CliRunner().invoke(app, ["lock", str(tmp_path)])

    assert first.exit_code == second.exit_code == 0
    assert first.stdout == second.stdout == f"Wrote {tmp_path / 'hf.lock'}\n"
    assert (tmp_path / "hf.lock").read_bytes() == first_bytes
    assert resolver.calls == [
        ("org/data", RepoType.DATASET, "literal-v1"),
        ("org/constant", RepoType.MODEL, "constant-v2"),
        ("org/model", RepoType.MODEL, "main"),
        ("org/data", RepoType.DATASET, "literal-v1"),
        ("org/constant", RepoType.MODEL, "constant-v2"),
        ("org/model", RepoType.MODEL, "main"),
    ]


@pytest.mark.parametrize(
    "source",
    [
        "AutoModel.from_pretrained(get_repo())\n",
        'AutoModel.from_pretrained("org/model", revision="v1")\n'
        'AutoModel.from_pretrained("org/model", revision="v2")\n',
        'hf_hub_download("org/model", repo_type=get_type())\n',
        'AutoModel.from_pretrained("org/model", revision=get_revision())\n',
        'AutoModel.from_pretrained("org/model", revision=os.environ["REV"])\n',
        'REV = "v1"\nREV = "v2"\n'
        'AutoModel.from_pretrained("org/model", revision=REV)\n',
        'AutoModel.from_pretrained("org/model", revision="")\n',
        'AutoModel.from_pretrained("org/model", revision="   ")\n',
        'REV = ""\nAutoModel.from_pretrained("org/model", revision=REV)\n',
        'REV = "   "\nAutoModel.from_pretrained("org/model", revision=REV)\n',
    ],
)
def test_lock_cli_failure_preserves_existing_file_without_network(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app.py").write_text(source, encoding="utf-8")
    destination = tmp_path / "hf.lock"
    existing = b'{\n  "version": 1,\n  "dependencies": []\n}\n'
    destination.write_bytes(existing)
    resolver = FakeResolver()
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)

    result = CliRunner().invoke(app, ["lock", str(tmp_path)])

    assert result.exit_code == 1
    assert "Error:" in result.stderr
    assert destination.read_bytes() == existing
    assert resolver.calls == []


def test_lock_invalid_path_is_an_invocation_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["lock", str(tmp_path / "missing")])

    assert result.exit_code == 2


@pytest.mark.parametrize(
    "existing",
    [
        b'{"version": 2, "dependencies": []}\n',
        b"not json at all\n",
    ],
)
def test_invalid_existing_lock_is_preserved_before_resolution(
    existing: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app.py").write_text(
        'AutoModel.from_pretrained("org/model")\n', encoding="utf-8"
    )
    destination = tmp_path / "hf.lock"
    destination.write_bytes(existing)
    resolver = FakeResolver()
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)

    result = CliRunner().invoke(app, ["lock", str(tmp_path)])

    assert result.exit_code == 1
    assert destination.read_bytes() == existing
    assert resolver.calls == []


def test_lock_resolution_failure_is_an_expected_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app.py").write_text(
        'AutoModel.from_pretrained("org/private")\n', encoding="utf-8"
    )
    resolver = FakeResolver()
    monkeypatch.setattr(
        resolver,
        "resolve",
        lambda *_: (_ for _ in ()).throw(HubResolutionError("not accessible")),
    )
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)

    result = CliRunner().invoke(app, ["lock", str(tmp_path)])

    assert result.exit_code == 1
    assert "not accessible" in result.stderr
    assert not (tmp_path / "hf.lock").exists()
