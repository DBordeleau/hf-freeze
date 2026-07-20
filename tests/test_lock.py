from pathlib import Path

import pytest
from typer.testing import CliRunner

import hf_freeze.cli
from hf_freeze.cli import app
from hf_freeze.hub import HubResolutionError
from hf_freeze.lockfile import read_lockfile, write_lockfile
from hf_freeze.models import (
    CallKind,
    DependencyKind,
    LockedDependency,
    LockedSource,
    Lockfile,
    RepoType,
)

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"
SECOND_SHA = "1111111111111111111111111111111111111111"


class FakeResolver:
    calls: list[tuple[str, RepoType, str]]

    def __init__(self, shas: dict[str, str] | None = None) -> None:
        self.calls = []
        self.shas = shas or {}

    def resolve(self, repo_id: str, repo_type: RepoType, revision: str) -> str:
        self.calls.append((repo_id, repo_type, revision))
        return self.shas.get(repo_id, SHA)


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


def test_lock_pin_relock_lifecycle_preserves_tracking_and_adds_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "app.py"
    source.write_text(
        'AutoModel.from_pretrained("org/first", revision="main")\n',
        encoding="utf-8",
    )
    resolver = FakeResolver({"org/first": SHA, "org/second": SECOND_SHA})
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)
    runner = CliRunner()

    first = runner.invoke(app, ["lock", str(tmp_path)])
    first_lock = (tmp_path / "hf.lock").read_bytes()
    pinned = runner.invoke(app, ["pin", str(tmp_path), "--write"])
    second = runner.invoke(app, ["lock", str(tmp_path)])

    assert first.exit_code == pinned.exit_code == second.exit_code == 0
    assert SHA in source.read_text(encoding="utf-8")
    assert (tmp_path / "hf.lock").read_bytes() == first_lock
    assert resolver.calls == [("org/first", RepoType.MODEL, "main")]

    with source.open("a", encoding="utf-8", newline="") as stream:
        stream.write('AutoModel.from_pretrained("org/second")\n')
    third = runner.invoke(app, ["lock", str(tmp_path)])
    lockfile = read_lockfile(tmp_path / "hf.lock")

    assert third.exit_code == 0
    assert [item.repo_id for item in lockfile.dependencies] == [
        "org/first",
        "org/second",
    ]
    assert (
        lockfile.dependencies[0].requested_revision,
        lockfile.dependencies[0].sha,
    ) == ("main", SHA)
    assert resolver.calls == [
        ("org/first", RepoType.MODEL, "main"),
        ("org/second", RepoType.MODEL, "main"),
    ]


def test_annotation_bound_complete_lifecycle_preserves_dynamic_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.hf-freeze.dependencies.primary-model]\n"
        'repo_id = "org/model"\n'
        'repo_type = "model"\n'
        'revision = "stable"\n',
        encoding="utf-8",
    )
    source = tmp_path / "app.py"
    source.write_text(
        "# keep this comment\n"
        "# hf-freeze: dependency=primary-model\n"
        "model = AutoModel.from_pretrained(settings.model)\n",
        encoding="utf-8",
    )
    resolver = FakeResolver({"org/model": SHA})
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)
    runner = CliRunner()

    locked = runner.invoke(app, ["lock", str(tmp_path)])
    pinned = runner.invoke(app, ["pin", str(tmp_path), "--write"])
    relocked = runner.invoke(app, ["lock", str(tmp_path)])
    checked = runner.invoke(app, ["check", str(tmp_path), "--frozen"])
    monkeypatch.chdir(tmp_path)
    diffed = runner.invoke(app, ["diff", "org/model"], catch_exceptions=False)
    updated = runner.invoke(app, ["update", "org/model"], catch_exceptions=False)

    assert [item.exit_code for item in (locked, pinned, relocked, checked)] == [
        0,
        0,
        0,
        0,
    ]
    assert diffed.exit_code == updated.exit_code == 0
    text = source.read_text(encoding="utf-8")
    assert "# keep this comment" in text
    assert "# hf-freeze: dependency=primary-model" in text
    assert "settings.model" in text
    assert f'revision="{SHA}"' in text
    dependency = read_lockfile(tmp_path / "hf.lock").dependencies[0]
    assert (dependency.requested_revision, dependency.sha) == ("stable", SHA)
    assert resolver.calls == [
        ("org/model", RepoType.MODEL, "stable"),
        ("org/model", RepoType.MODEL, "stable"),
        ("org/model", RepoType.MODEL, "stable"),
    ]


def test_environment_bound_complete_lifecycle_is_ambient_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.hf-freeze.dependencies.primary-model]\n"
        'repo_id = "org/model"\n'
        'repo_type = "model"\n'
        'revision = "stable"\n\n'
        "[tool.hf-freeze.bindings.environment]\n"
        'MODEL_ID = "primary-model"\n',
        encoding="utf-8",
    )
    source = tmp_path / "app.py"
    source.write_text(
        "# repository selection stays dynamic\n"
        'model_id = os.getenv("MODEL_ID", "org/fallback")\n'
        "model = AutoModel.from_pretrained(\n"
        "    model_id,  # preserve assignment and formatting\n"
        ")\n",
        encoding="utf-8",
    )
    resolver = FakeResolver({"org/model": SHA})
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)
    runner = CliRunner()

    monkeypatch.delenv("MODEL_ID", raising=False)
    absent_scan = runner.invoke(app, ["scan", str(tmp_path)])
    monkeypatch.setenv("MODEL_ID", "other/runtime-repository")
    conflicting_scan = runner.invoke(app, ["scan", str(tmp_path)])
    locked = runner.invoke(app, ["lock", str(tmp_path)])
    pinned = runner.invoke(app, ["pin", str(tmp_path), "--write"])
    relocked = runner.invoke(app, ["lock", str(tmp_path)])
    conflicting_check = runner.invoke(app, ["check", str(tmp_path), "--frozen"])
    monkeypatch.delenv("MODEL_ID", raising=False)
    absent_check = runner.invoke(app, ["check", str(tmp_path), "--frozen"])
    monkeypatch.chdir(tmp_path)
    diffed = runner.invoke(app, ["diff", "org/model"], catch_exceptions=False)
    updated = runner.invoke(app, ["update", "org/model"], catch_exceptions=False)

    assert absent_scan.exit_code == conflicting_scan.exit_code == 0
    assert absent_scan.stdout == conflicting_scan.stdout
    assert [item.exit_code for item in (locked, pinned, relocked)] == [0, 0, 0]
    assert conflicting_check.exit_code == absent_check.exit_code == 0
    assert conflicting_check.stdout == absent_check.stdout
    assert diffed.exit_code == updated.exit_code == 0
    text = source.read_text(encoding="utf-8")
    assert "# repository selection stays dynamic" in text
    assert 'model_id = os.getenv("MODEL_ID", "org/fallback")' in text
    assert "model_id,  # preserve assignment and formatting" in text
    assert f'revision="{SHA}"' in text
    dependency = read_lockfile(tmp_path / "hf.lock").dependencies[0]
    assert (dependency.requested_revision, dependency.sha) == ("stable", SHA)
    assert resolver.calls == [
        ("org/model", RepoType.MODEL, "stable"),
        ("org/model", RepoType.MODEL, "stable"),
        ("org/model", RepoType.MODEL, "stable"),
    ]


def test_environment_bound_lock_is_ambient_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = (
        "[tool.hf-freeze.dependencies.primary-model]\n"
        'repo_id = "org/model"\nrepo_type = "model"\nrevision = "stable"\n\n'
        "[tool.hf-freeze.bindings.environment]\n"
        'MODEL_ID = "primary-model"\n'
    )
    source = 'AutoModel.from_pretrained(os.getenv("MODEL_ID"))\n'
    absent_project = tmp_path / "absent"
    conflicting_project = tmp_path / "conflicting"
    for project in (absent_project, conflicting_project):
        project.mkdir()
        (project / "pyproject.toml").write_text(configuration, encoding="utf-8")
        (project / "app.py").write_text(source, encoding="utf-8")

    absent_resolver = FakeResolver({"org/model": SHA})
    conflicting_resolver = FakeResolver({"org/model": SHA})
    resolvers = iter((absent_resolver, conflicting_resolver))
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: next(resolvers))
    runner = CliRunner()

    monkeypatch.delenv("MODEL_ID", raising=False)
    absent = runner.invoke(app, ["lock", str(absent_project)])
    monkeypatch.setenv("MODEL_ID", "other/runtime-repository")
    conflicting = runner.invoke(app, ["lock", str(conflicting_project)])

    assert absent.exit_code == conflicting.exit_code == 0
    assert (absent_project / "hf.lock").read_bytes() == (
        conflicting_project / "hf.lock"
    ).read_bytes()
    assert (
        absent_resolver.calls
        == conflicting_resolver.calls
        == [("org/model", RepoType.MODEL, "stable")]
    )


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("missing", "MISSING_ENVIRONMENT_BINDING"),
        ("ambiguous", "AMBIGUOUS_ENVIRONMENT_REFERENCE"),
        ("incompatible", "ENVIRONMENT_BINDING_CONFLICT"),
        ("directive", "BINDING_DIRECTIVE_CONFLICT"),
    ],
)
def test_environment_binding_failures_prevent_hub_lock_and_source_writes(
    failure: str,
    expected_code: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_type = "dataset" if failure == "incompatible" else "model"
    configuration = (
        "[tool.hf-freeze.dependencies.primary-model]\n"
        f'repo_id = "org/model"\nrepo_type = "{repo_type}"\nrevision = "main"\n'
    )
    if failure != "missing":
        configuration += (
            '\n[tool.hf-freeze.bindings.environment]\nMODEL_ID = "primary-model"\n'
        )
    source = 'AutoModel.from_pretrained(os.getenv("UNBOUND", "org/fallback"))\n'
    if failure == "ambiguous":
        source = (
            'model_id = os.getenv("MODEL_ID")\nmodel_id = choose()\n'
            "AutoModel.from_pretrained(model_id)\n"
        )
    elif failure in {"incompatible", "directive"}:
        source = 'AutoModel.from_pretrained(os.getenv("MODEL_ID"))\n'
    if failure == "directive":
        configuration = configuration.replace(
            "[tool.hf-freeze.bindings.environment]",
            "[tool.hf-freeze.dependencies.other-model]\n"
            'repo_id = "org/other"\nrepo_type = "model"\nrevision = "main"\n\n'
            "[tool.hf-freeze.bindings.environment]",
        )
        source = "# hf-freeze: dependency=other-model\n" + source
    (tmp_path / "pyproject.toml").write_text(configuration, encoding="utf-8")
    source_path = tmp_path / "app.py"
    source_path.write_text(
        'AutoModel.from_pretrained("org/static")\n' + source, encoding="utf-8"
    )
    destination = tmp_path / "hf.lock"
    write_lockfile(
        destination,
        Lockfile(
            1,
            (
                LockedDependency(
                    "org/static",
                    RepoType.MODEL,
                    DependencyKind.MODEL,
                    "main",
                    SHA,
                    (LockedSource("app.py", 1, CallKind.FROM_PRETRAINED),),
                ),
            ),
        ),
    )
    original_lock = destination.read_bytes()
    original_source = source_path.read_bytes()
    resolver = FakeResolver()
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)
    runner = CliRunner()

    locked = runner.invoke(app, ["lock", str(tmp_path)])
    pinned = runner.invoke(app, ["pin", str(tmp_path), "--write"])

    assert locked.exit_code == pinned.exit_code == 1
    assert expected_code in pinned.stderr
    if failure == "missing":
        assert "literal fallback 'org/fallback' is not authoritative" in pinned.stderr
    assert resolver.calls == []
    assert destination.read_bytes() == original_lock
    assert source_path.read_bytes() == original_source


def test_unused_declaration_warns_without_hub_resolution_or_lock_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.hf-freeze.dependencies.unused]\n"
        'repo_id = "org/model"\n'
        'repo_type = "model"\n'
        'revision = "main"\n',
        encoding="utf-8",
    )
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    resolver = FakeResolver()
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)

    result = CliRunner().invoke(app, ["lock", str(tmp_path)])

    assert result.exit_code == 0
    assert "WARNING UNUSED_DECLARATION" in result.stderr
    assert read_lockfile(tmp_path / "hf.lock").dependencies == ()
    assert resolver.calls == []


def test_pin_fatal_directive_diagnostic_prevents_all_source_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text('AutoModel.from_pretrained("org/a")\n', encoding="utf-8")
    second.write_text('AutoModel.from_pretrained("org/b")\n', encoding="utf-8")
    resolver = FakeResolver({"org/a": SHA, "org/b": SECOND_SHA})
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)
    runner = CliRunner()
    assert runner.invoke(app, ["lock", str(tmp_path)]).exit_code == 0
    before = first.read_bytes()
    second.write_text(
        '# hf-freeze dependency=broken\nAutoModel.from_pretrained("org/b")\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["pin", str(tmp_path), "--write"])

    assert result.exit_code == 1
    assert "ERROR MALFORMED_DIRECTIVE" in result.stderr
    assert first.read_bytes() == before
    assert SHA not in first.read_text(encoding="utf-8")


def test_lock_pinned_sha_mismatch_is_actionable_and_preserves_lockfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "app.py"
    source.write_text(
        'AutoModel.from_pretrained("org/model", revision="main")\n',
        encoding="utf-8",
    )
    resolver = FakeResolver({"org/model": SHA})
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)
    runner = CliRunner()

    assert runner.invoke(app, ["lock", str(tmp_path)]).exit_code == 0
    assert runner.invoke(app, ["pin", str(tmp_path), "--write"]).exit_code == 0
    destination = tmp_path / "hf.lock"
    existing = destination.read_bytes()
    source.write_text(
        source.read_text(encoding="utf-8").replace(SHA, OTHER_SHA),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["lock", str(tmp_path)])

    assert result.exit_code == 1
    assert "org/model" in result.stderr
    assert SHA in result.stderr and OTHER_SHA in result.stderr
    assert destination.read_bytes() == existing
    assert resolver.calls == [("org/model", RepoType.MODEL, "main")]


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
        '# hf-freeze dependency=broken\nAutoModel.from_pretrained("org/model")\n',
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


def test_configured_omitted_path_lifecycle_uses_root_scope_and_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.hf-freeze]\nexclude = ["tests/**", "examples/**"]\n',
        encoding="utf-8",
    )
    source = tmp_path / "src" / "app.py"
    source.parent.mkdir(parents=True)
    source.write_text('AutoModel.from_pretrained("org/model")\n', encoding="utf-8")
    excluded = tmp_path / "tests" / "test_dynamic.py"
    excluded.parent.mkdir()
    excluded.write_text("AutoModel.from_pretrained(get_model())\n", encoding="utf-8")
    nested = tmp_path / "src" / "package"
    nested.mkdir()
    monkeypatch.chdir(nested)
    resolver = FakeResolver()
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", lambda: resolver)
    runner = CliRunner()

    locked = runner.invoke(app, ["lock"])
    pinned = runner.invoke(app, ["pin", "--write"])
    checked = runner.invoke(app, ["check", "--frozen"])

    assert locked.exit_code == pinned.exit_code == checked.exit_code == 0
    assert (tmp_path / "hf.lock").is_file()
    assert not (nested / "hf.lock").exists()
    assert read_lockfile(tmp_path / "hf.lock").dependencies[0].sources[0].path == (
        "src/app.py"
    )
    assert SHA in source.read_text(encoding="utf-8")
    assert "get_model()" in excluded.read_text(encoding="utf-8")

    previous_lock = (tmp_path / "hf.lock").read_bytes()
    source.write_text("AutoModel.from_pretrained(get_model())\n", encoding="utf-8")
    blocked = runner.invoke(app, ["lock"])

    assert blocked.exit_code == 1
    assert (tmp_path / "hf.lock").read_bytes() == previous_lock
    assert resolver.calls == [("org/model", RepoType.MODEL, "main")]


def test_configured_explicit_path_narrows_scope_but_writes_root_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.hf-freeze]\n", encoding="utf-8")
    selected = tmp_path / "src" / "app.py"
    selected.parent.mkdir()
    selected.write_text('AutoModel.from_pretrained("org/model")\n', encoding="utf-8")
    (tmp_path / "outside.py").write_text(
        "AutoModel.from_pretrained(get_model())\n", encoding="utf-8"
    )
    monkeypatch.setattr(hf_freeze.cli, "HfHubResolver", FakeResolver)

    result = CliRunner().invoke(app, ["lock", str(selected.parent)])

    assert result.exit_code == 0
    assert (tmp_path / "hf.lock").is_file()
    assert not (selected.parent / "hf.lock").exists()
    assert read_lockfile(tmp_path / "hf.lock").dependencies[0].sources[0].path == (
        "src/app.py"
    )
