import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

import hf_freeze.rewrite
from hf_freeze.cli import app
from hf_freeze.lockfile import write_lockfile
from hf_freeze.models import (
    CallKind,
    DependencyKind,
    LockedDependency,
    LockedSource,
    Lockfile,
    RepoType,
)
from hf_freeze.rewrite import apply_rewrite_plan, plan_rewrites
from hf_freeze.scan import scan_path

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


def dependency(
    path: str = "app.py",
    line: int = 1,
    *,
    sha: str = SHA,
    repo_id: str = "org/model",
    call: CallKind = CallKind.FROM_PRETRAINED,
) -> LockedDependency:
    kinds = {
        CallKind.FROM_PRETRAINED: (RepoType.MODEL, DependencyKind.MODEL),
        CallKind.LOAD_DATASET: (RepoType.DATASET, DependencyKind.DATASET),
        CallKind.HF_HUB_DOWNLOAD: (RepoType.MODEL, DependencyKind.DIRECT_FILE),
        CallKind.SNAPSHOT_DOWNLOAD: (RepoType.MODEL, DependencyKind.SNAPSHOT),
        CallKind.PIPELINE: (RepoType.MODEL, DependencyKind.MODEL),
        CallKind.SENTENCE_TRANSFORMER: (RepoType.MODEL, DependencyKind.MODEL),
        CallKind.PEFT_FROM_PRETRAINED: (RepoType.MODEL, DependencyKind.ADAPTER),
    }
    repo_type, kind = kinds[call]
    return LockedDependency(
        repo_id,
        repo_type,
        kind,
        "main",
        sha,
        (LockedSource(path, line, call),),
    )


def plan_source(
    tmp_path: Path,
    source: str,
    *dependencies: LockedDependency,
) -> tuple[Path, object]:
    path = tmp_path / "app.py"
    path.write_text(source, encoding="utf-8", newline="")
    lockfile = Lockfile(1, dependencies or (dependency(),))
    return path, plan_rewrites(tmp_path, lockfile, scan_path(tmp_path))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            'model = AutoModel.from_pretrained("org/model")\n',
            f'model = AutoModel.from_pretrained("org/model", revision="{SHA}")\n',
        ),
        (
            'model = AutoModel.from_pretrained(\n    "org/model",\n)\n',
            "model = AutoModel.from_pretrained(\n"
            '    "org/model",\n'
            f'    revision="{SHA}",\n'
            ")\n",
        ),
        (
            "model = AutoModel.from_pretrained(\n"
            '    "org/model",  # repository stays documented\n'
            ")\n",
            "model = AutoModel.from_pretrained(\n"
            '    "org/model",  # repository stays documented\n'
            f'    revision="{SHA}",\n'
            ")\n",
        ),
    ],
)
def test_insertion_preserves_complete_source(
    source: str, expected: str, tmp_path: Path
) -> None:
    path, plan = plan_source(tmp_path, source)

    assert not plan.skipped
    assert len(plan.changes) == 1
    assert plan.changes[0].after.decode() == expected
    assert path.read_text(encoding="utf-8") == source


@pytest.mark.parametrize(
    ("call", "source", "expected"),
    [
        (
            CallKind.PIPELINE,
            'pipeline("text-classification", model="org/model")\n',
            f'pipeline("text-classification", model="org/model", revision="{SHA}")\n',
        ),
        (
            CallKind.SENTENCE_TRANSFORMER,
            'SentenceTransformer("org/model", revision="floating")\n',
            f'SentenceTransformer("org/model", revision="{SHA}")\n',
        ),
        (
            CallKind.PEFT_FROM_PRETRAINED,
            'PeftModel.from_pretrained(base_model, "org/model")\n',
            f'PeftModel.from_pretrained(base_model, "org/model", revision="{SHA}")\n',
        ),
    ],
)
def test_new_call_kinds_insert_or_replace_and_are_idempotent(
    call: CallKind, source: str, expected: str, tmp_path: Path
) -> None:
    locked = dependency(call=call)
    path, plan = plan_source(tmp_path, source, locked)

    assert not plan.skipped
    assert plan.changes[0].after.decode() == expected
    path.write_bytes(plan.changes[0].after)

    second = plan_rewrites(tmp_path, Lockfile(1, (locked,)), scan_path(tmp_path))
    assert not second.changes and not second.skipped
    assert len(second.noops) == 1


@pytest.mark.parametrize(
    ("call", "source"),
    [
        (
            CallKind.PIPELINE,
            'pipeline(model="org/model", revision=REVISION)\n',
        ),
        (
            CallKind.SENTENCE_TRANSFORMER,
            'SentenceTransformer("org/model", revision=REVISION)\n',
        ),
        (
            CallKind.PEFT_FROM_PRETRAINED,
            'PeftModel.from_pretrained(base_model, "org/model", revision=REVISION)\n',
        ),
    ],
)
def test_new_call_kinds_do_not_overwrite_dynamic_revisions(
    call: CallKind, source: str, tmp_path: Path
) -> None:
    path, plan = plan_source(tmp_path, source, dependency(call=call))

    assert not plan.changes
    assert any("not a direct string literal" in item.reason for item in plan.skipped)
    assert path.read_text(encoding="utf-8") == source


def test_literal_replacement_and_matching_literal_noop(tmp_path: Path) -> None:
    source = (
        "before = object()\n"
        "model = AutoModel.from_pretrained(\n"
        "    \"org/model\", revision='floating',  # keep this\n"
        ")\n"
        "after = object()\n"
    )
    _, replacement = plan_source(tmp_path, source, dependency(line=2))
    expected = source.replace("'floating'", f"'{SHA}'")

    assert replacement.changes[0].after.decode() == expected
    (tmp_path / "app.py").write_text(expected, encoding="utf-8")
    noop = plan_rewrites(
        tmp_path, Lockfile(1, (dependency(line=2),)), scan_path(tmp_path)
    )

    assert not noop.changes
    assert len(noop.noops) == 1
    assert not noop.skipped


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        (
            'AutoModel.from_pretrained("org/model", revision=REVISION)\n',
            "not a direct string literal",
        ),
        (
            'AutoModel.from_pretrained("org/model", *extra)\n',
            "*args or **kwargs",
        ),
        (
            'AutoModel.from_pretrained("org/model", **options)\n',
            "*args or **kwargs",
        ),
        (
            'AutoModel.from_pretrained("org/model", revision="a", revision="b")\n',
            "duplicate revision= keywords",
        ),
        (
            'AutoModel.from_pretrained("org/model", revision="a" "b")\n',
            "not a direct string literal",
        ),
    ],
)
def test_unsafe_argument_shapes_are_skipped_without_changes(
    source: str, reason: str, tmp_path: Path
) -> None:
    path, plan = plan_source(tmp_path, source)

    assert not plan.changes
    assert any(reason in skipped.reason for skipped in plan.skipped)
    assert path.read_text(encoding="utf-8") == source


def test_stale_conflicting_and_same_location_ambiguity_are_skipped(
    tmp_path: Path,
) -> None:
    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    _, stale = plan_source(
        stale_dir,
        'AutoModel.from_pretrained("org/new")\n',
        dependency(repo_id="org/old"),
    )
    assert any("identity" in item.reason for item in stale.skipped)
    assert not stale.changes

    conflict_dir = tmp_path / "conflict"
    conflict_dir.mkdir()
    _, conflict = plan_source(
        conflict_dir,
        'AutoModel.from_pretrained("org/model")\n',
        dependency(sha=SHA),
        dependency(sha=OTHER_SHA),
    )
    assert any("conflicting lock targets" in item.reason for item in conflict.skipped)
    assert not conflict.changes

    ambiguous_dir = tmp_path / "ambiguous"
    ambiguous_dir.mkdir()
    _, ambiguous = plan_source(
        ambiguous_dir,
        'AutoModel.from_pretrained("org/model"); '
        'AutoModel.from_pretrained("org/model")\n',
    )
    assert any("multiple compatible calls" in item.reason for item in ambiguous.skipped)
    assert not ambiguous.changes


def test_multifile_diff_is_deterministic_and_unrelated_code_is_unchanged(
    tmp_path: Path,
) -> None:
    (tmp_path / "z.py").write_text(
        'z = 1\nload_dataset("org/data")\n', encoding="utf-8"
    )
    (tmp_path / "a.py").write_text(
        'a = 1\nAutoModel.from_pretrained("org/model")\n', encoding="utf-8"
    )
    dependencies = (
        dependency("z.py", 2, repo_id="org/data", call=CallKind.LOAD_DATASET),
        dependency("a.py", 2),
    )

    first = plan_rewrites(tmp_path, Lockfile(1, dependencies), scan_path(tmp_path))
    second = plan_rewrites(
        tmp_path, Lockfile(1, tuple(reversed(dependencies))), scan_path(tmp_path)
    )
    output = "".join(change.diff for change in first.changes)

    assert output == "".join(change.diff for change in second.changes)
    assert output.index("--- a/a.py") < output.index("--- a/z.py")
    assert " a = 1" in output and " z = 1" in output
    assert (
        first.changes[0].after.splitlines(keepends=True)[0]
        == (first.changes[0].before.splitlines(keepends=True)[0])
    )
    assert (
        first.changes[1].after.splitlines(keepends=True)[0]
        == (first.changes[1].before.splitlines(keepends=True)[0])
    )


def test_atomic_write_preserves_encoding_newlines_final_newline_and_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "app.py"
    before = (
        b"# coding: latin-1\r\n"
        b'model = AutoModel.from_pretrained("org/model")  # caf\xe9'
    )
    path.write_bytes(before)
    os.chmod(path, 0o744)
    original_mode = stat.S_IMODE(path.stat().st_mode)
    plan = plan_rewrites(
        tmp_path, Lockfile(1, (dependency(line=2),)), scan_path(tmp_path)
    )
    real_replace = os.replace
    replacements: list[tuple[Path, Path]] = []

    def recording_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(hf_freeze.rewrite.os, "replace", recording_replace)
    written, skipped = apply_rewrite_plan(tmp_path, plan)

    after = path.read_bytes()
    assert written == ("app.py",) and not skipped
    assert replacements[0][0].parent == replacements[0][1].parent == tmp_path
    assert b"\r\n" in after and not after.endswith((b"\n", b"\r"))
    assert b"caf\xe9" in after and SHA.encode() in after
    assert stat.S_IMODE(path.stat().st_mode) == original_mode


def test_cli_dry_run_never_writes_and_write_is_idempotent(tmp_path: Path) -> None:
    source = 'AutoModel.from_pretrained("org/model")\n'
    path = tmp_path / "app.py"
    path.write_text(source, encoding="utf-8")
    write_lockfile(tmp_path / "hf.lock", Lockfile(1, (dependency(),)))
    runner = CliRunner()

    preview = runner.invoke(app, ["pin", str(tmp_path)])
    assert preview.exit_code == 0
    assert preview.stdout.startswith("--- a/app.py\n+++ b/app.py\n")
    assert path.read_text(encoding="utf-8") == source

    first = runner.invoke(app, ["pin", str(tmp_path), "--write"])
    first_bytes = path.read_bytes()
    second = runner.invoke(app, ["pin", str(tmp_path), "--write"])

    assert first.exit_code == second.exit_code == 0
    assert first.stdout == "Wrote app.py\n"
    assert second.stdout.startswith("Already pinned app.py:1:")
    assert path.read_bytes() == first_bytes


@pytest.mark.parametrize(
    ("first_repo", "second_repo"),
    [("org/first", "org/second"), ("org/model", "org/model")],
)
def test_cli_second_write_reconciles_shifted_later_call(
    first_repo: str, second_repo: str, tmp_path: Path
) -> None:
    source = (
        f'AutoModel.from_pretrained(\n    "{first_repo}",\n)\n'
        f'AutoModel.from_pretrained("{second_repo}")\n'
    )
    path = tmp_path / "app.py"
    path.write_text(source, encoding="utf-8", newline="")
    lockfile = Lockfile(
        1,
        (
            dependency(line=1, repo_id=first_repo),
            dependency(line=4, repo_id=second_repo),
        ),
    )
    write_lockfile(tmp_path / "hf.lock", lockfile)
    runner = CliRunner()

    first = runner.invoke(app, ["pin", str(tmp_path), "--write"])
    written = path.read_bytes()
    second = runner.invoke(app, ["pin", str(tmp_path), "--write"])

    assert first.exit_code == second.exit_code == 0
    assert second.stderr == ""
    assert second.stdout.count("Already pinned app.py:") == 2
    assert path.read_bytes() == written


def test_cli_reconciles_two_shifted_same_repository_calls(tmp_path: Path) -> None:
    source = (
        'AutoModel.from_pretrained(\n    "org/model",\n)\n'
        'AutoModel.from_pretrained("org/model")\n\n'
        'AutoModel.from_pretrained("org/model")\n'
    )
    path = tmp_path / "app.py"
    path.write_text(source, encoding="utf-8", newline="")
    lockfile = Lockfile(
        1,
        (
            dependency(line=1),
            dependency(line=4),
            dependency(line=6),
        ),
    )
    write_lockfile(tmp_path / "hf.lock", lockfile)
    runner = CliRunner()

    first = runner.invoke(app, ["pin", str(tmp_path), "--write"])
    written = path.read_bytes()
    second = runner.invoke(app, ["pin", str(tmp_path), "--write"])

    assert first.exit_code == second.exit_code == 0
    assert second.stderr == ""
    assert second.stdout.count("Already pinned app.py:") == 3
    assert path.read_bytes() == written


def test_noop_reconciliation_rejects_group_count_mismatch(tmp_path: Path) -> None:
    locked = dependency()
    locked = LockedDependency(
        locked.repo_id,
        locked.repo_type,
        locked.kind,
        locked.requested_revision,
        locked.sha,
        (
            LockedSource("app.py", 1, CallKind.FROM_PRETRAINED),
            LockedSource("app.py", 2, CallKind.FROM_PRETRAINED),
        ),
    )
    source = f'\n\nAutoModel.from_pretrained("org/model", revision="{SHA}")\n'
    path, plan = plan_source(tmp_path, source, locked)

    assert not plan.changes and not plan.noops
    assert any(
        "found 1 uncovered exact-SHA call(s) for 2 unmatched lock source(s)"
        in item.reason
        for item in plan.skipped
    )
    assert path.read_text(encoding="utf-8") == source


def test_uniquely_relocated_unpinned_call_is_not_rewritten(tmp_path: Path) -> None:
    source = '\nAutoModel.from_pretrained("org/model")\n'
    path, plan = plan_source(tmp_path, source, dependency(line=1))

    assert not plan.changes and not plan.noops
    assert any("no supported call remains" in item.reason for item in plan.skipped)
    assert path.read_text(encoding="utf-8") == source


def test_atomic_replace_failure_preserves_source_and_cleans_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = b'AutoModel.from_pretrained("org/model")\n'
    path = tmp_path / "app.py"
    path.write_bytes(source)
    write_lockfile(tmp_path / "hf.lock", Lockfile(1, (dependency(),)))

    def fail_replace(*_: object) -> None:
        raise OSError("replace denied")

    monkeypatch.setattr(hf_freeze.rewrite.os, "replace", fail_replace)
    result = CliRunner().invoke(app, ["pin", str(tmp_path), "--write"])

    assert result.exit_code == 1
    assert "write failed: replace denied" in result.stderr
    assert path.read_bytes() == source
    assert not list(tmp_path.glob(".app.py.*.tmp"))


def test_cli_safely_applies_other_files_but_exits_one_for_skip(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        'AutoModel.from_pretrained("org/model")\n', encoding="utf-8"
    )
    (tmp_path / "b.py").write_text(
        'AutoModel.from_pretrained("org/other", revision=get_revision())\n',
        encoding="utf-8",
    )
    lockfile = Lockfile(
        1,
        (
            dependency("a.py"),
            dependency("b.py", repo_id="org/other"),
        ),
    )
    write_lockfile(tmp_path / "hf.lock", lockfile)

    result = CliRunner().invoke(app, ["pin", str(tmp_path), "--write"])

    assert result.exit_code == 1
    assert result.stdout == "Wrote a.py\n"
    assert "not a direct string literal" in result.stderr
    assert SHA in (tmp_path / "a.py").read_text(encoding="utf-8")
    assert "get_revision()" in (tmp_path / "b.py").read_text(encoding="utf-8")


def test_configured_pin_ignores_out_of_scope_lock_sources(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.hf-freeze]\ninclude = ["src/**/*.py"]\n', encoding="utf-8"
    )
    selected = tmp_path / "src" / "app.py"
    selected.parent.mkdir()
    selected.write_text('AutoModel.from_pretrained("org/model")\n', encoding="utf-8")
    excluded = tmp_path / "tests" / "old.py"
    excluded.parent.mkdir()
    excluded.write_text('AutoModel.from_pretrained("org/model")\n', encoding="utf-8")
    locked = dependency("src/app.py")
    locked = LockedDependency(
        locked.repo_id,
        locked.repo_type,
        locked.kind,
        locked.requested_revision,
        locked.sha,
        (
            LockedSource("src/app.py", 1, CallKind.FROM_PRETRAINED),
            LockedSource("tests/old.py", 1, CallKind.FROM_PRETRAINED),
        ),
    )
    write_lockfile(tmp_path / "hf.lock", Lockfile(1, (locked,)))

    result = CliRunner().invoke(app, ["pin", str(selected.parent), "--write"])

    assert result.exit_code == 0
    assert result.stdout == "Wrote src/app.py\n"
    assert SHA in selected.read_text(encoding="utf-8")
    assert SHA not in excluded.read_text(encoding="utf-8")


def test_symlink_and_missing_lock_or_path_fail_safely(tmp_path: Path) -> None:
    missing_path = CliRunner().invoke(app, ["pin", str(tmp_path / "missing")])
    assert missing_path.exit_code == 2

    (tmp_path / "app.py").write_text(
        'AutoModel.from_pretrained("org/model")\n', encoding="utf-8"
    )
    missing_lock = CliRunner().invoke(app, ["pin", str(tmp_path)])
    assert missing_lock.exit_code == 1
    assert "Error:" in missing_lock.stderr

    target = tmp_path / "real.py"
    target.write_text('AutoModel.from_pretrained("org/model")\n', encoding="utf-8")
    link = tmp_path / "link.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    link_lock = Lockfile(1, (dependency("link.py"),))
    link_plan = plan_rewrites(tmp_path, link_lock, scan_path(link))
    assert any("symlink" in item.reason for item in link_plan.skipped)
    assert not link_plan.changes
