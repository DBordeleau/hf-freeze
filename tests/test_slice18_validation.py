import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "validate_slice18.py"
SPEC = importlib.util.spec_from_file_location("validate_slice18", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Slice 18 Test",
            "-c",
            "user.email=slice18@example.invalid",
            *arguments,
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return completed.stdout.strip()


def _repository_with_commit(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    (repository / "evidence.txt").write_text("audited\n", encoding="utf-8")
    _git(repository, "add", "evidence.txt")
    _git(repository, "commit", "--quiet", "-m", "audited base")
    return repository, _git(repository, "rev-parse", "HEAD")


def test_manifest_is_bounded_exact_and_demonstrates_expanded_contract() -> None:
    root = Path(__file__).parents[1]
    manifest = validation.load_manifest(
        root / "validation" / "slice18" / "manifest.json", root
    )

    assert 10 <= len(manifest["projects"]) <= 12
    assert manifest["hf_freeze_baseline"] == (
        "2ca7646c4ebc510b71ad905d202e216183da1ae9"
    )
    assert (
        sum(project["cohort"] == "production-style" for project in manifest["projects"])
        >= 2
    )
    mechanisms = {
        mechanism
        for project in manifest["projects"]
        for mechanism in project["mechanisms"]
    }
    assert mechanisms >= {
        "application_scope",
        "environment_binding",
        "dependency_annotation",
        "ignore_directive",
    }


def test_audited_base_accepts_equal_head_offline(tmp_path: Path) -> None:
    repository, audited_base = _repository_with_commit(tmp_path)

    assert validation.validate_audited_base(repository, audited_base) == audited_base


def test_audited_base_accepts_descendant_head_offline(tmp_path: Path) -> None:
    repository, audited_base = _repository_with_commit(tmp_path)
    (repository / "evidence.txt").write_text("audited\ndescendant\n", encoding="utf-8")
    _git(repository, "add", "evidence.txt")
    _git(repository, "commit", "--quiet", "-m", "descendant")
    source_head = _git(repository, "rev-parse", "HEAD")

    assert validation.validate_audited_base(repository, audited_base) == source_head


def test_audited_base_rejects_non_descendant_and_unavailable_commits_offline(
    tmp_path: Path,
) -> None:
    repository, audited_base = _repository_with_commit(tmp_path)
    tree = _git(repository, "rev-parse", f"{audited_base}^{{tree}}")
    unrelated = _git(repository, "commit-tree", tree, "-m", "unrelated root")

    with pytest.raises(validation.ValidationError, match="is not an ancestor"):
        validation.validate_audited_base(repository, unrelated)
    with pytest.raises(validation.ValidationError, match="is unavailable"):
        validation.validate_audited_base(repository, "0" * 40)


def test_parse_coverage_requires_all_five_categories_in_order() -> None:
    output = "heading\n" + "\n".join(
        f"  {name}: {index}" for index, name in enumerate(validation.COVERAGE_ORDER)
    )

    assert validation.parse_coverage(output) == dict(
        zip(validation.COVERAGE_ORDER, range(5), strict=True)
    )

    with pytest.raises(validation.ValidationError, match="missing or out of order"):
        validation.parse_coverage("  UNRESOLVED: 1\n  LOCKED_STATIC: 0\n")


def test_sanitized_environment_never_uses_ambient_dependency_or_token_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MODEL_ID", "ambient/wrong-model")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("hugging_face_hub_token", "lowercase-secret")

    absent = validation.sanitized_environment(tmp_path, absent=("MODEL_ID",))
    conflicting = validation.sanitized_environment(
        tmp_path, absent=("MODEL_ID",), conflicting=("MODEL_ID",)
    )

    assert "MODEL_ID" not in absent
    assert absent["HF_HOME"] == str(tmp_path / "hf-home")
    assert "HF_TOKEN" not in absent
    assert "hugging_face_hub_token" not in absent
    assert conflicting["MODEL_ID"] == "hf-freeze/slice18-conflicting-value"
    assert "HF_TOKEN" not in conflicting


def test_lock_source_count_excludes_non_lock_entries() -> None:
    entries = [
        {"repo_id": "org/one", "sources": [{"path": "a.py"}, {"path": "b.py"}]},
        {"repo_id": "org/two", "sources": [{"path": "c.py"}]},
    ]

    assert validation._lock_source_count(entries) == 3


def test_classification_equivalence_ignores_only_uv_install_chatter() -> None:
    summary = "\n".join(
        f"  {name}: {int(name == 'LOCKED_ENV_BINDING') * 2}"
        for name in validation.COVERAGE_ORDER
    )
    first = (
        "Installed 26 packages in 511ms\n"
        "app.py:1:1  model org/repo coverage=LOCKED_ENV_BINDING\n"
        "Coverage summary:\n"
        f"{summary}\n"
    )
    second = first.replace("511ms", "754ms")

    assert validation._classification_equivalent(0, first, 0, second)
    assert not validation._classification_equivalent(
        0, first, 0, second.replace("org/repo", "ambient/wrong")
    )


def test_source_diff_excludes_uv_chatter_and_warning_only_output() -> None:
    diff = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n"

    assert validation._source_diff(f"Installed 26 packages in 1ms\n{diff}") == diff
    assert validation._source_diff("Installed 26 packages\nWARNING ignored\n") == ""


def test_tracked_results_are_complete_and_do_not_freeze_visible_exceptions() -> None:
    root = Path(__file__).parents[1]
    manifest = json.loads(
        (root / "validation" / "slice18" / "manifest.json").read_text(encoding="utf-8")
    )
    results = json.loads(
        (root / "validation" / "slice18" / "results.json").read_text(encoding="utf-8")
    )

    assert results["hf_freeze"]["audited_base"] == manifest["hf_freeze_baseline"]
    assert results["hf_freeze"]["source_head"] == (
        "2ca7646c4ebc510b71ad905d202e216183da1ae9"
    )
    assert "source_baseline" not in results["hf_freeze"]
    assert results["validation_errors"] == []
    assert {project["id"] for project in results["projects"]} == {
        project["id"] for project in manifest["projects"]
    }
    assert all(
        project["lifecycle"]["status"]
        in {"complete", "complete_with_acknowledged_dynamic"}
        for project in results["projects"]
    )
    for project in results["projects"]:
        coverage = project["configured_scope"]["coverage"]
        lockable = sum(coverage[name] for name in validation.COVERAGE_ORDER[:3])
        assert (
            validation._lock_source_count(project["lifecycle"]["lock_entries"])
            == lockable
        )
        assert coverage["UNRESOLVED"] == 0

    tsdae = next(project for project in results["projects"] if project["id"] == "tsdae")
    assert tsdae["configured_scope"]["coverage"]["ACKNOWLEDGED_DYNAMIC"] == 2
    assert tsdae["lifecycle"]["lock_entries"] == []
    assert tsdae["lifecycle"]["source_pin_diff"] == ""
    invariance = [
        project["ambient_environment_invariance"]
        for project in results["projects"]
        if project["ambient_environment_invariance"] is not None
    ]
    assert len(invariance) == 2
    assert all(
        item["classification_equivalent"] and item["lock_truth_equivalent"]
        for item in invariance
    )
