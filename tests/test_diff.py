import pytest

from hf_freeze.diff import (
    ArtifactCategory,
    ChangeState,
    DiffError,
    IdentityKind,
    RepositoryFile,
    categorize,
    compare_trees,
    render_diff,
    select_locked_dependency,
    with_semantic_diff,
)
from hf_freeze.models import (
    CallKind,
    DependencyKind,
    LockedDependency,
    LockedSource,
    Lockfile,
    RepoType,
)


def file(path: str, size: int | None, **identities: str) -> RepositoryFile:
    return RepositoryFile(path, size, **identities)


def dependency(
    repo_id: str = "org/repo",
    repo_type: RepoType = RepoType.MODEL,
    revision: str = "main",
    sha: str = "a" * 40,
    kind: DependencyKind = DependencyKind.MODEL,
) -> LockedDependency:
    return LockedDependency(
        repo_id,
        repo_type,
        kind,
        revision,
        sha,
        (LockedSource("app.py", 1, CallKind.FROM_PRETRAINED),),
    )


def test_compare_trees_reports_states_identities_bytes_and_hidden_unchanged() -> None:
    old = (
        file("model.safetensors", 100, lfs_sha256="old", blob_id="same"),
        file("tokenizer.json", 20, xet_hash="same"),
        file("README.md", 30, blob_id="removed"),
        file("config.json", 10),
        file("data.csv", 4),
        file("same.bin", 8, blob_id="same"),
    )
    new = (
        file("model.safetensors", 110, lfs_sha256="new", blob_id="same"),
        file("tokenizer.json", 20, xet_hash="same"),
        file("config.json", 10),
        file("data.csv", 5),
        file("same.bin", 8, blob_id="same"),
        file("custom.py", 7, blob_id="added"),
        file("unknown.bin", None, blob_id="added"),
    )

    result = compare_trees(old, new)
    changes = {item.path: item for item in result.files}

    assert changes["model.safetensors"].state is ChangeState.CHANGED
    assert changes["model.safetensors"].identity is IdentityKind.LFS_SHA256
    assert changes["tokenizer.json"].state is ChangeState.UNCHANGED
    assert changes["README.md"].state is ChangeState.REMOVED
    assert changes["config.json"].state is ChangeState.METADATA_UNAVAILABLE
    assert changes["data.csv"].identity is IdentityKind.SIZE
    assert result.changed_bytes == 122
    assert result.unknown_changed_sizes == 1

    output = render_diff("org/repo", "old", "new", result)
    assert "TOKENIZER     tokenizer.json" not in output
    assert "Metadata unavailable\n  CONFIG        config.json" in output
    assert "Remote Python code changed." in output
    assert output.index("custom.py") < output.index("unknown.bin")


def test_blob_fallback_and_lfs_priority_are_deterministic() -> None:
    result = compare_trees(
        (
            file("blob.txt", 1, blob_id="a"),
            file("priority.bin", 1, lfs_sha256="a", blob_id="same"),
        ),
        (
            file("blob.txt", 1, blob_id="b"),
            file("priority.bin", 1, lfs_sha256="b", blob_id="same"),
        ),
    )

    assert [item.identity for item in result.files] == [
        IdentityKind.GIT_BLOB,
        IdentityKind.LFS_SHA256,
    ]
    assert all(item.state is ChangeState.CHANGED for item in result.files)


@pytest.mark.parametrize(
    ("path", "category"),
    [
        ("adapter_model.safetensors", ArtifactCategory.ADAPTER),
        ("adapter_config.json", ArtifactCategory.ADAPTER),
        ("preprocessor_config.json", ArtifactCategory.PROCESSOR),
        ("modeling_custom.py", ArtifactCategory.REMOTE_CODE),
        ("LICENSE", ArtifactCategory.LICENSE),
    ],
)
def test_categories_use_ordered_specific_rules(
    path: str, category: ArtifactCategory
) -> None:
    assert categorize(path) is category


def test_semantic_json_diff_is_nested_bounded_and_safely_unavailable() -> None:
    result = compare_trees(
        (file("config.json", 20, blob_id="a"),),
        (file("config.json", 20, blob_id="b"),),
    )
    old = {f"key{index:02}": 0 for index in range(25)} | {"nested": {"x": 1}}
    new = {f"key{index:02}": 1 for index in range(25)} | {"nested": {"x": 2}}

    valid = with_semantic_diff(
        result,
        "config.json",
        str(old).replace("'", '"'),
        str(new).replace("'", '"'),
    )
    assert valid.files[0].semantic_paths is not None
    assert len(valid.files[0].semantic_paths) == 20
    assert valid.files[0].semantic_paths[0] == "key00"

    invalid = with_semantic_diff(result, "config.json", "{", "{}")
    assert invalid.files[0].state is ChangeState.CHANGED
    assert invalid.files[0].semantic_unavailable is True


def test_lock_selection_coalesces_only_matching_identity() -> None:
    first = dependency(kind=DependencyKind.MODEL)
    matching = dependency(kind=DependencyKind.SNAPSHOT)
    assert select_locked_dependency(Lockfile(1, (first, matching)), "org/repo") == first

    with pytest.raises(DiffError, match="ambiguous"):
        select_locked_dependency(
            Lockfile(
                1,
                (
                    first,
                    dependency(sha="b" * 40, kind=DependencyKind.SNAPSHOT),
                ),
            ),
            "org/repo",
        )
    with pytest.raises(DiffError, match="not present"):
        select_locked_dependency(Lockfile(1, ()), "org/repo")


def test_no_change_render_is_concise_and_remote_code_is_not_changed() -> None:
    result = compare_trees(
        (file("same.txt", 1, blob_id="x"),),
        (file("same.txt", 1, blob_id="x"),),
    )
    output = render_diff("org/repo", "old", "new", result)

    assert "No file changes detected." in output
    assert "same.txt" not in output
    assert "No remote Python code changed." in output


def test_remote_indicator_preserves_change_with_unavailable_metadata() -> None:
    result = compare_trees(
        (
            file("known.py", 10, blob_id="old"),
            file("uncertain.py", 10),
        ),
        (
            file("known.py", 10, blob_id="new"),
            file("uncertain.py", 10),
        ),
    )

    output = render_diff("org/repo", "old", "new", result)

    assert "REMOTE CODE   known.py" in output
    assert "REMOTE CODE   uncertain.py" in output
    assert (
        "Remote Python code changed; additional remote-code metadata was unavailable."
        in output
    )
