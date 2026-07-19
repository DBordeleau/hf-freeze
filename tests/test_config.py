from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from hf_freeze.config import (
    ConfigError,
    ProjectConfig,
    iter_scoped_python_files,
    resolve_project_context,
)
from hf_freeze.models import RepoType


def write_config(directory: Path, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pyproject.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_no_config_preserves_requested_root_behavior(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    source = nested / "app.py"
    source.write_text("", encoding="utf-8")
    write_config(tmp_path, "[project]\nname = 'unconfigured'\n")

    directory_context = resolve_project_context(nested)
    file_context = resolve_project_context(source)

    assert directory_context.root == nested.resolve()
    assert file_context.root == nested.resolve()
    assert directory_context.config_path is None
    assert directory_context.config == file_context.config == ProjectConfig()


def test_root_nested_and_file_resolve_same_immutable_context(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, "[tool.hf-freeze]\n")
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)
    source = nested / "app.py"
    source.write_text("", encoding="utf-8")

    contexts = (
        resolve_project_context(tmp_path),
        resolve_project_context(nested),
        resolve_project_context(source),
    )

    assert contexts[0] == contexts[1] == contexts[2]
    assert contexts[0].root == tmp_path.resolve()
    assert contexts[0].config_path == config_path.resolve()
    with pytest.raises(FrozenInstanceError):
        contexts[0].root = nested  # type: ignore[misc]


def test_nearest_configured_ancestor_wins_without_merging(tmp_path: Path) -> None:
    write_config(
        tmp_path,
        "[tool.hf-freeze]\ninclude = ['parent.py']\nexclude = ['parent/**']\n",
    )
    child = tmp_path / "child"
    child_config = write_config(
        child,
        "[tool.hf-freeze]\ninclude = ['child.py']\n",
    )
    nested = child / "src"
    nested.mkdir()

    context = resolve_project_context(nested)

    assert context.root == child.resolve()
    assert context.config_path == child_config.resolve()
    assert context.config.include == ("child.py",)
    assert context.config.exclude == ()


def test_unconfigured_pyproject_does_not_hide_configured_ancestor(
    tmp_path: Path,
) -> None:
    parent_config = write_config(tmp_path, "[tool.hf-freeze]\n")
    nested = tmp_path / "nested"
    write_config(nested, "[project]\nname = 'child'\n")

    context = resolve_project_context(nested)

    assert context.config_path == parent_config.resolve()


def test_typed_configuration_loads_in_deterministic_order(tmp_path: Path) -> None:
    write_config(
        tmp_path,
        """
[tool.hf-freeze]
include = ["src/**/*.py", "app.py"]
exclude = ["tests/**"]

[tool.hf-freeze.dependencies.zeta]
repo_id = "org/zeta"
repo_type = "dataset"
revision = "stable"

[tool.hf-freeze.dependencies.alpha]
repo_id = "org/alpha"
repo_type = "model"
revision = "main"

[tool.hf-freeze.bindings.environment]
Z_MODEL = "zeta"
A_MODEL = "alpha"
""",
    )

    config = resolve_project_context(tmp_path).config

    assert config.include == ("src/**/*.py", "app.py")
    assert config.exclude == ("tests/**",)
    assert [(item.name, item.repo_type) for item in config.dependencies] == [
        ("alpha", RepoType.MODEL),
        ("zeta", RepoType.DATASET),
    ]
    assert config.environment_bindings == (("A_MODEL", "alpha"), ("Z_MODEL", "zeta"))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[tool.hf-freeze\n", "malformed TOML"),
        ("[tool.hf-freeze]\nsurprise = true\n", "unknown key"),
        ("[tool.hf-freeze]\ninclude = 'src/**'\n", "array of strings"),
        ("[tool.hf-freeze]\nexclude = [1]\n", "array of strings"),
        ("[tool.hf-freeze]\ndependencies = []\n", "dependencies must be a table"),
        ("[tool.hf-freeze]\nbindings = []\n", "bindings must be a table"),
    ],
)
def test_invalid_configuration_is_path_specific(
    content: str, message: str, tmp_path: Path
) -> None:
    config_path = write_config(tmp_path, content)

    with pytest.raises(ConfigError) as raised:
        resolve_project_context(tmp_path)

    assert str(config_path.resolve()) in str(raised.value)
    assert message in str(raised.value)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            "[tool.hf-freeze.dependencies.\"bad.name\"]\nrepo_id='x'\n"
            "repo_type='model'\nrevision='main'\n",
            "dependency name",
        ),
        (
            "[tool.hf-freeze.dependencies.item]\nrepo_id='x'\n"
            "repo_type='space'\nrevision='main'\n",
            "'model' or 'dataset'",
        ),
        (
            "[tool.hf-freeze.bindings.environment]\nMODEL_ID='missing'\n",
            "unknown dependency",
        ),
    ],
)
def test_nested_placeholder_validation(
    content: str, message: str, tmp_path: Path
) -> None:
    write_config(tmp_path, content)

    with pytest.raises(ConfigError, match=message):
        resolve_project_context(tmp_path)


def test_scope_uses_gitignore_patterns_exclude_precedence_and_hard_exclusions(
    tmp_path: Path,
) -> None:
    write_config(
        tmp_path,
        """
[tool.hf-freeze]
include = ["app.py", "src/**/*.py", "build/reincluded.py"]
exclude = ["src/excluded/**"]
""",
    )
    for relative_path in (
        "app.py",
        "other.py",
        "src/first.py",
        "src/nested/second.py",
        "src/excluded/ignored.py",
        "build/reincluded.py",
    ):
        source = tmp_path / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("", encoding="utf-8")

    context = resolve_project_context(tmp_path)
    selected = iter_scoped_python_files(context, tmp_path)
    narrowed = iter_scoped_python_files(context, tmp_path / "src" / "nested")

    assert [display for _, display in selected] == [
        "app.py",
        "src/first.py",
        "src/nested/second.py",
    ]
    assert [display for _, display in narrowed] == ["src/nested/second.py"]


@pytest.mark.parametrize("field", ["include", "exclude"])
def test_invalid_gitignore_pattern_names_configuration_path_and_field(
    field: str, tmp_path: Path
) -> None:
    config_path = write_config(
        tmp_path, f'[tool.hf-freeze]\n{field} = ["invalid\\\\"]\n'
    )

    with pytest.raises(ConfigError) as raised:
        resolve_project_context(tmp_path)

    message = str(raised.value)
    assert str(config_path.resolve()) in message
    assert f"{field}[0]" in message
    assert "gitignore pattern" in message
