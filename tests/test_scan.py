from pathlib import Path

from typer.testing import CliRunner

from hf_freeze.cli import app
from hf_freeze.models import CallKind, RepoType
from hf_freeze.scan import scan_path


def write_project(tmp_path: Path, files: dict[str, str]) -> Path:
    for relative_path, source in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return tmp_path


def test_discovers_supported_calls_constants_and_revisions(tmp_path: Path) -> None:
    project = write_project(
        tmp_path,
        {
            "calls.py": """\
MODEL_ID = "org/model"
DATASET_ID = "org/data"

model = AutoModel.from_pretrained(MODEL_ID, revision="model-v1")
data = load_dataset(DATASET_ID, revision="data-v2")
file = hf_hub_download(
    repo_id="org/files", repo_type="dataset", revision="files-v3"
)
snapshot = snapshot_download("gpt2", revision="snapshot-v4")
""",
        },
    )

    result = scan_path(project)

    assert [finding.call_kind for finding in result.findings] == [
        CallKind.FROM_PRETRAINED,
        CallKind.LOAD_DATASET,
        CallKind.HF_HUB_DOWNLOAD,
        CallKind.SNAPSHOT_DOWNLOAD,
    ]
    assert [finding.repo_id for finding in result.findings] == [
        "org/model",
        "org/data",
        "org/files",
        "gpt2",
    ]
    assert [finding.repo_type for finding in result.findings] == [
        RepoType.MODEL,
        RepoType.DATASET,
        RepoType.DATASET,
        RepoType.MODEL,
    ]
    assert [finding.requested_revision for finding in result.findings] == [
        "model-v1",
        "data-v2",
        "files-v3",
        "snapshot-v4",
    ]
    assert [
        (finding.source.line, finding.source.column) for finding in result.findings
    ] == [
        (4, 8),
        (5, 7),
        (6, 7),
        (9, 11),
    ]
    assert result.diagnostics == ()


def test_reports_dynamic_ids_and_ambiguous_constants(tmp_path: Path) -> None:
    project = write_project(
        tmp_path,
        {
            "dynamic.py": """\
import os
MODEL_ID = "org/first"
MODEL_ID = "org/second"

AutoModel.from_pretrained(os.environ["MODEL_ID"])
load_dataset(f"org/{name}")
hf_hub_download(repo_id=choose_repo())
snapshot_download(repo_id=MODEL_ID)
""",
        },
    )

    findings = scan_path(project).findings

    assert [finding.repo_id for finding in findings] == [None, None, None, None]
    assert [finding.unresolved_reason for finding in findings] == [
        "repository ID is a subscript expression",
        "repository ID is an interpolated string",
        "repository ID is returned by a function call",
        "repository ID name 'MODEL_ID' does not have one unambiguous string assignment",
    ]


def test_constant_resolution_respects_function_scopes(tmp_path: Path) -> None:
    project = write_project(
        tmp_path,
        {
            "scopes.py": """\
def hidden_binding():
    REPO_ID = "org/hidden"

def no_binding_here():
    AutoModel.from_pretrained(REPO_ID)

class HiddenClass:
    CLASS_ID = "org/class-hidden"

def no_class_binding_here():
    snapshot_download(CLASS_ID)

def first():
    REPO_ID = "org/first"
    load_dataset(REPO_ID)

def second():
    REPO_ID = "org/second"
    snapshot_download(REPO_ID)
""",
        },
    )

    findings = scan_path(project).findings

    assert [finding.repo_id for finding in findings] == [
        None,
        None,
        "org/first",
        "org/second",
    ]
    assert findings[0].unresolved_reason == (
        "repository ID name 'REPO_ID' does not have one unambiguous string assignment"
    )
    assert findings[1].unresolved_reason == (
        "repository ID name 'CLASS_ID' does not have one unambiguous string assignment"
    )


def test_rebound_and_nonliteral_local_names_are_unresolved(tmp_path: Path) -> None:
    project = write_project(
        tmp_path,
        {
            "rebound.py": """\
def rebound():
    REPO_ID = "org/model"
    REPO_ID = choose_repo()
    AutoModel.from_pretrained(REPO_ID)

def nonliteral():
    REPO_ID = choose_repo()
    load_dataset(REPO_ID)
""",
        },
    )

    findings = scan_path(project).findings

    assert [finding.repo_id for finding in findings] == [None, None]
    assert all(
        finding.unresolved_reason is not None
        and "unambiguous string assignment" in finding.unresolved_reason
        for finding in findings
    )


def test_ignores_obvious_local_paths_but_keeps_single_segment_ids(
    tmp_path: Path,
) -> None:
    project = write_project(
        tmp_path,
        {
            "paths.py": """\
AutoModel.from_pretrained("./model")
AutoModel.from_pretrained("../model")
AutoModel.from_pretrained(".\\\\model")
AutoModel.from_pretrained("..\\\\model")
AutoModel.from_pretrained("/var/models/model")
AutoModel.from_pretrained("\\\\models\\\\model")
AutoModel.from_pretrained("C:\\\\models\\\\model")
AutoModel.from_pretrained("C:models\\\\model")
AutoModel.from_pretrained("file:///models/model")
AutoModel.from_pretrained("gpt2")
""",
        },
    )

    findings = scan_path(project).findings

    assert [finding.repo_id for finding in findings] == ["gpt2"]


def test_default_exclusions_and_no_project_imports(tmp_path: Path) -> None:
    project = write_project(
        tmp_path,
        {
            "included.py": 'AutoModel.from_pretrained("org/included")\n',
            "danger.py": 'raise RuntimeError("scanner imported project code")\n',
            ".venv/ignored.py": 'AutoModel.from_pretrained("org/venv")\n',
            "build/ignored.py": 'load_dataset("org/build")\n',
            "nested/__pycache__/ignored.py": 'snapshot_download("org/cache")\n',
        },
    )

    result = scan_path(project)

    assert [finding.repo_id for finding in result.findings] == ["org/included"]
    assert result.diagnostics == ()


def test_recovers_from_parse_errors_and_orders_by_path_and_position(
    tmp_path: Path,
) -> None:
    project = write_project(
        tmp_path,
        {
            "z.py": """\
snapshot_download("org/second")
load_dataset("org/third")
""",
            "broken.py": "def broken(:\n",
            "a.py": 'AutoModel.from_pretrained("org/first")\n',
        },
    )

    result = scan_path(project)

    assert [(finding.source.path, finding.repo_id) for finding in result.findings] == [
        ("a.py", "org/first"),
        ("z.py", "org/second"),
        ("z.py", "org/third"),
    ]
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].source.path == "broken.py"
    assert result.diagnostics[0].source.line == 1
    assert "parse error" in result.diagnostics[0].message


def test_scan_cli_renders_findings_unresolved_and_parse_diagnostics(
    tmp_path: Path,
) -> None:
    project = write_project(
        tmp_path,
        {
            "valid.py": """\
AutoModel.from_pretrained("org/model", revision="v1")
load_dataset(get_dataset())
""",
            "invalid.py": "if True print('no')\n",
        },
    )

    result = CliRunner().invoke(app, ["scan", str(project)])

    assert result.exit_code == 0
    assert (
        "valid.py:1:1  from_pretrained  model  org/model  revision=v1" in result.stdout
    )
    assert "valid.py:2:1  load_dataset  unresolved:" in result.stdout
    assert "repository ID is returned by a function call" in result.stdout
    assert "invalid.py:1:" in result.stdout
    assert "parse error" in result.stdout


def test_scan_cli_fixture_smoke() -> None:
    fixture = Path(__file__).parent / "fixtures" / "scan_project"

    result = CliRunner().invoke(app, ["scan", str(fixture)])

    assert result.exit_code == 0
    assert (
        "app.py:3:9  from_pretrained  model  org/model  revision=main" in result.stdout
    )
    assert (
        "app.py:4:8  load_dataset  dataset  org/data  revision=<default>"
        in result.stdout
    )


def test_revision_discovery_distinguishes_omitted_resolved_and_dynamic(
    tmp_path: Path,
) -> None:
    project = write_project(
        tmp_path,
        {
            "revisions.py": """\
REVISION = "constant-v2"
AutoModel.from_pretrained("org/default")
AutoModel.from_pretrained("org/literal", revision="literal-v1")
AutoModel.from_pretrained("org/constant", revision=REVISION)
AutoModel.from_pretrained("org/dynamic", revision=get_revision())
""",
        },
    )

    findings = scan_path(project).findings

    assert [finding.requested_revision for finding in findings] == [
        None,
        "literal-v1",
        "constant-v2",
        None,
    ]
    assert findings[0].revision_unresolved_reason is None
    assert findings[3].revision_unresolved_reason == (
        "revision is returned by a function call"
    )

    result = CliRunner().invoke(app, ["scan", str(project)])
    assert (
        "revision=<unresolved: revision is returned by a function call>"
        in result.stdout
    )
    assert "org/dynamic  revision=<default>" not in result.stdout


def test_empty_and_whitespace_revisions_are_explicitly_unresolved(
    tmp_path: Path,
) -> None:
    project = write_project(
        tmp_path,
        {
            "empty.py": """\
EMPTY = ""
SPACE = "   "
AutoModel.from_pretrained("org/empty-literal", revision="")
AutoModel.from_pretrained("org/space-literal", revision="   ")
AutoModel.from_pretrained("org/empty-constant", revision=EMPTY)
AutoModel.from_pretrained("org/space-constant", revision=SPACE)
""",
        },
    )

    findings = scan_path(project).findings

    assert all(finding.requested_revision is None for finding in findings)
    assert all(
        finding.revision_unresolved_reason == "revision is empty or whitespace-only"
        for finding in findings
    )
    result = CliRunner().invoke(app, ["scan", str(project)])
    assert (
        result.stdout.count(
            "revision=<unresolved: revision is empty or whitespace-only>"
        )
        == 4
    )


def test_extracts_literal_and_scope_safe_trust_remote_code(tmp_path: Path) -> None:
    project = write_project(
        tmp_path,
        {
            "trust.py": """\
ALLOW = True
DENY = False
AutoModel.from_pretrained("org/model", trust_remote_code=True)
load_dataset("org/data", trust_remote_code=False)
hf_hub_download("org/file", trust_remote_code=ALLOW)
snapshot_download("org/snapshot", trust_remote_code=DENY)
""",
        },
    )

    findings = scan_path(project).findings

    assert [finding.trust_remote_code for finding in findings] == [
        True,
        False,
        True,
        False,
    ]
    assert all(
        finding.trust_remote_code_unresolved_reason is None for finding in findings
    )


def test_reports_dynamic_or_ambiguous_trust_remote_code(tmp_path: Path) -> None:
    project = write_project(
        tmp_path,
        {
            "trust.py": """\
ALLOW = True
ALLOW = False
AutoModel.from_pretrained("org/model", trust_remote_code=get_policy())
load_dataset("org/data", trust_remote_code=ALLOW)
""",
        },
    )

    findings = scan_path(project).findings

    assert findings[0].trust_remote_code_unresolved_reason == (
        "trust_remote_code is a dynamic expression"
    )
    assert findings[1].trust_remote_code_unresolved_reason == (
        "trust_remote_code name 'ALLOW' does not have one unambiguous boolean "
        "assignment"
    )
