from pathlib import Path

import pytest
from typer.testing import CliRunner

from hf_freeze.cli import app
from hf_freeze.config import resolve_project_context
from hf_freeze.models import CallKind, DiagnosticSeverity, RepoType
from hf_freeze.scan import scan_path


def write_project(tmp_path: Path, files: dict[str, str]) -> Path:
    for relative_path, source in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return tmp_path


def write_declared_project(
    tmp_path: Path, source: str, *, repo_type: str = "model"
) -> tuple[Path, object]:
    project = write_project(
        tmp_path,
        {
            "pyproject.toml": "[tool.hf-freeze.dependencies.primary-model]\n"
            'repo_id = "org/model"\n'
            f'repo_type = "{repo_type}"\n'
            'revision = "stable"\n',
            "app.py": source,
        },
    )
    return project, resolve_project_context(project)


def write_environment_project(
    tmp_path: Path,
    source: str,
) -> tuple[Path, object]:
    project = write_project(
        tmp_path,
        {
            "pyproject.toml": "[tool.hf-freeze.dependencies.primary-model]\n"
            "repo_id = 'org/model'\n"
            "repo_type = 'model'\n"
            "revision = 'stable'\n\n"
            "[tool.hf-freeze.bindings.environment]\n"
            "MODEL_ID = 'primary-model'\n",
            "app.py": source,
        },
    )
    return project, resolve_project_context(project)


def test_dependency_directive_resolves_dynamic_call_and_tracking_revision(
    tmp_path: Path,
) -> None:
    project, context = write_declared_project(
        tmp_path,
        "# hf-freeze: dependency=primary-model\n"
        "model = AutoModel.from_pretrained(settings.model)\n"
        "# hf-freeze: dependency=primary-model\n"
        'other = AutoModel.from_pretrained("org/model", revision="stable")\n',
    )

    result = scan_path(project, context=context)

    assert result.diagnostics == ()
    resolved = [
        (item.repo_id, item.repo_type, item.requested_revision)
        for item in result.findings
    ]
    assert resolved == [
        ("org/model", RepoType.MODEL, "stable"),
        ("org/model", RepoType.MODEL, "stable"),
    ]


@pytest.mark.parametrize(
    "expression",
    [
        'os.environ["MODEL_ID"]',
        'os.environ.get("MODEL_ID")',
        'os.getenv("MODEL_ID")',
        'os.getenv("MODEL_ID", "org/fallback")',
        'os.getenv("MODEL_ID", choose_fallback())',
    ],
)
@pytest.mark.parametrize("assigned", [False, True])
def test_environment_bindings_resolve_each_accepted_expression(
    tmp_path: Path, expression: str, assigned: bool
) -> None:
    source = (
        f"model_id = {expression}\nAutoModel.from_pretrained(model_id)\n"
        if assigned
        else f"AutoModel.from_pretrained({expression})\n"
    )
    project, context = write_environment_project(tmp_path, source)

    result = scan_path(project, context=context)

    assert result.diagnostics == ()
    finding = result.findings[0]
    assert (finding.repo_id, finding.repo_type, finding.requested_revision) == (
        "org/model",
        RepoType.MODEL,
        "stable",
    )


@pytest.mark.parametrize("variable", ["MODEL_ID", "UNBOUND"])
def test_environment_expression_accepts_matching_or_fallback_directive(
    tmp_path: Path, variable: str
) -> None:
    project, context = write_environment_project(
        tmp_path,
        "# hf-freeze: dependency=primary-model\n"
        f'AutoModel.from_pretrained(os.getenv("{variable}"))\n',
    )

    result = scan_path(project, context=context)

    assert result.diagnostics == ()
    assert result.findings[0].repo_id == "org/model"


def test_environment_subscript_tuple_key_is_not_recognized(tmp_path: Path) -> None:
    project, context = write_environment_project(
        tmp_path, 'AutoModel.from_pretrained(os.environ["MODEL_ID",])\n'
    )

    result = scan_path(project, context=context)

    assert result.findings[0].repo_id is None


@pytest.mark.parametrize(
    "source",
    [
        'model_id = os.getenv("MODEL_ID")\nmodel_id = choose()\n'
        "AutoModel.from_pretrained(model_id)\n",
        'model_id = os.getenv("MODEL_ID")\nalias = model_id\n'
        "AutoModel.from_pretrained(alias)\n",
        'model_id, other = os.getenv("MODEL_ID"), None\n'
        "AutoModel.from_pretrained(model_id)\n",
        'first = second = os.getenv("MODEL_ID")\nAutoModel.from_pretrained(first)\n',
        'AutoModel.from_pretrained(model_id)\nmodel_id = os.getenv("MODEL_ID")\n',
        'model_id = os.getenv("MODEL_ID")\n\ndef load():\n'
        "    AutoModel.from_pretrained(model_id)\n",
    ],
)
def test_ambiguous_alias_and_cross_scope_environment_flow_stays_unresolved(
    tmp_path: Path, source: str
) -> None:
    project, context = write_environment_project(tmp_path, source)

    result = scan_path(project, context=context)

    if result.findings:
        assert result.findings[0].repo_id is None
        assert "unambiguous string assignment" in result.findings[0].unresolved_reason
    else:
        assert any(
            item.code == "AMBIGUOUS_ENVIRONMENT_REFERENCE"
            for item in result.diagnostics
        )


@pytest.mark.parametrize(
    ("source", "repo_type", "code"),
    [
        (
            "# hf-freeze: dependency=missing\n"
            "AutoModel.from_pretrained(settings.model)\n",
            "model",
            "UNKNOWN_DECLARATION",
        ),
        (
            "# hf-freeze dependency=primary-model\n"
            "AutoModel.from_pretrained(settings.model)\n",
            "model",
            "MALFORMED_DIRECTIVE",
        ),
        (
            "# hf-freeze: dependency=primary-model\n\n"
            "AutoModel.from_pretrained(settings.model)\n",
            "model",
            "DETACHED_DIRECTIVE",
        ),
        (
            "# hf-freeze: dependency=primary-model\n"
            "# ordinary intervening comment\n"
            "AutoModel.from_pretrained(settings.model)\n",
            "model",
            "DETACHED_DIRECTIVE",
        ),
        (
            "# hf-freeze: dependency=primary-model\n"
            "# hf-freeze: dependency=primary-model\n"
            "AutoModel.from_pretrained(settings.model)\n",
            "model",
            "MULTIPLE_DIRECTIVES",
        ),
        (
            "# hf-freeze: dependency=primary-model\nvalue = settings.model\n",
            "model",
            "DIRECTIVE_CALL_COUNT",
        ),
        (
            "# hf-freeze: dependency=primary-model\n"
            "a = AutoModel.from_pretrained(settings.a); "
            "b = AutoModel.from_pretrained(settings.b)\n",
            "model",
            "DIRECTIVE_CALL_COUNT",
        ),
        (
            "# hf-freeze: dependency=primary-model\n"
            "AutoModel.from_pretrained(settings.model)\n",
            "dataset",
            "DIRECTIVE_CONFLICT",
        ),
        (
            "# hf-freeze: dependency=primary-model\n"
            'AutoModel.from_pretrained("other/model")\n',
            "model",
            "DIRECTIVE_CONFLICT",
        ),
        (
            "# hf-freeze: dependency=primary-model\n"
            "hf_hub_download(settings.model, repo_type=get_type())\n",
            "model",
            "DIRECTIVE_CONFLICT",
        ),
        (
            "# hf-freeze: dependency=primary-model\n"
            'AutoModel.from_pretrained(settings.model, revision="other")\n',
            "model",
            "DIRECTIVE_CONFLICT",
        ),
        (
            "# hf-freeze: dependency=primary-model\n"
            "if enabled:\n"
            "    AutoModel.from_pretrained(settings.model)\n",
            "model",
            "DETACHED_DIRECTIVE",
        ),
        (
            "value = 1  # hf-freeze: dependency=primary-model\n"
            "AutoModel.from_pretrained(settings.model)\n",
            "model",
            "DETACHED_DIRECTIVE",
        ),
    ],
)
def test_dependency_directive_errors_are_deterministic(
    tmp_path: Path, source: str, repo_type: str, code: str
) -> None:
    project, context = write_declared_project(tmp_path, source, repo_type=repo_type)

    result = scan_path(project, context=context)

    assert code in {item.code for item in result.diagnostics}
    assert any(item.severity is DiagnosticSeverity.ERROR for item in result.diagnostics)


def test_unused_declaration_is_a_nonfatal_deterministic_warning(
    tmp_path: Path,
) -> None:
    project, context = write_declared_project(
        tmp_path, 'AutoModel.from_pretrained("other/model")\n'
    )

    result = scan_path(project, context=context)

    warning = result.diagnostics[0]
    assert warning.code == "UNUSED_DECLARATION"
    assert warning.severity is DiagnosticSeverity.WARNING
    assert result.findings[0].repo_id == "other/model"


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


def test_discovers_pipeline_sentence_transformer_and_peft_calls(
    tmp_path: Path,
) -> None:
    project = write_project(
        tmp_path,
        {
            "calls.py": """\
PIPELINE_ID = "org/pipeline"
SENTENCE_ID = "org/sentence"
ADAPTER_ID = "org/adapter"

pipeline("text-classification", model=PIPELINE_ID, trust_remote_code=True)
SentenceTransformer(model_name_or_path=SENTENCE_ID, revision="sentence-v1")
PeftModel.from_pretrained(base_model, ADAPTER_ID)
""",
        },
    )

    findings = scan_path(project).findings

    assert [finding.call_kind for finding in findings] == [
        CallKind.PIPELINE,
        CallKind.SENTENCE_TRANSFORMER,
        CallKind.PEFT_FROM_PRETRAINED,
    ]
    assert [finding.repo_id for finding in findings] == [
        "org/pipeline",
        "org/sentence",
        "org/adapter",
    ]
    assert [finding.repo_type for finding in findings] == [RepoType.MODEL] * 3
    assert findings[0].trust_remote_code is True
    assert findings[1].requested_revision == "sentence-v1"


def test_ignores_non_hub_sentence_transformer_and_local_dataset_builders(
    tmp_path: Path,
) -> None:
    project = write_project(
        tmp_path,
        {
            "local_inputs.py": """\
import os

SentenceTransformer(modules=[transformer, pooling])
load_dataset("json", data_files="train.jsonl")
load_dataset("parquet", data_files={"train": os.path.join(root, "train.parquet")})
""",
        },
    )

    assert scan_path(project).findings == ()


def test_hf_data_files_are_actionable_unresolved_instead_of_fake_builder_ids(
    tmp_path: Path,
) -> None:
    project = write_project(
        tmp_path,
        {
            "hub_files.py": """\
load_dataset(
    "parquet",
    data_files={"train": "hf://datasets/org/data/train.parquet"},
)
load_dataset(
    "parquet",
    data_files={"train": f"hf://datasets/org/data/{split}/train.parquet"},
)
load_dataset(
    "parquet",
    data_files="https://huggingface.co/datasets/org/data/resolve/main/train.parquet",
)
""",
        },
    )

    findings = scan_path(project).findings

    assert len(findings) == 3
    assert all(finding.call_kind is CallKind.LOAD_DATASET for finding in findings)
    assert all(finding.repo_id is None for finding in findings)
    assert {finding.unresolved_reason for finding in findings} == {
        "load_dataset uses Hugging Face data_files; repository ID extraction from "
        "data_files is unsupported"
    }


def test_unknown_dataset_builder_data_files_are_actionable_unresolved(
    tmp_path: Path,
) -> None:
    project = write_project(
        tmp_path,
        {
            "unknown_files.py": """\
DATA_FILES = get_data_files()
load_dataset("json", data_files=DATA_FILES)
""",
        },
    )

    finding = scan_path(project).findings[0]

    assert finding.repo_id is None
    assert finding.unresolved_reason == (
        "load_dataset packaged builder 'json' does not have confidently local "
        "data_files"
    )


def test_namespaced_dataset_with_data_files_remains_a_hub_finding(
    tmp_path: Path,
) -> None:
    project = write_project(
        tmp_path,
        {"hub_dataset.py": 'load_dataset("org/data", data_files="train.jsonl")\n'},
    )

    findings = scan_path(project).findings

    assert [(finding.call_kind, finding.repo_id) for finding in findings] == [
        (CallKind.LOAD_DATASET, "org/data")
    ]


def test_new_matchers_report_dynamic_ids_and_ignore_unsupported_pipeline_forms(
    tmp_path: Path,
) -> None:
    project = write_project(
        tmp_path,
        {
            "dynamic.py": """\
pipeline("text-classification")
pipeline("text-classification", "org/positional")
pipeline("text-classification", model=choose_model())
SentenceTransformer(CONFIG["model"])
PeftModel.from_pretrained(base_model, model_id=f"org/{adapter}")
""",
        },
    )

    findings = scan_path(project).findings

    assert [finding.call_kind for finding in findings] == [
        CallKind.PIPELINE,
        CallKind.SENTENCE_TRANSFORMER,
        CallKind.PEFT_FROM_PRETRAINED,
    ]
    assert [finding.unresolved_reason for finding in findings] == [
        "repository ID is returned by a function call",
        "repository ID is a subscript expression",
        "repository ID is an interpolated string",
    ]


def test_peft_precedes_generic_matcher_and_local_paths_remain_ignored(
    tmp_path: Path,
) -> None:
    project = write_project(
        tmp_path,
        {
            "boundaries.py": """\
peft.PeftModel.from_pretrained(base_model, model_id="org/adapter")
PeftModel.from_pretrained(base_model, "./local-adapter")
SentenceTransformer("../local-sentence")
pipeline(model="C:\\\\models\\\\local")
OtherPeftModel.from_pretrained(base_model, "org/not-an-adapter")
""",
        },
    )

    findings = scan_path(project).findings

    assert findings[0].call_kind is CallKind.PEFT_FROM_PRETRAINED
    assert findings[0].repo_id == "org/adapter"
    assert findings[1].call_kind is CallKind.FROM_PRETRAINED
    assert findings[1].repo_id is None


def test_generic_diffusers_calls_keep_revision_and_trust_extraction(
    tmp_path: Path,
) -> None:
    # Matchers intentionally use terminal names without import resolution. This
    # covers common Diffusers calls but can match project-defined names too.
    project = write_project(
        tmp_path,
        {
            "diffusers.py": """\
MODEL_ID = "org/diffusion"
DiffusionPipeline.from_pretrained(MODEL_ID, revision="diff-v1")
diffusers.StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path="org/stable", trust_remote_code=True
)
""",
        },
    )

    findings = scan_path(project).findings

    assert [finding.call_kind for finding in findings] == [
        CallKind.FROM_PRETRAINED,
        CallKind.FROM_PRETRAINED,
    ]
    assert [finding.repo_id for finding in findings] == [
        "org/diffusion",
        "org/stable",
    ]
    assert findings[0].requested_revision == "diff-v1"
    assert findings[1].trust_remote_code is True


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

    result = scan_path(project)
    findings = result.findings

    assert [finding.repo_id for finding in findings] == [None, None, None]
    assert [finding.unresolved_reason for finding in findings] == [
        "repository ID is an interpolated string",
        "repository ID is returned by a function call",
        "repository ID name 'MODEL_ID' does not have one unambiguous string assignment",
    ]
    assert result.diagnostics[0].code == "MISSING_ENVIRONMENT_BINDING"


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


def test_configured_scan_uses_root_relative_posix_paths_for_narrow_scope(
    tmp_path: Path,
) -> None:
    project = write_project(
        tmp_path,
        {
            "pyproject.toml": '[tool.hf-freeze]\ninclude = ["src/**/*.py"]\n',
            "src/nested/app.py": 'AutoModel.from_pretrained("org/model")\n',
            "src/other.py": 'AutoModel.from_pretrained("org/other")\n',
        },
    )
    context = resolve_project_context(project / "src" / "nested")

    result = scan_path(project / "src" / "nested", context=context)

    assert [(item.source.path, item.repo_id) for item in result.findings] == [
        ("src/nested/app.py", "org/model")
    ]


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
