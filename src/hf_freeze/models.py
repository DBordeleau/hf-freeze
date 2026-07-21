"""Domain objects shared by hf-freeze commands."""

from dataclasses import dataclass
from enum import Enum


class CallKind(str, Enum):
    """Supported source call shapes."""

    FROM_PRETRAINED = "from_pretrained"
    LOAD_DATASET = "load_dataset"
    HF_HUB_DOWNLOAD = "hf_hub_download"
    SNAPSHOT_DOWNLOAD = "snapshot_download"
    PIPELINE = "pipeline"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    PEFT_FROM_PRETRAINED = "peft_from_pretrained"


class RepoType(str, Enum):
    """Hub repository types the scanner can identify reliably."""

    MODEL = "model"
    DATASET = "dataset"


class DiagnosticSeverity(str, Enum):
    """Whether a scan diagnostic blocks lifecycle commands."""

    ERROR = "error"
    WARNING = "warning"


class DependencyKind(str, Enum):
    """Dependency categories stored in schema-v1 lockfiles."""

    MODEL = "model"
    DATASET = "dataset"
    DIRECT_FILE = "direct_file"
    SNAPSHOT = "snapshot"
    ADAPTER = "adapter"


class CoverageKind(str, Enum):
    """Stable source-call coverage identifiers in deterministic display order."""

    LOCKED_STATIC = "LOCKED_STATIC"
    LOCKED_ENV_BINDING = "LOCKED_ENV_BINDING"
    LOCKED_ANNOTATION = "LOCKED_ANNOTATION"
    ACKNOWLEDGED_DYNAMIC = "ACKNOWLEDGED_DYNAMIC"
    UNRESOLVED = "UNRESOLVED"


COVERAGE_ORDER = tuple(CoverageKind)


CALL_KIND_TO_DEPENDENCY_KIND = {
    CallKind.FROM_PRETRAINED: DependencyKind.MODEL,
    CallKind.LOAD_DATASET: DependencyKind.DATASET,
    CallKind.HF_HUB_DOWNLOAD: DependencyKind.DIRECT_FILE,
    CallKind.SNAPSHOT_DOWNLOAD: DependencyKind.SNAPSHOT,
    CallKind.PIPELINE: DependencyKind.MODEL,
    CallKind.SENTENCE_TRANSFORMER: DependencyKind.MODEL,
    CallKind.PEFT_FROM_PRETRAINED: DependencyKind.ADAPTER,
}


@dataclass(frozen=True)
class SourceLocation:
    """A zero-based column and one-based line in a scanned source file."""

    path: str
    line: int
    column: int


@dataclass(frozen=True)
class DependencyFinding:
    """A supported Hub call, whether statically resolved or not."""

    repo_id: str | None
    repo_type: RepoType | None
    call_kind: CallKind
    requested_revision: str | None
    source: SourceLocation
    unresolved_reason: str | None = None
    revision_unresolved_reason: str | None = None
    trust_remote_code: bool = False
    trust_remote_code_unresolved_reason: str | None = None


@dataclass(frozen=True)
class AcknowledgedDynamicFinding:
    """One supported call explicitly kept outside the frozen guarantee."""

    call_kind: CallKind
    source: SourceLocation
    reason: str


@dataclass(frozen=True)
class ScanDiagnostic:
    """A file-specific problem that did not stop the wider scan."""

    source: SourceLocation
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    code: str = "SCAN_DIAGNOSTIC"


@dataclass(frozen=True)
class CallCoverage:
    """Exactly one coverage classification for one supported source call site."""

    kind: CoverageKind
    call_kind: CallKind
    source: SourceLocation


@dataclass(frozen=True)
class ScanResult:
    """Deterministically ordered findings and recoverable diagnostics."""

    findings: tuple[DependencyFinding, ...]
    diagnostics: tuple[ScanDiagnostic, ...]
    acknowledged: tuple[AcknowledgedDynamicFinding, ...] = ()
    coverage: tuple[CallCoverage, ...] = ()


def coverage_counts(result: ScanResult) -> tuple[tuple[CoverageKind, int], ...]:
    """Count source call sites by stable coverage category."""

    records = result.coverage
    if not records:
        records = tuple(
            CallCoverage(_finding_coverage(finding), finding.call_kind, finding.source)
            for finding in result.findings
        ) + tuple(
            CallCoverage(CoverageKind.ACKNOWLEDGED_DYNAMIC, item.call_kind, item.source)
            for item in result.acknowledged
        )
    return tuple(
        (kind, sum(record.kind is kind for record in records))
        for kind in COVERAGE_ORDER
    )


def _finding_coverage(finding: DependencyFinding) -> CoverageKind:
    if (
        finding.repo_id is None
        or finding.repo_type is None
        or finding.revision_unresolved_reason is not None
    ):
        return CoverageKind.UNRESOLVED
    return CoverageKind.LOCKED_STATIC


@dataclass(frozen=True)
class LockedSource:
    """A deterministic, project-relative source reference."""

    path: str
    line: int
    call: CallKind


@dataclass(frozen=True)
class LockedDependency:
    """One immutable Hub dependency in a lockfile."""

    repo_id: str
    repo_type: RepoType
    kind: DependencyKind
    requested_revision: str
    sha: str
    sources: tuple[LockedSource, ...]


@dataclass(frozen=True)
class Lockfile:
    """The supported schema-v1 lockfile."""

    version: int
    dependencies: tuple[LockedDependency, ...]
