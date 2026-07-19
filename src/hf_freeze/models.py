"""Domain objects shared by hf-freeze commands."""

from dataclasses import dataclass
from enum import Enum


class CallKind(str, Enum):
    """Supported source call shapes."""

    FROM_PRETRAINED = "from_pretrained"
    LOAD_DATASET = "load_dataset"
    HF_HUB_DOWNLOAD = "hf_hub_download"
    SNAPSHOT_DOWNLOAD = "snapshot_download"


class RepoType(str, Enum):
    """Hub repository types the scanner can identify reliably."""

    MODEL = "model"
    DATASET = "dataset"


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


@dataclass(frozen=True)
class ScanDiagnostic:
    """A file-specific problem that did not stop the wider scan."""

    source: SourceLocation
    message: str


@dataclass(frozen=True)
class ScanResult:
    """Deterministically ordered findings and recoverable diagnostics."""

    findings: tuple[DependencyFinding, ...]
    diagnostics: tuple[ScanDiagnostic, ...]
