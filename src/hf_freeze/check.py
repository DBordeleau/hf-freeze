"""Pure, offline validation of source findings against a parsed lockfile."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from hf_freeze.models import (
    CALL_KIND_TO_DEPENDENCY_KIND,
    DependencyFinding,
    DependencyKind,
    LockedDependency,
    Lockfile,
    RepoType,
    ScanResult,
    SourceLocation,
)

_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")


class IssueSeverity(str, Enum):
    """Severity used to determine frozen-check exit status."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class CheckIssue:
    """One deterministic, actionable frozen-check result."""

    severity: IssueSeverity
    code: str
    message: str
    remediation: str
    source: SourceLocation | None = None


@dataclass(frozen=True)
class CheckOptions:
    """Policy switches understood by the checker."""

    frozen: bool = True


Identity = tuple[RepoType, str, DependencyKind]


def check_lockfile(
    scan_result: ScanResult,
    lockfile: Lockfile,
    options: CheckOptions = CheckOptions(),
) -> tuple[CheckIssue, ...]:
    """Return deterministic issues without filesystem or network access."""

    if not options.frozen:
        raise ValueError("non-frozen checking is not supported")

    issues: list[CheckIssue] = []
    for diagnostic in scan_result.diagnostics:
        issues.append(
            _source_issue(
                "SCAN_DIAGNOSTIC",
                diagnostic.message,
                "Fix the source file so it can be scanned, then rerun the check.",
                diagnostic.source,
            )
        )

    locked: dict[Identity, list[LockedDependency]] = {}
    for dependency in lockfile.dependencies:
        identity = (dependency.repo_type, dependency.repo_id, dependency.kind)
        locked.setdefault(identity, []).append(dependency)
        if not is_commit_sha(dependency.sha):
            issues.append(
                CheckIssue(
                    IssueSeverity.ERROR,
                    "INVALID_LOCKED_SHA",
                    f"locked dependency {_identity_text(identity)} has invalid SHA "
                    f"'{dependency.sha}'",
                    "Regenerate hf.lock so this dependency has a full 40-character "
                    "commit SHA.",
                )
            )

    ambiguous = {identity for identity, entries in locked.items() if len(entries) > 1}
    for identity in ambiguous:
        issues.append(
            CheckIssue(
                IssueSeverity.ERROR,
                "DUPLICATE_LOCK_IDENTITY",
                f"hf.lock contains multiple entries for {_identity_text(identity)}",
                "Remove duplicate entries or regenerate hf.lock.",
            )
        )

    findings: dict[Identity, list[DependencyFinding]] = {}
    used: set[Identity] = set()
    for finding in scan_result.findings:
        unresolved = _unresolved_reason(finding)
        identity = None if unresolved is not None else _finding_identity(finding)
        entries = [] if identity is None else locked.get(identity, [])
        trust_issue = _check_trust_remote_code(finding, entries)
        if trust_issue is not None:
            issues.append(trust_issue)
        if unresolved is not None:
            issues.append(
                _source_issue(
                    "UNRESOLVED_DEPENDENCY",
                    unresolved,
                    "Replace the dynamic value with one lexical-scope-safe literal "
                    "constant.",
                    finding.source,
                )
            )
            continue

        assert identity is not None
        findings.setdefault(identity, []).append(finding)
        if not entries:
            issues.append(
                _source_issue(
                    "MISSING_LOCK_ENTRY",
                    f"{_identity_text(identity)} is missing from hf.lock",
                    "Run 'hf-freeze lock' and commit the updated hf.lock.",
                    finding.source,
                )
            )
        elif identity not in ambiguous:
            used.add(identity)
            issues.extend(_check_finding(finding, entries[0]))

    for identity, grouped in findings.items():
        revisions = {_requested_revision(item) for item in grouped}
        if len(revisions) > 1:
            rendered = ", ".join(repr(item) for item in sorted(revisions))
            for finding in grouped:
                issues.append(
                    _source_issue(
                        "CONFLICTING_REVISIONS",
                        f"{_identity_text(identity)} is requested with conflicting "
                        f"revisions: {rendered}",
                        "Use the same full 40-character commit SHA at every call site.",
                        finding.source,
                    )
                )

    for identity in locked.keys() - used - ambiguous:
        issues.append(
            CheckIssue(
                IssueSeverity.WARNING,
                "UNUSED_LOCK_ENTRY",
                f"{_identity_text(identity)} is no longer used by scanned source",
                "Remove the stale entry by regenerating hf.lock.",
            )
        )
    return tuple(sorted(issues, key=_issue_key))


def is_commit_sha(value: str) -> bool:
    """Return whether value is an immutable full commit SHA."""

    return _COMMIT_SHA.fullmatch(value) is not None


def check_remote_code_without_lock(
    scan_result: ScanResult,
) -> tuple[CheckIssue, ...]:
    """Check only remote-code policy when no lockfile can provide coverage."""

    issues = (
        issue
        for finding in scan_result.findings
        if (issue := _check_trust_remote_code(finding, [])) is not None
    )
    return tuple(sorted(issues, key=_issue_key))


def _check_finding(
    finding: DependencyFinding, locked: LockedDependency
) -> list[CheckIssue]:
    issues: list[CheckIssue] = []
    revision = finding.requested_revision
    if revision is None or not is_commit_sha(revision):
        issues.append(
            _source_issue(
                "FLOATING_REVISION",
                "source revision is omitted or is not a full 40-character commit SHA",
                f"Set revision='{locked.sha}' at this call site.",
                finding.source,
            )
        )
    elif revision != locked.sha:
        issues.append(
            _source_issue(
                "SHA_MISMATCH",
                f"source revision '{revision}' differs from locked SHA '{locked.sha}'",
                f"Set revision='{locked.sha}' or intentionally refresh hf.lock.",
                finding.source,
            )
        )

    return issues


def _check_trust_remote_code(
    finding: DependencyFinding, locked: list[LockedDependency]
) -> CheckIssue | None:
    if finding.trust_remote_code_unresolved_reason is not None:
        return _source_issue(
            "DYNAMIC_TRUST_REMOTE_CODE",
            finding.trust_remote_code_unresolved_reason,
            "Set trust_remote_code to literal False, or use literal True with "
            "an exact locked SHA.",
            finding.source,
        )
    if not finding.trust_remote_code:
        return None

    revision = finding.requested_revision
    covered = (
        len(locked) == 1
        and revision is not None
        and is_commit_sha(revision)
        and is_commit_sha(locked[0].sha)
        and revision == locked[0].sha
    )
    if covered:
        return None
    return _source_issue(
        "UNPINNED_REMOTE_CODE",
        "trust_remote_code=True is not covered by an exact source and lock SHA",
        "Add one matching lock entry with a valid SHA, pin that exact revision, "
        "or disable trust_remote_code.",
        finding.source,
    )


def _unresolved_reason(finding: DependencyFinding) -> str | None:
    if finding.repo_id is None:
        return finding.unresolved_reason or "repository ID is unresolved"
    if finding.repo_type is None:
        return "repository type is unresolved"
    if finding.revision_unresolved_reason is not None:
        return finding.revision_unresolved_reason
    return None


def _finding_identity(finding: DependencyFinding) -> Identity:
    assert finding.repo_id is not None and finding.repo_type is not None
    return (
        finding.repo_type,
        finding.repo_id,
        CALL_KIND_TO_DEPENDENCY_KIND[finding.call_kind],
    )


def _requested_revision(finding: DependencyFinding) -> str:
    return finding.requested_revision or "main"


def _identity_text(identity: Identity) -> str:
    repo_type, repo_id, kind = identity
    return f"{repo_type.value} '{repo_id}' ({kind.value})"


def _source_issue(
    code: str, message: str, remediation: str, source: SourceLocation
) -> CheckIssue:
    return CheckIssue(IssueSeverity.ERROR, code, message, remediation, source)


def _issue_key(issue: CheckIssue) -> tuple[str, int, int, str, str, str]:
    source = issue.source or SourceLocation("", 0, 0)
    return (
        source.path,
        source.line,
        source.column,
        issue.severity.value,
        issue.code,
        issue.message,
    )
