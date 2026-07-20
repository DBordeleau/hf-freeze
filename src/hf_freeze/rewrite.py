"""Safe, formatting-preserving source pin planning and atomic application."""

from __future__ import annotations

import difflib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from hf_freeze.check import is_commit_sha
from hf_freeze.models import (
    CALL_KIND_TO_DEPENDENCY_KIND,
    CallKind,
    DependencyFinding,
    DiagnosticSeverity,
    LockedDependency,
    LockedSource,
    Lockfile,
    ScanResult,
)
from hf_freeze.scan import match_call

_GroupKey = tuple[str, str, str, str, str, str]


@dataclass(frozen=True)
class RewriteTarget:
    path: str
    line: int
    column: int
    call_kind: CallKind
    repo_id: str
    sha: str


@dataclass(frozen=True)
class PlannedFileChange:
    path: str
    before: bytes
    after: bytes
    mode: int
    diff: str
    targets: tuple[RewriteTarget, ...]


@dataclass(frozen=True)
class NoOpTarget:
    target: RewriteTarget


@dataclass(frozen=True)
class SkippedTarget:
    path: str
    line: int
    column: int | None
    call_kind: CallKind | None
    reason: str


@dataclass(frozen=True)
class RewritePlan:
    changes: tuple[PlannedFileChange, ...]
    noops: tuple[NoOpTarget, ...]
    skipped: tuple[SkippedTarget, ...]


class _TargetTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, targets: dict[tuple[int, int], RewriteTarget]) -> None:
        self.targets = targets
        self.changed: list[RewriteTarget] = []
        self.noops: list[RewriteTarget] = []
        self.skipped: list[SkippedTarget] = []
        self.seen: set[tuple[int, int]] = set()

    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.BaseExpression:
        position = self.get_metadata(PositionProvider, original_node).start
        key = (position.line, position.column)
        target = self.targets.get(key)
        if target is None:
            return updated_node
        self.seen.add(key)

        spec = match_call(original_node.func)
        if spec is None or spec.kind is not target.call_kind:
            self._skip(target, "call kind changed after scanning")
            return updated_node
        if any(argument.star in {"*", "**"} for argument in original_node.args):
            self._skip(target, "call contains *args or **kwargs")
            return updated_node

        revisions = [
            (index, argument)
            for index, argument in enumerate(original_node.args)
            if argument.keyword is not None and argument.keyword.value == "revision"
        ]
        if len(revisions) > 1:
            self._skip(target, "call contains duplicate revision= keywords")
            return updated_node
        if revisions:
            index, argument = revisions[0]
            if not isinstance(argument.value, cst.SimpleString):
                self._skip(target, "revision= is not a direct string literal")
                return updated_node
            try:
                value = argument.value.evaluated_value
            except ValueError:
                value = None
            if not isinstance(value, str):
                self._skip(target, "revision= is not a direct string literal")
                return updated_node
            if value == target.sha:
                self.noops.append(target)
                return updated_node
            replacement = cst.SimpleString(
                f"{argument.value.prefix}{argument.value.quote}"
                f"{target.sha}{argument.value.quote}"
            )
            arguments = list(updated_node.args)
            arguments[index] = arguments[index].with_changes(value=replacement)
            self.changed.append(target)
            return updated_node.with_changes(args=arguments)

        self.changed.append(target)
        return _append_revision(updated_node, target.sha)

    def _skip(self, target: RewriteTarget, reason: str) -> None:
        self.skipped.append(_skipped(target, reason))


def plan_rewrites(
    root: Path,
    lockfile: Lockfile,
    scan_result: ScanResult,
    *,
    source_filter: frozenset[str] | None = None,
) -> RewritePlan:
    """Correlate lock sources to exact scanner columns and prepare safe outputs."""

    root = Path(root)
    findings = [
        finding
        for finding in scan_result.findings
        if source_filter is None or finding.source.path in source_filter
    ]
    locked = _locked_targets(lockfile, source_filter)
    lock_targets: dict[tuple[str, int, CallKind], set[tuple[object, ...]]] = {}
    for dependency, source in locked:
        key = (source.path, source.line, source.call)
        lock_targets.setdefault(key, set()).add(
            (dependency.repo_id, dependency.repo_type, dependency.kind, dependency.sha)
        )
    conflicts = {key for key, values in lock_targets.items() if len(values) > 1}
    skipped: list[SkippedTarget] = []
    correlated: list[RewriteTarget] = []
    reconciled_noops: list[NoOpTarget] = []
    unmatched: list[tuple[LockedDependency, LockedSource, str]] = []
    covered_findings: set[int] = set()
    diagnostics = {
        diagnostic.source.path: diagnostic
        for diagnostic in scan_result.diagnostics
        if diagnostic.severity is DiagnosticSeverity.ERROR
    }

    for dependency, source in locked:
        destination = root / Path(source.path)
        if (source.path, source.line, source.call) in conflicts:
            skipped.append(
                _skip_source(source, "conflicting lock targets name this location")
            )
            continue
        if not is_commit_sha(dependency.sha):
            skipped.append(
                _skip_source(source, "locked SHA is not a full 40-character commit SHA")
            )
            continue
        if destination.is_symlink():
            skipped.append(_skip_source(source, "source path is a symlink"))
            continue
        if not destination.is_file():
            skipped.append(_skip_source(source, "source file is missing"))
            continue
        if source.path in diagnostics:
            skipped.append(
                _skip_source(
                    source,
                    f"source cannot be scanned: {diagnostics[source.path].message}",
                )
            )
            continue

        same_location = [
            (index, finding)
            for index, finding in enumerate(findings)
            if finding.source.path == source.path
            and finding.source.line == source.line
            and finding.call_kind is source.call
        ]
        compatible = [
            (index, finding)
            for index, finding in same_location
            if _matches_dependency(finding, dependency)
        ]
        if len(compatible) > 1:
            covered_findings.update(index for index, _ in compatible)
            skipped.append(
                _skip_source(
                    source, "multiple compatible calls exist at the recorded location"
                )
            )
            continue
        if compatible:
            index, finding = compatible[0]
            covered_findings.add(index)
            correlated.append(_target(dependency, finding))
        else:
            reason = (
                "repository identity or dependency kind changed"
                if same_location
                else "no supported call remains at the recorded location"
            )
            unmatched.append((dependency, source, reason))

    unmatched_groups: dict[
        _GroupKey, list[tuple[LockedDependency, LockedSource, str]]
    ] = {}
    for dependency, source, reason in unmatched:
        unmatched_groups.setdefault(_lock_group(dependency, source), []).append(
            (dependency, source, reason)
        )
    eligible_groups: dict[_GroupKey, list[tuple[int, DependencyFinding]]] = {}
    for index, finding in enumerate(findings):
        if index in covered_findings:
            continue
        key = _finding_group(finding)
        if key is not None:
            eligible_groups.setdefault(key, []).append((index, finding))

    for key in sorted(unmatched_groups):
        locked_group = unmatched_groups[key]
        available = eligible_groups.get(key, [])
        if available and len(locked_group) == len(available):
            for (dependency, _, _), (index, finding) in zip(
                locked_group, available, strict=True
            ):
                covered_findings.add(index)
                reconciled_noops.append(NoOpTarget(_target(dependency, finding)))
            continue
        count_reason = (
            f"found {len(available)} uncovered exact-SHA call(s) for "
            f"{len(locked_group)} unmatched lock source(s)"
        )
        for _, source, source_reason in locked_group:
            skipped.append(
                _skip_source(source, count_reason if available else source_reason)
            )

    targets: dict[tuple[str, int, int], RewriteTarget] = {}
    for target in correlated:
        key = (target.path, target.line, target.column)
        targets.setdefault(key, target)

    for index, finding in enumerate(findings):
        if index not in covered_findings:
            reason = (
                finding.unresolved_reason or "current call has no matching lock source"
            )
            skipped.append(
                SkippedTarget(
                    finding.source.path,
                    finding.source.line,
                    finding.source.column,
                    finding.call_kind,
                    reason,
                )
            )
    for diagnostic in scan_result.diagnostics:
        if diagnostic.severity is DiagnosticSeverity.WARNING:
            continue
        if source_filter is None or diagnostic.source.path in source_filter:
            skipped.append(
                SkippedTarget(
                    diagnostic.source.path,
                    diagnostic.source.line,
                    diagnostic.source.column,
                    None,
                    diagnostic.message,
                )
            )

    changes: list[PlannedFileChange] = []
    noops = reconciled_noops
    by_path: dict[str, dict[tuple[int, int], RewriteTarget]] = {}
    for (path, line, column), target in targets.items():
        by_path.setdefault(path, {})[(line, column)] = target
    for path in sorted(by_path):
        change, file_noops, file_skips = _plan_file(root, path, by_path[path])
        if change is not None:
            changes.append(change)
        noops.extend(NoOpTarget(target) for target in file_noops)
        skipped.extend(file_skips)

    return RewritePlan(
        changes=tuple(changes),
        noops=tuple(sorted(noops, key=lambda item: _target_key(item.target))),
        skipped=tuple(sorted(set(skipped), key=_skip_key)),
    )


def apply_rewrite_plan(
    root: Path, plan: RewritePlan
) -> tuple[tuple[str, ...], tuple[SkippedTarget, ...]]:
    """Atomically replace each unchanged source file with its prepared bytes."""

    written: list[str] = []
    skipped: list[SkippedTarget] = []
    for change in plan.changes:
        destination = Path(root) / Path(change.path)
        temporary_path: Path | None = None
        try:
            if destination.is_symlink():
                raise OSError("source path became a symlink before writing")
            if destination.read_bytes() != change.before:
                raise OSError("source changed after rewrite planning")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(change.after)
                temporary.flush()
            os.chmod(temporary_path, change.mode)
            os.replace(temporary_path, destination)
            temporary_path = None
            written.append(change.path)
        except OSError as error:
            skipped.append(
                SkippedTarget(change.path, 1, None, None, f"write failed: {error}")
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return tuple(written), tuple(skipped)


def _plan_file(
    root: Path, path: str, targets: dict[tuple[int, int], RewriteTarget]
) -> tuple[PlannedFileChange | None, list[RewriteTarget], list[SkippedTarget]]:
    destination = root / Path(path)
    try:
        before = destination.read_bytes()
        mode = stat.S_IMODE(destination.stat(follow_symlinks=False).st_mode)
        module = cst.parse_module(before)
    except (OSError, cst.ParserSyntaxError) as error:
        reason = f"source preparation failed: {error}"
        return None, [], [_skipped(target, reason) for target in targets.values()]

    transformer = _TargetTransformer(targets)
    updated = MetadataWrapper(module).visit(transformer)
    for key, target in targets.items():
        if key not in transformer.seen:
            transformer.skipped.append(
                _skipped(target, "exact scanner location no longer identifies the call")
            )
    after = updated.bytes
    change = None
    if transformer.changed:
        change = PlannedFileChange(
            path,
            before,
            after,
            mode,
            _unified_diff(path, before, after, module.encoding),
            tuple(sorted(transformer.changed, key=_target_key)),
        )
    return change, transformer.noops, transformer.skipped


def _append_revision(call: cst.Call, sha: str) -> cst.Call:
    equal = cst.AssignEqual(
        whitespace_before=cst.SimpleWhitespace(""),
        whitespace_after=cst.SimpleWhitespace(""),
    )
    new_argument = cst.Arg(
        value=cst.SimpleString(f'"{sha}"'),
        keyword=cst.Name("revision"),
        equal=equal,
    )
    if not call.args or isinstance(call.args[-1].comma, cst.MaybeSentinel):
        return call.with_changes(args=(*call.args, new_argument))

    arguments = list(call.args)
    previous = arguments[-1]
    comma = previous.comma
    assert isinstance(comma, cst.Comma)
    whitespace = comma.whitespace_after
    if isinstance(whitespace, cst.ParenthesizedWhitespace):
        indent = "    "
        if isinstance(call.whitespace_before_args, cst.ParenthesizedWhitespace):
            indent = call.whitespace_before_args.last_line.value
        previous_whitespace = whitespace.with_changes(
            last_line=cst.SimpleWhitespace(indent)
        )
        arguments[-1] = previous.with_changes(
            comma=comma.with_changes(whitespace_after=previous_whitespace)
        )
        closing_whitespace = cst.ParenthesizedWhitespace(
            first_line=cst.TrailingWhitespace(newline=whitespace.first_line.newline),
            indent=whitespace.indent,
            last_line=whitespace.last_line,
        )
        new_argument = new_argument.with_changes(
            comma=cst.Comma(whitespace_after=closing_whitespace)
        )
    else:
        arguments[-1] = previous.with_changes(
            comma=comma.with_changes(whitespace_after=cst.SimpleWhitespace(" "))
        )
        new_argument = new_argument.with_changes(comma=comma)
    arguments.append(new_argument)
    return call.with_changes(args=arguments)


def _locked_targets(
    lockfile: Lockfile, source_filter: frozenset[str] | None
) -> list[tuple[LockedDependency, LockedSource]]:
    targets = [
        (dependency, source)
        for dependency in lockfile.dependencies
        for source in dependency.sources
        if source_filter is None or source.path in source_filter
    ]
    return sorted(
        targets,
        key=lambda item: (
            item[1].path,
            item[1].line,
            item[1].call.value,
            item[0].repo_type.value,
            item[0].repo_id,
            item[0].kind.value,
            item[0].sha,
        ),
    )


def _matches_dependency(
    finding: DependencyFinding, dependency: LockedDependency
) -> bool:
    return (
        finding.repo_id == dependency.repo_id
        and finding.repo_type is dependency.repo_type
        and CALL_KIND_TO_DEPENDENCY_KIND[finding.call_kind] is dependency.kind
    )


def _lock_group(dependency: LockedDependency, source: LockedSource) -> _GroupKey:
    return (
        source.path,
        dependency.repo_id,
        dependency.repo_type.value,
        dependency.kind.value,
        dependency.sha,
        source.call.value,
    )


def _finding_group(finding: DependencyFinding) -> _GroupKey | None:
    if (
        finding.repo_id is None
        or finding.repo_type is None
        or finding.requested_revision is None
        or finding.revision_unresolved_reason is not None
    ):
        return None
    return (
        finding.source.path,
        finding.repo_id,
        finding.repo_type.value,
        CALL_KIND_TO_DEPENDENCY_KIND[finding.call_kind].value,
        finding.requested_revision,
        finding.call_kind.value,
    )


def _target(dependency: LockedDependency, finding: DependencyFinding) -> RewriteTarget:
    return RewriteTarget(
        finding.source.path,
        finding.source.line,
        finding.source.column,
        finding.call_kind,
        dependency.repo_id,
        dependency.sha,
    )


def _unified_diff(path: str, before: bytes, after: bytes, encoding: str) -> str:
    old_lines = before.decode(encoding).splitlines()
    new_lines = after.decode(encoding).splitlines()
    lines = difflib.unified_diff(
        old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
    )
    return "\n".join(lines) + "\n"


def _skip_source(source: LockedSource, reason: str) -> SkippedTarget:
    return SkippedTarget(source.path, source.line, None, source.call, reason)


def _skipped(target: RewriteTarget, reason: str) -> SkippedTarget:
    return SkippedTarget(
        target.path, target.line, target.column, target.call_kind, reason
    )


def _target_key(target: RewriteTarget) -> tuple[str, int, int, str]:
    return (target.path, target.line, target.column, target.call_kind.value)


def _skip_key(item: SkippedTarget) -> tuple[str, int, int, str, str]:
    return (
        item.path,
        item.line,
        -1 if item.column is None else item.column,
        "" if item.call_kind is None else item.call_kind.value,
        item.reason,
    )
