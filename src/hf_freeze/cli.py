"""Command-line interface for hf-freeze."""

from pathlib import Path

import typer

from hf_freeze import __version__
from hf_freeze.check import (
    CheckIssue,
    IssueSeverity,
    check_lockfile,
    check_remote_code_without_lock,
)
from hf_freeze.config import (
    ConfigError,
    ProjectContext,
    iter_scoped_python_files,
    resolve_project_context,
)
from hf_freeze.diff import (
    DiffError,
    preview_locked_dependency,
    select_locked_dependency,
)
from hf_freeze.hub import HfHubResolver, HubResolutionError
from hf_freeze.lockfile import (
    LockError,
    read_lockfile,
    resolve_lockfile,
    select_repository_entries,
    update_repository_entries,
    write_lockfile,
)
from hf_freeze.models import (
    DependencyFinding,
    DiagnosticSeverity,
    ScanDiagnostic,
    ScanResult,
)
from hf_freeze.rewrite import apply_rewrite_plan, plan_rewrites
from hf_freeze.scan import scan_path

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Freeze Hugging Face Hub dependencies."""


@app.command()
def version() -> None:
    """Show the installed hf-freeze version."""
    typer.echo(f"hf-freeze {__version__}")


@app.command()
def scan(path: Path | None = typer.Argument(None)) -> None:
    """Discover supported Hugging Face Hub calls in Python source."""
    context, requested_path = _resolve_scope(path)
    try:
        result = scan_path(requested_path, context=context)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="PATH") from error

    entries: list[DependencyFinding | ScanDiagnostic] = [
        *result.findings,
        *result.diagnostics,
    ]
    entries.sort(
        key=lambda item: (item.source.path, item.source.line, item.source.column)
    )
    for entry in entries:
        location = f"{entry.source.path}:{entry.source.line}:{entry.source.column + 1}"
        if isinstance(entry, ScanDiagnostic):
            severity = (
                f"{entry.severity.value.upper()} {entry.code}  "
                if entry.severity is DiagnosticSeverity.WARNING
                else ""
            )
            typer.echo(f"{location}  {severity}{entry.message}")
        elif entry.repo_id is None:
            typer.echo(
                f"{location}  {entry.call_kind.value}  unresolved: "
                f"{entry.unresolved_reason}"
            )
        else:
            repo_type = entry.repo_type.value if entry.repo_type else "unknown"
            if entry.revision_unresolved_reason is not None:
                revision = f"<unresolved: {entry.revision_unresolved_reason}>"
            else:
                revision = entry.requested_revision or "<default>"
            typer.echo(
                f"{location}  {entry.call_kind.value}  {repo_type}  "
                f"{entry.repo_id}  revision={revision}"
            )


@app.command()
def lock(path: Path | None = typer.Argument(None)) -> None:
    """Resolve discovered Hub dependencies and atomically write hf.lock."""
    context, requested_path = _resolve_scope(path)
    try:
        result = scan_path(requested_path, context=context)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="PATH") from error

    destination = _source_lock_path(context, requested_path)
    _render_scan_warnings(result)
    try:
        existing_lockfile = read_lockfile(destination) if destination.exists() else None
        lockfile = resolve_lockfile(
            result,
            HfHubResolver(),
            existing_lockfile=existing_lockfile,
        )
        write_lockfile(destination, lockfile)
    except (LockError, HubResolutionError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Wrote {destination}")


@app.command()
def check(
    path: Path | None = typer.Argument(None),
    frozen: bool = typer.Option(False, "--frozen"),
) -> None:
    """Check that source is immutably covered by hf.lock without network access."""
    if not frozen:
        raise typer.BadParameter("required for this command", param_hint="--frozen")
    context, requested_path = _resolve_scope(path)
    try:
        result = scan_path(requested_path, context=context)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="PATH") from error

    destination = _source_lock_path(context, requested_path)
    try:
        if not destination.exists():
            issues = _unusable_lock_issues(
                result, "MISSING_LOCKFILE", "hf.lock does not exist"
            )
        else:
            lockfile = read_lockfile(destination)
            issues = check_lockfile(result, lockfile)
    except LockError as error:
        code = (
            "UNSUPPORTED_LOCK_VERSION"
            if "unsupported lockfile version" in str(error)
            else "MALFORMED_LOCKFILE"
        )
        issues = _unusable_lock_issues(result, code, str(error))
    except OSError as error:
        issues = _unusable_lock_issues(result, "LOCKFILE_READ_ERROR", str(error))

    for issue in issues:
        _render_check_issue(issue)
    if any(issue.severity is IssueSeverity.ERROR for issue in issues):
        raise typer.Exit(code=1)


@app.command(name="diff")
def diff_command(
    repo_id: str = typer.Argument(...),
    revision: str | None = typer.Option(None, "--revision"),
) -> None:
    """Compare a locked commit with a candidate Hub revision."""

    context = _resolve_context(Path.cwd())
    try:
        lockfile = read_lockfile(_cwd_lock_path(context))
        locked = select_locked_dependency(lockfile, repo_id)
    except (LockError, DiffError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    candidate_revision = revision if revision is not None else locked.requested_revision
    resolver = HfHubResolver()
    try:
        preview = preview_locked_dependency(locked, candidate_revision, resolver)
    except (HubResolutionError, DiffError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(preview.rendered)


@app.command()
def update(
    repo_id: str = typer.Argument(...),
    revision: str | None = typer.Option(None, "--revision"),
    write: bool = typer.Option(False, "--write"),
) -> None:
    """Preview or atomically accept one repository update into hf.lock."""

    context = _resolve_context(Path.cwd())
    destination = _cwd_lock_path(context)
    try:
        lockfile = read_lockfile(destination)
        locked = select_repository_entries(lockfile, repo_id)[0]
    except (LockError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    candidate_revision = revision if revision is not None else locked.requested_revision
    try:
        preview = preview_locked_dependency(locked, candidate_revision, HfHubResolver())
        updated = update_repository_entries(
            lockfile, repo_id, candidate_revision, preview.candidate_sha
        )
    except (HubResolutionError, DiffError, LockError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(preview.rendered)
    typer.echo(
        "\nProposed hf.lock change:\n"
        f"  requested_revision: {locked.requested_revision} -> {candidate_revision}\n"
        f"  sha: {locked.sha} -> {preview.candidate_sha}"
    )
    effective_change = (
        candidate_revision != locked.requested_revision
        or preview.candidate_sha != locked.sha
    )
    if not effective_change:
        typer.echo("\nhf.lock is already current; no file was replaced.")
        return
    if not write:
        typer.echo("\nDry run; pass --write to update hf.lock.")
        return
    try:
        write_lockfile(destination, updated)
    except (LockError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        "\nWrote hf.lock.\n"
        "Next run 'hf-freeze pin --write', then 'hf-freeze check --frozen'."
    )


@app.command()
def pin(
    path: Path | None = typer.Argument(None),
    write: bool = typer.Option(False, "--write"),
) -> None:
    """Preview or atomically apply exact locked revisions to supported calls."""

    context, requested_path = _resolve_scope(path)
    try:
        result = scan_path(requested_path, context=context)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="PATH") from error

    fatal_diagnostics = _fatal_scan_diagnostics(result)
    if fatal_diagnostics:
        for diagnostic in fatal_diagnostics:
            _render_scan_error(diagnostic)
        raise typer.Exit(code=1)
    _render_scan_warnings(result)

    if context.config_path is None:
        root = requested_path if requested_path.is_dir() else requested_path.parent
        source_filter = (
            None if requested_path.is_dir() else frozenset({requested_path.name})
        )
    else:
        root = context.root
        source_filter = frozenset(
            display_path
            for _, display_path in iter_scoped_python_files(context, requested_path)
        )
    try:
        lockfile = read_lockfile(_source_lock_path(context, requested_path))
    except (LockError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    plan = plan_rewrites(root, lockfile, result, source_filter=source_filter)
    write_skips = ()
    if write:
        written, write_skips = apply_rewrite_plan(root, plan)
        for changed_path in written:
            typer.echo(f"Wrote {changed_path}")
    else:
        for change in plan.changes:
            typer.echo(change.diff, nl=False)

    for noop in plan.noops:
        target = noop.target
        typer.echo(f"Already pinned {target.path}:{target.line}:{target.column + 1}")
    for skipped in (*plan.skipped, *write_skips):
        column = "" if skipped.column is None else f":{skipped.column + 1}"
        typer.echo(
            f"Skipped {skipped.path}:{skipped.line}{column}: {skipped.reason}",
            err=True,
        )
    if plan.skipped or write_skips:
        raise typer.Exit(code=1)


def _resolve_scope(path: Path | None) -> tuple[ProjectContext, Path]:
    requested_path = path if path is not None else Path(".")
    context = _resolve_context(requested_path)
    if path is None and context.config_path is not None:
        requested_path = context.root
    return context, requested_path


def _resolve_context(path: Path) -> ProjectContext:
    try:
        return resolve_project_context(path)
    except ConfigError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error


def _source_lock_path(context: ProjectContext, requested_path: Path) -> Path:
    if context.config_path is not None:
        return context.root / "hf.lock"
    root = requested_path if requested_path.is_dir() else requested_path.parent
    return root / "hf.lock"


def _cwd_lock_path(context: ProjectContext) -> Path:
    return (
        context.root / "hf.lock" if context.config_path is not None else Path("hf.lock")
    )


def _lock_issue(code: str, message: str) -> CheckIssue:
    return CheckIssue(
        IssueSeverity.ERROR,
        code,
        message,
        "Create or repair hf.lock with 'hf-freeze lock'.",
    )


def _unusable_lock_issues(
    result: ScanResult, code: str, message: str
) -> tuple[CheckIssue, ...]:
    return (
        _lock_issue(code, message),
        *(
            CheckIssue(
                IssueSeverity.WARNING
                if diagnostic.severity is DiagnosticSeverity.WARNING
                else IssueSeverity.ERROR,
                diagnostic.code,
                diagnostic.message,
                "Remove the unused declaration or bind it to a source call."
                if diagnostic.code == "UNUSED_DECLARATION"
                else "Fix the source directive or file, then rerun the check.",
                diagnostic.source,
            )
            for diagnostic in result.diagnostics
        ),
        *check_remote_code_without_lock(result),
    )


def _fatal_scan_diagnostics(result: ScanResult) -> tuple[ScanDiagnostic, ...]:
    return tuple(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.severity is DiagnosticSeverity.ERROR
    )


def _render_scan_warnings(result: ScanResult) -> None:
    for diagnostic in result.diagnostics:
        if diagnostic.severity is DiagnosticSeverity.WARNING:
            location = (
                f"{diagnostic.source.path}:{diagnostic.source.line}:"
                f"{diagnostic.source.column + 1}"
            )
            typer.echo(
                f"{location}  WARNING {diagnostic.code}  {diagnostic.message}",
                err=True,
            )


def _render_scan_error(diagnostic: ScanDiagnostic) -> None:
    location = (
        f"{diagnostic.source.path}:{diagnostic.source.line}:"
        f"{diagnostic.source.column + 1}"
    )
    typer.echo(f"{location}  ERROR {diagnostic.code}  {diagnostic.message}", err=True)


def _render_check_issue(issue: CheckIssue) -> None:
    prefix = ""
    if issue.source is not None:
        prefix = f"{issue.source.path}:{issue.source.line}:{issue.source.column + 1}  "
    typer.echo(
        f"{prefix}{issue.severity.value.upper()} {issue.code}  {issue.message}  "
        f"Fix: {issue.remediation}"
    )
