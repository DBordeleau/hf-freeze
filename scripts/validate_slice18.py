"""Reproduce Slice 18 validation without importing or executing target projects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

COVERAGE_ORDER = (
    "LOCKED_STATIC",
    "LOCKED_ENV_BINDING",
    "LOCKED_ANNOTATION",
    "ACKNOWLEDGED_DYNAMIC",
    "UNRESOLVED",
)
TOKEN_VARIABLES = frozenset(
    {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_API_TOKEN",
    }
)
REQUIRED_REPOSITORIES = frozenset(
    {
        "tloen/alpaca-lora",
        "louisbrulenaudet/tsdae",
        "cloneofsimo/lora",
        "huggingface/transformers-bloom-inference",
        "huggingface/peft",
        "luvris2/streamlit_chatbot",
        "fun-research/TiTok",
        "nik-dim/tall_masks",
    }
)


class ValidationError(RuntimeError):
    """A deterministic validation precondition or result failure."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def parse_coverage(output: str) -> dict[str, int]:
    """Parse the five stable coverage summary rows in contract order."""

    found: dict[str, int] = {}
    pattern = re.compile(
        r"^  (LOCKED_STATIC|LOCKED_ENV_BINDING|LOCKED_ANNOTATION|"
        r"ACKNOWLEDGED_DYNAMIC|UNRESOLVED): (\d+)$"
    )
    for line in output.splitlines():
        match = pattern.fullmatch(line)
        if match:
            found[match.group(1)] = int(match.group(2))
    if tuple(found) != COVERAGE_ORDER:
        raise ValidationError(
            "coverage summary missing or out of order: "
            f"expected {COVERAGE_ORDER!r}, found {tuple(found)!r}"
        )
    return found


def coverage_from_manifest(values: list[int]) -> dict[str, int]:
    if len(values) != len(COVERAGE_ORDER) or any(
        not isinstance(value, int) or value < 0 for value in values
    ):
        raise ValidationError(
            "coverage expectations must be five non-negative integers"
        )
    return dict(zip(COVERAGE_ORDER, values, strict=True))


def sanitized_environment(
    evidence_dir: Path,
    *,
    absent: tuple[str, ...] = (),
    conflicting: tuple[str, ...] = (),
) -> dict[str, str]:
    """Remove credentials and make committed config the only dependency truth."""

    removed_names = {name.upper() for name in (*TOKEN_VARIABLES, *absent, *conflicting)}
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper() not in removed_names
    }
    environment.update(
        {
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "HF_HOME": str(evidence_dir / "hf-home"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    for name in conflicting:
        environment[name] = "hf-freeze/slice18-conflicting-value"
    return environment


def load_manifest(path: Path, repository_root: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValidationError("manifest schema_version must be 1")
    baseline = document.get("hf_freeze_baseline")
    if not isinstance(baseline, str) or re.fullmatch(r"[0-9a-f]{40}", baseline) is None:
        raise ValidationError("manifest hf_freeze_baseline must be a full Git SHA")
    projects = document.get("projects")
    if not isinstance(projects, list) or not 10 <= len(projects) <= 12:
        raise ValidationError("manifest must contain 10-12 projects")

    ids: set[str] = set()
    repositories: set[str] = set()
    production_style = 0
    mechanisms: set[str] = set()
    for project in projects:
        if not isinstance(project, dict):
            raise ValidationError("every project entry must be an object")
        project_id = project.get("id")
        repository = project.get("repository")
        commit = project.get("commit")
        if not isinstance(project_id, str) or not project_id:
            raise ValidationError("every project must have a non-empty id")
        if project_id in ids:
            raise ValidationError(f"duplicate project id: {project_id}")
        ids.add(project_id)
        if not isinstance(repository, str) or "/" not in repository:
            raise ValidationError(f"{project_id}: repository must be owner/name")
        repositories.add(repository)
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValidationError(f"{project_id}: commit must be a full Git SHA")
        for key in ("project_url", "clone_url", "application_boundary"):
            if not isinstance(project.get(key), str) or not project[key]:
                raise ValidationError(f"{project_id}: {key} must be a non-empty string")
        overlay = project.get("overlay")
        if not isinstance(overlay, str):
            raise ValidationError(f"{project_id}: overlay must be a path")
        overlay_path = (repository_root / overlay).resolve()
        try:
            overlay_path.relative_to(repository_root.resolve())
        except ValueError as error:
            raise ValidationError(
                f"{project_id}: overlay leaves repository root"
            ) from error
        if not overlay_path.is_file():
            raise ValidationError(f"{project_id}: overlay does not exist: {overlay}")
        coverage_from_manifest(project.get("full_tree_counts", []))
        coverage_from_manifest(project.get("configured_counts", []))
        project_mechanisms = project.get("mechanisms")
        if not isinstance(project_mechanisms, list) or any(
            not isinstance(item, str) for item in project_mechanisms
        ):
            raise ValidationError(f"{project_id}: mechanisms must be strings")
        mechanisms.update(project_mechanisms)
        ambient = project.get("ambient_variables")
        if not isinstance(ambient, list) or any(
            not isinstance(item, str) or not item for item in ambient
        ):
            raise ValidationError(f"{project_id}: ambient_variables must be strings")
        if project.get("cohort") == "production-style":
            production_style += 1

    missing = REQUIRED_REPOSITORIES - repositories
    if missing:
        raise ValidationError(
            f"manifest omits required repositories: {sorted(missing)}"
        )
    if production_style < 2:
        raise ValidationError("manifest needs at least two production-style projects")
    required_mechanisms = {
        "application_scope",
        "environment_binding",
        "dependency_annotation",
        "ignore_directive",
    }
    if missing_mechanisms := required_mechanisms - mechanisms:
        raise ValidationError(
            f"manifest omits required mechanisms: {sorted(missing_mechanisms)}"
        )
    return document


def validate_audited_base(repository_root: Path, audited_base: str) -> str:
    """Return HEAD when the configured audited base is in its ancestry."""

    head_result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if head_result.returncode:
        raise ValidationError(
            f"source HEAD is unavailable in {repository_root}: "
            f"{head_result.stderr.strip()}"
        )
    source_head = head_result.stdout.strip()

    base_result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{audited_base}^{{commit}}"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if base_result.returncode:
        raise ValidationError(
            f"audited base {audited_base} is unavailable in {repository_root}"
        )

    ancestry_result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", audited_base, source_head],
        cwd=repository_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if ancestry_result.returncode == 1:
        raise ValidationError(
            f"audited base {audited_base} is not an ancestor of source HEAD "
            f"{source_head}"
        )
    if ancestry_result.returncode:
        raise ValidationError(
            f"could not validate audited base {audited_base} against source HEAD "
            f"{source_head}: {ancestry_result.stderr.strip()}"
        )
    return source_head


def _run_logged(
    arguments: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        code, output = completed.returncode, completed.stdout
    except subprocess.TimeoutExpired as error:
        partial = error.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        code, output = 124, f"{partial}\nCOMMAND TIMED OUT\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8", newline="\n")
    return code, output


def _tool_command(wheel: Path, *arguments: str) -> list[str]:
    return [
        "uv",
        "tool",
        "run",
        "--isolated",
        "--refresh-package",
        "hf-freeze",
        "--from",
        str(wheel),
        "hf-freeze",
        *arguments,
    ]


def _clone_project(
    project: dict[str, Any],
    checkout: Path,
    environment: dict[str, str],
    logs: Path,
) -> int:
    checkout.mkdir(parents=True)
    commands = (
        ["git", "init", "--quiet"],
        ["git", "remote", "add", "origin", project["clone_url"]],
    )
    for index, command in enumerate(commands, start=1):
        code, _ = _run_logged(
            command,
            cwd=checkout,
            environment=environment,
            log_path=logs / f"clone-{index}.log",
            timeout_seconds=60,
        )
        if code:
            raise ValidationError(f"{project['id']}: clone setup failed")

    attempts = 0
    for attempts in (1, 2):
        code, _ = _run_logged(
            [
                "git",
                "-c",
                "credential.helper=",
                "fetch",
                "--quiet",
                "--depth",
                "1",
                "origin",
                project["commit"],
            ],
            cwd=checkout,
            environment=environment,
            log_path=logs / f"fetch-{attempts}.log",
            timeout_seconds=120,
        )
        if code == 0:
            break
    else:
        raise ValidationError(
            f"{project['id']}: exact commit fetch failed after two attempts"
        )

    code, _ = _run_logged(
        ["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        cwd=checkout,
        environment=environment,
        log_path=logs / "checkout.log",
        timeout_seconds=60,
    )
    if code:
        raise ValidationError(f"{project['id']}: exact commit checkout failed")
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.strip()
    if actual != project["commit"]:
        raise ValidationError(
            f"{project['id']}: checkout mismatch {actual} != {project['commit']}"
        )
    return attempts


def _apply_overlay(
    overlay: Path,
    checkout: Path,
    environment: dict[str, str],
    logs: Path,
) -> None:
    for name, suffix in (("overlay-check", "--check"), ("overlay-apply", None)):
        command = ["git", "apply"]
        if suffix:
            command.append(suffix)
        command.append(str(overlay))
        code, _ = _run_logged(
            command,
            cwd=checkout,
            environment=environment,
            log_path=logs / f"{name}.log",
            timeout_seconds=60,
        )
        if code:
            raise ValidationError(f"overlay failed for {checkout.name}: {name}")


def _finding_lines(output: str) -> list[str]:
    lines: list[str] = []
    for line in output.splitlines():
        if line.startswith("Coverage summary"):
            break
        if (
            line
            and re.fullmatch(
                r"(?:Audited|Installed|Prepared|Resolved) \d+ packages?.*", line
            )
            is None
        ):
            lines.append(line)
    return lines


def _classification_equivalent(
    first_code: int, first_output: str, second_code: int, second_output: str
) -> bool:
    return (
        first_code == second_code
        and _finding_lines(first_output) == _finding_lines(second_output)
        and parse_coverage(first_output) == parse_coverage(second_output)
    )


def _source_diff(output: str) -> str:
    """Return only the unified source diff from pin's mixed human output."""

    lines = output.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.startswith("--- a/")), None
    )
    if start is None:
        return ""
    return "\n".join(lines[start:]) + "\n"


def _command_record(command: str, code: int) -> dict[str, Any]:
    return {"command": command, "exit_code": code}


def _lock_entries(lock_path: Path) -> list[dict[str, Any]]:
    document = json.loads(lock_path.read_text(encoding="utf-8"))
    dependencies = document.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValidationError("generated hf.lock has no dependencies array")
    return dependencies


def _lock_source_count(entries: list[dict[str, Any]]) -> int:
    return sum(len(entry.get("sources", [])) for entry in entries)


def _validate_project(
    project: dict[str, Any],
    *,
    repository_root: Path,
    wheel: Path,
    evidence_dir: Path,
    base_environment: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    project_id = project["id"]
    checkout = evidence_dir / "checkouts" / project_id
    logs = evidence_dir / "logs" / project_id
    errors: list[str] = []
    fetch_attempts = _clone_project(project, checkout, base_environment, logs)
    full_timeout = int(project.get("full_scan_timeout_seconds", 180))
    full_code, full_output = _run_logged(
        _tool_command(wheel, "scan", "."),
        cwd=checkout,
        environment=base_environment,
        log_path=logs / "full-tree-scan.log",
        timeout_seconds=full_timeout,
    )
    full_counts = parse_coverage(full_output)
    expected_full = coverage_from_manifest(project["full_tree_counts"])
    if full_code != 0:
        errors.append(f"{project_id}: full-tree scan exited {full_code}")
    if full_counts != expected_full:
        errors.append(
            f"{project_id}: full-tree counts {full_counts} != {expected_full}"
        )

    overlay_relative = project["overlay"]
    overlay = (repository_root / overlay_relative).resolve()
    _apply_overlay(overlay, checkout, base_environment, logs)

    ambient = tuple(project["ambient_variables"])
    absent_environment = sanitized_environment(evidence_dir, absent=ambient)
    configured_code, configured_output = _run_logged(
        _tool_command(wheel, "scan", "."),
        cwd=checkout,
        environment=absent_environment,
        log_path=logs / "configured-scan-ambient-absent.log",
        timeout_seconds=180,
    )
    configured_counts = parse_coverage(configured_output)
    expected_configured = coverage_from_manifest(project["configured_counts"])
    if configured_code != 0:
        errors.append(f"{project_id}: configured scan exited {configured_code}")
    if configured_counts != expected_configured:
        errors.append(
            f"{project_id}: configured counts {configured_counts} != "
            f"{expected_configured}"
        )

    commands = [_command_record("hf-freeze scan .", configured_code)]
    lock_code, _ = _run_logged(
        _tool_command(wheel, "lock", "."),
        cwd=checkout,
        environment=absent_environment,
        log_path=logs / "lock-ambient-absent.log",
        timeout_seconds=180,
    )
    commands.append(_command_record("hf-freeze lock .", lock_code))
    lock_path = checkout / "hf.lock"
    absent_lock = (
        lock_path.read_bytes() if lock_code == 0 and lock_path.is_file() else None
    )

    invariance: dict[str, Any] | None = None
    if ambient:
        if absent_lock is not None:
            lock_path.unlink()
        conflicting_environment = sanitized_environment(
            evidence_dir, absent=ambient, conflicting=ambient
        )
        conflict_scan_code, conflict_scan_output = _run_logged(
            _tool_command(wheel, "scan", "."),
            cwd=checkout,
            environment=conflicting_environment,
            log_path=logs / "configured-scan-ambient-conflicting.log",
            timeout_seconds=180,
        )
        conflict_lock_code, _ = _run_logged(
            _tool_command(wheel, "lock", "."),
            cwd=checkout,
            environment=conflicting_environment,
            log_path=logs / "lock-ambient-conflicting.log",
            timeout_seconds=180,
        )
        conflict_lock = (
            lock_path.read_bytes()
            if conflict_lock_code == 0 and lock_path.is_file()
            else None
        )
        invariance = {
            "variables": list(ambient),
            "absent_scan_exit_code": configured_code,
            "conflicting_scan_exit_code": conflict_scan_code,
            "classification_equivalent": _classification_equivalent(
                configured_code,
                configured_output,
                conflict_scan_code,
                conflict_scan_output,
            ),
            "absent_lock_exit_code": lock_code,
            "conflicting_lock_exit_code": conflict_lock_code,
            "lock_truth_equivalent": absent_lock == conflict_lock
            and absent_lock is not None,
            "lock_sha256": None if absent_lock is None else sha256_bytes(absent_lock),
        }
        if not invariance["classification_equivalent"]:
            errors.append(f"{project_id}: ambient value changed scan classification")
        if not invariance["lock_truth_equivalent"]:
            errors.append(f"{project_id}: ambient value changed lock truth")
        if conflict_lock_code != 0:
            lock_code = conflict_lock_code

    lifecycle_status = "lock_failed"
    entries: list[dict[str, Any]] = []
    pin_preview = ""
    changed_files: list[str] = []
    if lock_code == 0 and lock_path.is_file():
        entries = _lock_entries(lock_path)
        lockable_sites = sum(configured_counts[name] for name in COVERAGE_ORDER[:3])
        if _lock_source_count(entries) != lockable_sites:
            errors.append(
                f"{project_id}: lock sources {_lock_source_count(entries)} != "
                f"lockable sites {lockable_sites}; ignored/unresolved coverage "
                "may be wrong"
            )

        preview_code, pin_output = _run_logged(
            _tool_command(wheel, "pin", "."),
            cwd=checkout,
            environment=absent_environment,
            log_path=logs / "pin-preview.log",
            timeout_seconds=180,
        )
        pin_preview = _source_diff(pin_output)
        commands.append(_command_record("hf-freeze pin .", preview_code))
        write_code, _ = _run_logged(
            _tool_command(wheel, "pin", ".", "--write"),
            cwd=checkout,
            environment=absent_environment,
            log_path=logs / "pin-write.log",
            timeout_seconds=180,
        )
        commands.append(_command_record("hf-freeze pin . --write", write_code))
        check_code, check_output = _run_logged(
            _tool_command(wheel, "check", ".", "--frozen"),
            cwd=checkout,
            environment=absent_environment,
            log_path=logs / "check-frozen.log",
            timeout_seconds=180,
        )
        commands.append(_command_record("hf-freeze check . --frozen", check_code))
        check_counts = parse_coverage(check_output)
        if check_counts != configured_counts:
            errors.append(f"{project_id}: frozen-check coverage changed")
        diff_code, diff_output = _run_logged(
            ["git", "diff", "--name-only", "--", "*.py"],
            cwd=checkout,
            environment=base_environment,
            log_path=logs / "changed-python-files.log",
            timeout_seconds=60,
        )
        if diff_code:
            errors.append(f"{project_id}: could not list source changes")
        changed_files = [line for line in diff_output.splitlines() if line]
        if any(code != 0 for code in (preview_code, write_code, check_code)):
            errors.append(
                f"{project_id}: lifecycle exits pin={preview_code}, "
                f"pin-write={write_code}, check={check_code}"
            )
            lifecycle_status = "failed"
        elif configured_counts["ACKNOWLEDGED_DYNAMIC"]:
            lifecycle_status = "complete_with_acknowledged_dynamic"
        else:
            lifecycle_status = "complete"
    else:
        errors.append(f"{project_id}: configured lock exited {lock_code}")

    full_outcome = (
        "scan_complete_lock_ineligible"
        if full_counts["UNRESOLVED"]
        else "scan_complete_lock_eligible"
    )
    result = {
        "id": project_id,
        "repository": project["repository"],
        "project_url": project["project_url"],
        "revision_url": f"{project['project_url']}/tree/{project['commit']}",
        "commit": project["commit"],
        "cohort": project["cohort"],
        "fetch_attempts": fetch_attempts,
        "application_boundary": project["application_boundary"],
        "mechanisms": project["mechanisms"],
        "full_tree": {
            "scan_exit_code": full_code,
            "outcome": full_outcome,
            "classification": project.get("full_tree_classification", "none"),
            "coverage": full_counts,
            "output_sha256": sha256_bytes(full_output.encode("utf-8")),
        },
        "configured_scope": {
            "scan_exit_code": configured_code,
            "outcome": "no_unresolved_findings"
            if configured_counts["UNRESOLVED"] == 0
            else "unresolved_findings",
            "coverage": configured_counts,
            "findings": _finding_lines(configured_output),
            "overlay": overlay_relative,
            "overlay_sha256": sha256_file(overlay),
        },
        "ambient_environment_invariance": invariance,
        "lifecycle": {
            "status": lifecycle_status,
            "commands": commands,
            "lock_entries": entries,
            "source_pin_diff": pin_preview,
            "source_pin_diff_sha256": sha256_bytes(pin_preview.encode("utf-8")),
            "changed_python_files": changed_files,
        },
    }
    return result, errors


def _write_json(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _reconcile_existing_evidence(
    *,
    manifest: dict[str, Any],
    wheel: Path,
    evidence_dir: Path,
    results_path: Path,
) -> int:
    """Re-normalize a completed run without repeating external operations."""

    if not evidence_dir.is_dir() or not results_path.is_file():
        raise ValidationError("completed evidence directory and results are required")
    document = json.loads(results_path.read_text(encoding="utf-8"))
    if document.get("hf_freeze", {}).get("wheel_sha256") != sha256_file(wheel):
        raise ValidationError("existing results were produced by different wheel bytes")
    projects_by_id = {
        project["id"]: project for project in document.get("projects", [])
    }
    if set(projects_by_id) != {project["id"] for project in manifest["projects"]}:
        raise ValidationError("existing results do not match the manifest project set")

    removed_errors = {
        f"{project['id']}: ambient value changed scan classification"
        for project in manifest["projects"]
        if project["ambient_variables"]
    }
    errors = [
        error
        for error in document.get("validation_errors", [])
        if error not in removed_errors
    ]
    for project in manifest["projects"]:
        project_id = project["id"]
        normalized = projects_by_id[project_id]
        logs = evidence_dir / "logs" / project_id
        absent_output = (logs / "configured-scan-ambient-absent.log").read_text(
            encoding="utf-8"
        )
        normalized["configured_scope"]["findings"] = _finding_lines(absent_output)
        pin_output = (logs / "pin-preview.log").read_text(encoding="utf-8")
        source_diff = _source_diff(pin_output)
        normalized["lifecycle"]["source_pin_diff"] = source_diff
        normalized["lifecycle"]["source_pin_diff_sha256"] = sha256_bytes(
            source_diff.encode("utf-8")
        )
        ambient = project["ambient_variables"]
        if not ambient:
            continue
        conflict_output = (logs / "configured-scan-ambient-conflicting.log").read_text(
            encoding="utf-8"
        )
        invariance = normalized["ambient_environment_invariance"]
        equivalent = _classification_equivalent(
            invariance["absent_scan_exit_code"],
            absent_output,
            invariance["conflicting_scan_exit_code"],
            conflict_output,
        )
        invariance["classification_equivalent"] = equivalent
        if not equivalent:
            errors.append(f"{project_id}: ambient value changed scan classification")

    document["validation_errors"] = errors
    _write_json(results_path, document)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Reconciled normalized results from {evidence_dir}")
    return 0


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--reconcile-existing-evidence",
        action="store_true",
        help=(
            "Re-normalize a completed run without repeating network or lifecycle "
            "commands."
        ),
    )
    options = parser.parse_args(arguments)

    repository_root = Path(__file__).resolve().parents[1]
    manifest_path = options.manifest.resolve()
    wheel = options.wheel.resolve()
    evidence_dir = options.evidence_dir.resolve()
    results_path = options.results.resolve()
    manifest = load_manifest(manifest_path, repository_root)
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValidationError(f"wheel does not exist: {wheel}")

    audited_base = manifest["hf_freeze_baseline"]
    source_head = validate_audited_base(repository_root, audited_base)

    if options.reconcile_existing_evidence:
        return _reconcile_existing_evidence(
            manifest=manifest,
            wheel=wheel,
            evidence_dir=evidence_dir,
            results_path=results_path,
        )
    if evidence_dir.exists():
        raise ValidationError(f"evidence directory already exists: {evidence_dir}")
    evidence_dir.mkdir(parents=True)

    base_environment = sanitized_environment(evidence_dir)
    smoke_logs = evidence_dir / "logs" / "hf-freeze"
    help_code, help_output = _run_logged(
        _tool_command(wheel, "--help"),
        cwd=repository_root,
        environment=base_environment,
        log_path=smoke_logs / "help.log",
        timeout_seconds=60,
    )
    version_code, version_output = _run_logged(
        _tool_command(wheel, "version"),
        cwd=repository_root,
        environment=base_environment,
        log_path=smoke_logs / "version.log",
        timeout_seconds=60,
    )

    projects: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    for project in manifest["projects"]:
        print(f"Validating {project['id']} @ {project['commit']}", flush=True)
        result, errors = _validate_project(
            project,
            repository_root=repository_root,
            wheel=wheel,
            evidence_dir=evidence_dir,
            base_environment=base_environment,
        )
        projects.append(result)
        validation_errors.extend(errors)

    if help_code:
        validation_errors.append(f"isolated wheel help exited {help_code}")
    if version_code:
        validation_errors.append(f"isolated wheel version exited {version_code}")
    document = {
        "schema_version": 1,
        "hf_freeze": {
            "audited_base": audited_base,
            "source_head": source_head,
            "wheel_filename": wheel.name,
            "wheel_sha256": sha256_file(wheel),
            "version_output": version_output.strip(),
            "help_exit_code": help_code,
            "help_output_sha256": sha256_bytes(help_output.encode("utf-8")),
        },
        "coverage_order": list(COVERAGE_ORDER),
        "method": {
            "project_count": len(projects),
            "target_code_installed_imported_or_executed": False,
            "ambient_or_dotenv_dependency_input": False,
            "credentials_used": False,
            "model_weights_downloaded": False,
            "raw_logs_location": "external --evidence-dir only",
        },
        "validation_errors": validation_errors,
        "projects": projects,
    }
    _write_json(results_path, document)
    if validation_errors:
        for error in validation_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote normalized results for {len(projects)} projects to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
