"""Static, no-network discovery of supported Hugging Face Hub calls."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider, ScopeProvider

from hf_freeze.config import (
    DEFAULT_EXCLUDED_DIRECTORIES as _DEFAULT_EXCLUDED_DIRECTORIES,
)
from hf_freeze.config import (
    DeclaredDependency,
    ProjectConfig,
    ProjectContext,
    iter_scoped_python_files,
)
from hf_freeze.models import (
    AcknowledgedDynamicFinding,
    CallKind,
    DependencyFinding,
    DiagnosticSeverity,
    RepoType,
    ScanDiagnostic,
    ScanResult,
    SourceLocation,
)

DEFAULT_EXCLUDED_DIRECTORIES = _DEFAULT_EXCLUDED_DIRECTORIES

LOCAL_DATASET_BUILDERS = frozenset({"json", "parquet"})
_DIRECTIVE_PREFIX = "hf-freeze"
_DEPENDENCY_DIRECTIVE = re.compile(r"hf-freeze:\s*dependency=([A-Za-z0-9_-]+)\Z")
_IGNORE_DIRECTIVE = re.compile(r"hf-freeze:\s*ignore=([A-Za-z0-9_-]+)\Z")
_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
_MALFORMED_DIRECTIVE = (
    "malformed hf-freeze directive; expected '# hf-freeze: dependency=<name>' "
    "or '# hf-freeze: ignore=<reason>' using a nonempty ASCII identifier"
)


@dataclass(frozen=True)
class CallSpec:
    kind: CallKind
    repo_type: RepoType
    keyword_names: tuple[str, ...]
    positional_index: int | None = 0


@dataclass(frozen=True)
class _EnvironmentReference:
    """A narrow, committed-config-only environment lookup fact."""

    name: str
    fallback: str | None = None


@dataclass(frozen=True)
class _SourceDirective:
    """One validated directive attached to exactly one supported call."""

    declaration: DeclaredDependency | None = None
    ignore_reason: str | None = None


class _ConstantCollector(cst.CSTVisitor):
    """Record literal values for simple binding nodes."""

    def __init__(self) -> None:
        self.values: dict[int, str | bool | _EnvironmentReference | None] = {}

    def visit_Assign(self, node: cst.Assign) -> None:
        value = _literal_value(node.value)
        if value is None and len(node.targets) == 1:
            value = _environment_reference(node.value)
        for target in node.targets:
            if isinstance(target.target, cst.Name):
                self.values[id(target.target)] = value

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        if isinstance(node.target, cst.Name):
            value = _literal_value(node.value) if node.value is not None else None
            if value is None and node.value is not None:
                value = _environment_reference(node.value)
            self.values[id(node.target)] = value

    def visit_AugAssign(self, node: cst.AugAssign) -> None:
        if isinstance(node.target, cst.Name):
            self.values[id(node.target)] = None


class _SupportedCallCollector(cst.CSTVisitor):
    """Collect calls recognized by the existing matcher within one statement."""

    def __init__(self) -> None:
        self.calls: list[cst.Call] = []

    def visit_Call(self, node: cst.Call) -> None:
        if match_call(node.func) is not None:
            self.calls.append(node)


class _StructureCollector(cst.CSTVisitor):
    """Collect comments and simple statements with exact source positions."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self) -> None:
        self.comments: list[tuple[cst.Comment, int, int]] = []
        self.statements: list[tuple[cst.SimpleStatementLine, int]] = []

    def visit_Comment(self, node: cst.Comment) -> None:
        position = self.get_metadata(PositionProvider, node).start
        self.comments.append((node, position.line, position.column))

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> None:
        position = self.get_metadata(PositionProvider, node).start
        self.statements.append((node, position.line))


class _FindingVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, ScopeProvider)

    def __init__(
        self,
        display_path: str,
        binding_values: dict[int, str | bool | _EnvironmentReference | None],
        directives: dict[int, _SourceDirective],
        dependencies: tuple[DeclaredDependency, ...],
        environment_bindings: tuple[tuple[str, str], ...],
    ) -> None:
        self.display_path = display_path
        self.binding_values = binding_values
        self.directives = directives
        self.declarations = {item.name: item for item in dependencies}
        self.environment_bindings = dict(environment_bindings)
        self.used_declarations: set[str] = set()
        self.findings: list[DependencyFinding] = []
        self.acknowledged: list[AcknowledgedDynamicFinding] = []
        self.diagnostics: list[ScanDiagnostic] = []

    def visit_Call(self, node: cst.Call) -> None:
        spec = match_call(node.func)
        if spec is None:
            return

        source_directive = self.directives.get(id(node))
        directive = (
            source_directive.declaration if source_directive is not None else None
        )
        ignore_reason = (
            source_directive.ignore_reason if source_directive is not None else None
        )

        repo_expression = _find_argument(
            node, spec.keyword_names, positional_index=spec.positional_index
        )
        if (
            source_directive is None
            and spec.kind
            in {
                CallKind.PIPELINE,
                CallKind.SENTENCE_TRANSFORMER,
            }
            and (repo_expression is None)
        ):
            return
        repo_id, environment, unresolved_reason = _resolve_repo_reference(
            repo_expression,
            self._resolve_string_name,
            self._resolve_environment_name,
        )
        if (
            source_directive is None
            and environment is None
            and repo_id is None
            and isinstance(repo_expression, cst.Name)
        ):
            environment_problem = self._environment_name_problem(repo_expression)
            if environment_problem is not None:
                self._add_error(
                    node, environment_problem, "AMBIGUOUS_ENVIRONMENT_REFERENCE"
                )
                return
        environment_declaration = None
        if environment is not None:
            target = self.environment_bindings.get(environment.name)
            if target is not None:
                environment_declaration = self.declarations[target]
                self.used_declarations.add(environment_declaration.name)
            elif source_directive is None:
                self._add_error(
                    node,
                    _missing_environment_binding_reason(environment),
                    "MISSING_ENVIRONMENT_BINDING",
                )
                return

        if (
            environment_declaration is not None
            and directive is not None
            and environment_declaration.name != directive.name
        ):
            position = self.get_metadata(PositionProvider, node).start
            self.diagnostics.append(
                ScanDiagnostic(
                    SourceLocation(self.display_path, position.line, position.column),
                    f"environment variable {environment.name!r} binds dependency "
                    f"{environment_declaration.name!r}, but the source directive "
                    f"binds {directive.name!r}",
                    code="BINDING_DIRECTIVE_CONFLICT",
                )
            )
            return
        declaration = environment_declaration or directive
        if ignore_reason is not None:
            conflict = self._ignore_conflict(
                node,
                spec,
                repo_expression,
                repo_id,
                environment,
                environment_declaration,
                unresolved_reason,
            )
            if conflict is not None:
                self._add_error(node, conflict, "IGNORE_CONFLICT")
                return
            position = self.get_metadata(PositionProvider, node).start
            self.acknowledged.append(
                AcknowledgedDynamicFinding(
                    spec.kind,
                    SourceLocation(self.display_path, position.line, position.column),
                    ignore_reason,
                )
            )
            return
        if (
            declaration is None
            and repo_id is not None
            and _is_obvious_local_path(repo_id)
        ):
            return
        if (
            declaration is None
            and spec.kind is CallKind.LOAD_DATASET
            and repo_id in LOCAL_DATASET_BUILDERS
        ):
            builder_name = repo_id
            data_files = _find_argument(node, ("data_files",), positional_index=None)
            data_files_kind = _classify_data_files(data_files)
            if data_files_kind == "non_hub":
                return
            repo_id = None
            unresolved_reason = (
                "load_dataset uses Hugging Face data_files; repository ID "
                "extraction from data_files is unsupported"
                if data_files_kind == "hub"
                else (
                    f"load_dataset packaged builder '{builder_name}' "
                    "does not have confidently local data_files"
                )
            )

        revision_expression = _find_argument(node, ("revision",), positional_index=None)
        requested_revision, revision_unresolved_reason = _resolve_revision(
            revision_expression, self._resolve_string_name
        )
        repo_type = _repo_type(node, spec.repo_type, self._resolve_string_name)
        if declaration is not None:
            conflict = _apply_declaration(
                declaration,
                spec.kind,
                repo_id,
                repo_type,
                requested_revision,
                revision_expression is not None,
                revision_unresolved_reason,
            )
            if conflict is not None:
                position = self.get_metadata(PositionProvider, node).start
                self.diagnostics.append(
                    ScanDiagnostic(
                        SourceLocation(
                            self.display_path, position.line, position.column
                        ),
                        conflict,
                        code=(
                            "ENVIRONMENT_BINDING_CONFLICT"
                            if environment_declaration is not None
                            else "DIRECTIVE_CONFLICT"
                        ),
                    )
                )
                return
            repo_id = declaration.repo_id
            repo_type = declaration.repo_type
            unresolved_reason = None
            if revision_expression is None:
                requested_revision = declaration.revision
            revision_unresolved_reason = None
        trust_expression = _find_argument(
            node, ("trust_remote_code",), positional_index=None
        )
        trust_remote_code, trust_unresolved_reason = _resolve_boolean(
            trust_expression, self._resolve_boolean_name
        )
        position = self.get_metadata(PositionProvider, node).start
        self.findings.append(
            DependencyFinding(
                repo_id=repo_id,
                repo_type=repo_type,
                call_kind=spec.kind,
                requested_revision=requested_revision,
                source=SourceLocation(
                    path=self.display_path,
                    line=position.line,
                    column=position.column,
                ),
                unresolved_reason=unresolved_reason,
                revision_unresolved_reason=revision_unresolved_reason,
                trust_remote_code=trust_remote_code,
                trust_remote_code_unresolved_reason=trust_unresolved_reason,
            )
        )

    def _ignore_conflict(
        self,
        node: cst.Call,
        spec: CallSpec,
        repo_expression: cst.BaseExpression | None,
        repo_id: str | None,
        environment: _EnvironmentReference | None,
        environment_declaration: DeclaredDependency | None,
        unresolved_reason: str | None,
    ) -> str | None:
        """Reject ignores that would hide anything except repository dynamism."""

        if (
            spec.kind in {CallKind.PIPELINE, CallKind.SENTENCE_TRANSFORMER}
            and repo_expression is None
        ):
            return "ignore directive conflicts with a call the scanner would omit"
        if environment_declaration is not None:
            return (
                "ignore directive conflicts with committed environment binding "
                f"for {environment.name!r}"
            )
        if repo_id in LOCAL_DATASET_BUILDERS and spec.kind is CallKind.LOAD_DATASET:
            data_files = _find_argument(node, ("data_files",), positional_index=None)
            data_files_kind = _classify_data_files(data_files)
            if data_files_kind == "non_hub":
                return (
                    "ignore directive conflicts with a confidently local dataset call"
                )
            repo_id = None
            unresolved_reason = (
                "load_dataset uses Hugging Face data_files; repository ID extraction "
                "from data_files is unsupported"
                if data_files_kind == "hub"
                else "load_dataset packaged builder does not have confidently local "
                "data_files"
            )
        if repo_id is not None:
            if _is_obvious_local_path(repo_id):
                return "ignore directive conflicts with a confidently local path"
            return "ignore directive conflicts with an already resolved repository ID"
        if isinstance(repo_expression, cst.Name) and environment is None:
            environment_problem = self._environment_name_problem(repo_expression)
            if environment_problem is not None:
                return f"ignore directive cannot override: {environment_problem}"
        if unresolved_reason is None and environment is None:
            return "ignore directive applies to a call that is not unresolved"

        revision_expression = _find_argument(node, ("revision",), positional_index=None)
        _, revision_error = _resolve_revision(
            revision_expression, self._resolve_string_name
        )
        if revision_error is not None:
            return f"ignore directive cannot override source revision: {revision_error}"
        if _repo_type(node, spec.repo_type, self._resolve_string_name) is None:
            return "ignore directive cannot override an unresolved explicit repo_type"
        trust_expression = _find_argument(
            node, ("trust_remote_code",), positional_index=None
        )
        trust_remote_code, trust_error = _resolve_boolean(
            trust_expression, self._resolve_boolean_name
        )
        if trust_error is not None:
            return f"ignore directive cannot override: {trust_error}"
        if trust_remote_code:
            return "ignore directive cannot acknowledge trust_remote_code=True"
        return None

    def _binding_value(
        self, expression: cst.Name
    ) -> str | bool | _EnvironmentReference | None:
        scope = self.get_metadata(ScopeProvider, expression)
        assignments = scope[expression.value]
        if len(assignments) != 1:
            return None
        assignment = next(iter(assignments))
        return self.binding_values.get(id(assignment.node))

    def _resolve_string_name(self, expression: cst.Name) -> str | None:
        value = self._binding_value(expression)
        return value if isinstance(value, str) else None

    def _resolve_boolean_name(self, expression: cst.Name) -> bool | None:
        value = self._binding_value(expression)
        return value if isinstance(value, bool) else None

    def _resolve_environment_name(
        self, expression: cst.Name
    ) -> _EnvironmentReference | None:
        scope = self.get_metadata(ScopeProvider, expression)
        assignments = scope[expression.value]
        if len(assignments) != 1:
            return None
        assignment = next(iter(assignments))
        if assignment.scope is not scope:
            return None
        assignment_position = self.get_metadata(PositionProvider, assignment.node).start
        use_position = self.get_metadata(PositionProvider, expression).start
        if (assignment_position.line, assignment_position.column) >= (
            use_position.line,
            use_position.column,
        ):
            return None
        value = self.binding_values.get(id(assignment.node))
        return value if isinstance(value, _EnvironmentReference) else None

    def _environment_name_problem(self, expression: cst.Name) -> str | None:
        scope = self.get_metadata(ScopeProvider, expression)
        assignments = scope[expression.value]
        environment_assignments = [
            assignment
            for assignment in assignments
            if isinstance(
                self.binding_values.get(id(assignment.node)), _EnvironmentReference
            )
        ]
        if not environment_assignments:
            return None
        same_scope = [
            assignment
            for assignment in environment_assignments
            if assignment.scope is scope
        ]
        if not same_scope:
            return (
                f"repository ID name {expression.value!r} has an environment-reference "
                "assignment outside the relevant lexical scope"
            )
        if len(assignments) != 1 or len(same_scope) != 1:
            return (
                f"repository ID name {expression.value!r} does not have exactly one "
                "unambiguous environment-reference assignment in this lexical scope"
            )
        assignment_position = self.get_metadata(
            PositionProvider, same_scope[0].node
        ).start
        use_position = self.get_metadata(PositionProvider, expression).start
        if (assignment_position.line, assignment_position.column) >= (
            use_position.line,
            use_position.column,
        ):
            return (
                f"repository ID name {expression.value!r} must be used after its "
                "environment-reference assignment"
            )
        return None

    def _add_error(self, node: cst.Call, message: str, code: str) -> None:
        position = self.get_metadata(PositionProvider, node).start
        self.diagnostics.append(
            ScanDiagnostic(
                SourceLocation(self.display_path, position.line, position.column),
                message,
                code=code,
            )
        )


def scan_path(
    path: str | Path = ".", *, context: ProjectContext | None = None
) -> ScanResult:
    """Scan one Python file or a directory tree without importing project code."""

    target = Path(path)
    if context is None:
        root = target.resolve(strict=False)
        root = root.parent if root.is_file() else root
        context = ProjectContext(root, None, ProjectConfig())

    findings: list[DependencyFinding] = []
    acknowledged: list[AcknowledgedDynamicFinding] = []
    diagnostics: list[ScanDiagnostic] = []
    used_declarations: set[str] = set()
    for source_path, display_path in iter_scoped_python_files(context, target):
        file_findings, file_acknowledged, file_diagnostics, used_names = _scan_file(
            source_path,
            display_path,
            context.config.dependencies,
            context.config.environment_bindings,
        )
        findings.extend(file_findings)
        acknowledged.extend(file_acknowledged)
        diagnostics.extend(file_diagnostics)
        used_declarations.update(used_names)

    config_display_path = (
        context.config_path.relative_to(context.root).as_posix()
        if context.config_path is not None
        else "pyproject.toml"
    )
    for declaration in context.config.dependencies:
        if declaration.name not in used_declarations:
            diagnostics.append(
                ScanDiagnostic(
                    SourceLocation(config_display_path, 1, 0),
                    f"dependency {declaration.name!r} is declared but not used in "
                    "scanned source",
                    DiagnosticSeverity.WARNING,
                    "UNUSED_DECLARATION",
                )
            )

    def location_key(
        item: DependencyFinding | AcknowledgedDynamicFinding | ScanDiagnostic,
    ) -> tuple[str, int, int]:
        return (item.source.path, item.source.line, item.source.column)

    return ScanResult(
        findings=tuple(sorted(findings, key=location_key)),
        diagnostics=tuple(sorted(diagnostics, key=location_key)),
        acknowledged=tuple(sorted(acknowledged, key=location_key)),
    )


def _scan_file(
    source_path: Path,
    display_path: str,
    dependencies: tuple[DeclaredDependency, ...],
    environment_bindings: tuple[tuple[str, str], ...],
) -> tuple[
    list[DependencyFinding],
    list[AcknowledgedDynamicFinding],
    list[ScanDiagnostic],
    set[str],
]:
    try:
        module = cst.parse_module(source_path.read_bytes())
    except cst.ParserSyntaxError as error:
        message = " ".join(error.message.split())
        return (
            [],
            [],
            [
                ScanDiagnostic(
                    source=SourceLocation(
                        path=display_path,
                        line=error.raw_line,
                        column=error.raw_column,
                    ),
                    message=f"parse error: {message}",
                )
            ],
            set(),
        )
    except OSError as error:
        return (
            [],
            [],
            [
                ScanDiagnostic(
                    source=SourceLocation(path=display_path, line=1, column=0),
                    message=f"read error: {error}",
                )
            ],
            set(),
        )

    wrapper = MetadataWrapper(module)
    directives, diagnostics, used_names = _analyze_directives(
        wrapper, display_path, dependencies
    )
    collector = _ConstantCollector()
    wrapper.module.visit(collector)
    visitor = _FindingVisitor(
        display_path,
        collector.values,
        directives,
        dependencies,
        environment_bindings,
    )
    wrapper.visit(visitor)
    diagnostics.extend(visitor.diagnostics)
    return (
        visitor.findings,
        visitor.acknowledged,
        diagnostics,
        used_names | visitor.used_declarations,
    )


def _analyze_directives(
    wrapper: MetadataWrapper,
    display_path: str,
    dependencies: tuple[DeclaredDependency, ...],
) -> tuple[
    dict[int, _SourceDirective],
    list[ScanDiagnostic],
    set[str],
]:
    """Attach canonical comments to exactly one supported call."""

    structure = _StructureCollector()
    wrapper.visit(structure)
    source_lines = wrapper.module.code.splitlines()
    comments = {
        line: (node, column)
        for node, line, column in structure.comments
        if not source_lines[line - 1][:column].strip()
    }
    directive_comments = {
        line: (node, column)
        for node, line, column in structure.comments
        if _is_directive_like(node.value)
    }
    declarations = {item.name: item for item in dependencies}
    handled: set[int] = set()
    bindings: dict[int, _SourceDirective] = {}
    diagnostics: list[ScanDiagnostic] = []
    used_names: set[str] = set()

    for statement, statement_line in structure.statements:
        previous = statement_line - 1
        if previous not in comments or previous not in directive_comments:
            continue

        block_directives: list[tuple[cst.Comment, int, int]] = []
        line = previous
        while line in comments and line in directive_comments:
            comment, column = comments[line]
            block_directives.append((comment, line, column))
            line -= 1

        call_collector = _SupportedCallCollector()
        statement.visit(call_collector)
        calls = call_collector.calls
        handled.update(id(comment) for comment, _, _ in block_directives)

        if len(block_directives) > 1:
            diagnostics.append(
                _directive_diagnostic(
                    display_path,
                    block_directives[0],
                    "MULTIPLE_DIRECTIVES",
                    "multiple hf-freeze directives are attached to one statement",
                )
            )
            continue

        record = block_directives[0]
        comment = record[0]
        directive_text = comment.value[1:].strip()
        dependency_match = _DEPENDENCY_DIRECTIVE.fullmatch(directive_text)
        ignore_match = _IGNORE_DIRECTIVE.fullmatch(directive_text)
        if dependency_match is None and ignore_match is None:
            diagnostics.append(
                _directive_diagnostic(
                    display_path, record, "MALFORMED_DIRECTIVE", _MALFORMED_DIRECTIVE
                )
            )
            continue
        if len(calls) != 1:
            diagnostics.append(
                _directive_diagnostic(
                    display_path,
                    record,
                    "DIRECTIVE_CALL_COUNT",
                    "hf-freeze directive must attach to a statement containing "
                    f"exactly one supported Hub call; found {len(calls)}",
                )
            )
            continue
        if len(statement.body) != 1:
            diagnostics.append(
                _directive_diagnostic(
                    display_path,
                    record,
                    "DIRECTIVE_COMPOUND_STATEMENT",
                    "hf-freeze directive must attach to one simple statement, not a "
                    "compound semicolon statement",
                )
            )
            continue

        call = calls[0]
        if ignore_match is not None:
            bindings[id(call)] = _SourceDirective(ignore_reason=ignore_match.group(1))
            continue

        assert dependency_match is not None
        name = dependency_match.group(1)
        declaration = declarations.get(name)
        if declaration is None:
            diagnostics.append(
                _directive_diagnostic(
                    display_path,
                    record,
                    "UNKNOWN_DECLARATION",
                    f"dependency directive references unknown declaration {name!r}",
                )
            )
            continue
        used_names.add(name)
        bindings[id(call)] = _SourceDirective(declaration=declaration)

    for line, (comment, column) in sorted(directive_comments.items()):
        if id(comment) in handled:
            continue
        record = (comment, line, column)
        directive_text = comment.value[1:].strip()
        if (
            _DEPENDENCY_DIRECTIVE.fullmatch(directive_text) is None
            and _IGNORE_DIRECTIVE.fullmatch(directive_text) is None
        ):
            code, message = "MALFORMED_DIRECTIVE", _MALFORMED_DIRECTIVE
        else:
            code, message = (
                "DETACHED_DIRECTIVE",
                "hf-freeze directive is detached; place it immediately above a "
                "simple statement with exactly one supported Hub call",
            )
        diagnostics.append(_directive_diagnostic(display_path, record, code, message))
    return bindings, diagnostics, used_names


def _is_directive_like(comment: str) -> bool:
    return comment[1:].strip().startswith(_DIRECTIVE_PREFIX)


def _directive_diagnostic(
    path: str,
    record: tuple[cst.Comment, int, int],
    code: str,
    message: str,
) -> ScanDiagnostic:
    _, line, column = record
    return ScanDiagnostic(SourceLocation(path, line, column), message, code=code)


def _apply_declaration(
    declaration: DeclaredDependency,
    call_kind: CallKind,
    repo_id: str | None,
    repo_type: RepoType | None,
    requested_revision: str | None,
    has_revision: bool,
    revision_error: str | None,
) -> str | None:
    """Return a deterministic conflict message, if committed inputs disagree."""

    if repo_type is None:
        return (
            f"dependency {declaration.name!r} cannot override an unresolved explicit "
            "repo_type expression"
        )
    required_type = (
        RepoType.DATASET
        if call_kind is CallKind.LOAD_DATASET
        else RepoType.MODEL
        if call_kind not in {CallKind.HF_HUB_DOWNLOAD, CallKind.SNAPSHOT_DOWNLOAD}
        else repo_type
    )
    if declaration.repo_type is not required_type:
        return (
            f"dependency {declaration.name!r} has repo_type "
            f"{declaration.repo_type.value!r}, which is incompatible with "
            f"{call_kind.value}"
        )
    if repo_type is not declaration.repo_type:
        return (
            f"dependency {declaration.name!r} has repo_type "
            f"{declaration.repo_type.value!r}, but the call resolves to "
            f"{repo_type.value!r}"
        )
    if repo_id is not None and repo_id != declaration.repo_id:
        return (
            f"dependency {declaration.name!r} declares repository "
            f"{declaration.repo_id!r}, but source resolves to {repo_id!r}"
        )
    if not has_revision:
        return None
    if revision_error is not None:
        return (
            f"dependency {declaration.name!r} cannot override source revision: "
            f"{revision_error}"
        )
    if requested_revision is not None and _COMMIT_SHA.fullmatch(requested_revision):
        return None
    if requested_revision != declaration.revision:
        return (
            f"dependency {declaration.name!r} tracks revision "
            f"{declaration.revision!r}, but source resolves to {requested_revision!r}"
        )
    return None


def match_call(function: cst.BaseExpression) -> CallSpec | None:
    """Return the scanner's supported call specification for a function."""

    if isinstance(function, cst.Attribute):
        name = function.attr.value
    elif isinstance(function, cst.Name):
        name = function.value
    else:
        return None

    if name == "load_dataset":
        return CallSpec(CallKind.LOAD_DATASET, RepoType.DATASET, ("path",))
    if name == "hf_hub_download":
        return CallSpec(CallKind.HF_HUB_DOWNLOAD, RepoType.MODEL, ("repo_id",))
    if name == "snapshot_download":
        return CallSpec(CallKind.SNAPSHOT_DOWNLOAD, RepoType.MODEL, ("repo_id",))

    # These boundaries intentionally use terminal names without import resolution.
    # A project-defined object with the same terminal name can therefore be a false
    # positive; keeping this conservative rule explicit avoids guessing provenance.
    if (
        name == "from_pretrained"
        and isinstance(function, cst.Attribute)
        and _terminal_name(function.value) == "PeftModel"
    ):
        return CallSpec(
            CallKind.PEFT_FROM_PRETRAINED,
            RepoType.MODEL,
            ("model_id",),
            positional_index=1,
        )
    if name == "pipeline":
        return CallSpec(
            CallKind.PIPELINE,
            RepoType.MODEL,
            ("model",),
            positional_index=None,
        )
    if name == "SentenceTransformer":
        return CallSpec(
            CallKind.SENTENCE_TRANSFORMER,
            RepoType.MODEL,
            ("model_name_or_path",),
        )
    if name == "from_pretrained" and isinstance(function, cst.Attribute):
        return CallSpec(
            CallKind.FROM_PRETRAINED,
            RepoType.MODEL,
            ("pretrained_model_name_or_path", "model_name_or_path"),
        )
    return None


def _terminal_name(expression: cst.BaseExpression) -> str | None:
    if isinstance(expression, cst.Name):
        return expression.value
    if isinstance(expression, cst.Attribute):
        return expression.attr.value
    return None


def _find_argument(
    call: cst.Call,
    keyword_names: tuple[str, ...],
    positional_index: int | None = 0,
) -> cst.BaseExpression | None:
    for argument in call.args:
        if argument.keyword is not None and argument.keyword.value in keyword_names:
            return argument.value
    if positional_index is None:
        return None
    positional = [
        argument.value
        for argument in call.args
        if argument.keyword is None and not argument.star
    ]
    if len(positional) > positional_index:
        return positional[positional_index]
    return None


def _literal_string(expression: cst.BaseExpression | None) -> str | None:
    if not isinstance(expression, cst.SimpleString):
        return None
    try:
        value = expression.evaluated_value
    except ValueError:
        return None
    return value if isinstance(value, str) else None


def _literal_value(expression: cst.BaseExpression) -> str | bool | None:
    string = _literal_string(expression)
    if string is not None:
        return string
    if isinstance(expression, cst.Name) and expression.value in {"True", "False"}:
        return expression.value == "True"
    return None


def _environment_reference(
    expression: cst.BaseExpression | None,
) -> _EnvironmentReference | None:
    """Recognize only the approved ``os.environ`` and ``os.getenv`` shapes."""

    if isinstance(expression, cst.Subscript):
        if not _is_os_environ(expression.value) or len(expression.slice) != 1:
            return None
        element = expression.slice[0]
        if not (
            isinstance(element.slice, cst.Index)
            and element.comma is cst.MaybeSentinel.DEFAULT
        ):
            return None
        name = _literal_string(element.slice.value)
        return _EnvironmentReference(name) if name else None

    if not isinstance(expression, cst.Call):
        return None
    arguments = expression.args
    if any(argument.keyword is not None or argument.star for argument in arguments):
        return None
    if (
        isinstance(expression.func, cst.Attribute)
        and expression.func.attr.value == "get"
        and _is_os_environ(expression.func.value)
    ):
        if len(arguments) != 1:
            return None
        name = _literal_string(arguments[0].value)
        return _EnvironmentReference(name) if name else None
    if not (
        isinstance(expression.func, cst.Attribute)
        and expression.func.attr.value == "getenv"
        and isinstance(expression.func.value, cst.Name)
        and expression.func.value.value == "os"
        and len(arguments) in {1, 2}
    ):
        return None
    name = _literal_string(arguments[0].value)
    if not name:
        return None
    fallback = _literal_string(arguments[1].value) if len(arguments) == 2 else None
    return _EnvironmentReference(name, fallback)


def _is_os_environ(expression: cst.BaseExpression) -> bool:
    return (
        isinstance(expression, cst.Attribute)
        and expression.attr.value == "environ"
        and isinstance(expression.value, cst.Name)
        and expression.value.value == "os"
    )


def _classify_data_files(expression: cst.BaseExpression | None) -> str:
    if expression is None:
        return "unknown"
    value = _literal_string(expression)
    if value is not None:
        normalized = value.strip().lower()
        if normalized.startswith(("hf://", "https://huggingface.co/")):
            return "hub"
        return "non_hub"
    if isinstance(expression, cst.FormattedString):
        if any(
            isinstance(part, cst.FormattedStringText)
            and any(
                marker in part.value.lower()
                for marker in ("hf://", "https://huggingface.co/")
            )
            for part in expression.parts
        ):
            return "hub"
        return "unknown"
    if isinstance(expression, (cst.List, cst.Tuple)):
        values = [
            element.value
            for element in expression.elements
            if isinstance(element, cst.Element)
        ]
        if len(values) != len(expression.elements):
            return "unknown"
        return _combine_data_files_kinds(values)
    if isinstance(expression, cst.Dict):
        values = [
            element.value
            for element in expression.elements
            if isinstance(element, cst.DictElement)
        ]
        if len(values) != len(expression.elements):
            return "unknown"
        return _combine_data_files_kinds(values)
    if isinstance(expression, cst.Call) and _is_os_path_join(expression.func):
        return "non_hub"
    return "unknown"


def _combine_data_files_kinds(expressions: list[cst.BaseExpression]) -> str:
    kinds = {_classify_data_files(expression) for expression in expressions}
    if "hub" in kinds:
        return "hub"
    if kinds == {"non_hub"}:
        return "non_hub"
    return "unknown"


def _is_os_path_join(expression: cst.BaseExpression) -> bool:
    return (
        isinstance(expression, cst.Attribute)
        and expression.attr.value == "join"
        and isinstance(expression.value, cst.Attribute)
        and expression.value.attr.value == "path"
        and isinstance(expression.value.value, cst.Name)
        and expression.value.value.value == "os"
    )


def _resolve_boolean(
    expression: cst.BaseExpression | None,
    resolve_name: Callable[[cst.Name], bool | None],
) -> tuple[bool, str | None]:
    if expression is None:
        return False, None
    if isinstance(expression, cst.Name):
        if expression.value in {"True", "False"}:
            return expression.value == "True", None
        value = resolve_name(expression)
        if value is not None:
            return value, None
        return False, (
            f"trust_remote_code name '{expression.value}' does not have one "
            "unambiguous boolean assignment"
        )
    return False, "trust_remote_code is a dynamic expression"


def _resolve_optional_string(
    expression: cst.BaseExpression | None,
    resolve_name: Callable[[cst.Name], str | None],
) -> str | None:
    if expression is None:
        return None
    literal = _literal_string(expression)
    if literal is not None:
        return literal
    if isinstance(expression, cst.Name):
        return resolve_name(expression)
    return None


def _resolve_revision(
    expression: cst.BaseExpression | None,
    resolve_name: Callable[[cst.Name], str | None],
) -> tuple[str | None, str | None]:
    if expression is None:
        return None, None
    literal = _literal_string(expression)
    if literal is not None:
        if literal.strip():
            return literal, None
        return None, "revision is empty or whitespace-only"
    if isinstance(expression, cst.Name):
        value = resolve_name(expression)
        if value is not None:
            if value.strip():
                return value, None
            return None, "revision is empty or whitespace-only"
        return (
            None,
            f"revision name '{expression.value}' does not have one "
            "unambiguous string assignment",
        )
    if isinstance(expression, cst.Subscript):
        reason = "revision is a subscript expression"
    elif isinstance(expression, cst.FormattedString):
        reason = "revision is an interpolated string"
    elif isinstance(expression, cst.Call):
        reason = "revision is returned by a function call"
    elif isinstance(expression, cst.IfExp):
        reason = "revision is a conditional expression"
    elif isinstance(expression, cst.Attribute):
        reason = "revision is an attribute expression"
    else:
        reason = "revision is a dynamic expression"
    return None, reason


def _resolve_repo_reference(
    expression: cst.BaseExpression | None,
    resolve_name: Callable[[cst.Name], str | None],
    resolve_environment_name: Callable[[cst.Name], _EnvironmentReference | None],
) -> tuple[str | None, _EnvironmentReference | None, str | None]:
    if expression is None:
        return None, None, "repository ID argument is missing"
    literal = _literal_string(expression)
    if literal is not None:
        return literal, None, None
    if isinstance(expression, cst.Name):
        value = resolve_name(expression)
        if value is not None:
            return value, None, None
        environment = resolve_environment_name(expression)
        if environment is not None:
            return None, environment, None
        return (
            None,
            None,
            f"repository ID name '{expression.value}' does not have one "
            "unambiguous string assignment",
        )
    environment = _environment_reference(expression)
    if environment is not None:
        return None, environment, None
    if isinstance(expression, cst.Subscript):
        reason = "repository ID is a subscript expression"
    elif isinstance(expression, cst.FormattedString):
        reason = "repository ID is an interpolated string"
    elif isinstance(expression, cst.Call):
        reason = "repository ID is returned by a function call"
    elif isinstance(expression, cst.IfExp):
        reason = "repository ID is a conditional expression"
    elif isinstance(expression, cst.Attribute):
        reason = "repository ID is an attribute expression"
    else:
        reason = "repository ID is a dynamic expression"
    return None, None, reason


def _missing_environment_binding_reason(reference: _EnvironmentReference) -> str:
    reason = (
        f"environment variable {reference.name!r} has no committed binding in "
        "[tool.hf-freeze.bindings.environment]"
    )
    if reference.fallback is not None:
        reason += (
            f"; literal fallback {reference.fallback!r} is not authoritative because "
            "the runtime environment may override it"
        )
    return reason


def _repo_type(
    call: cst.Call,
    default: RepoType,
    resolve_name: Callable[[cst.Name], str | None],
) -> RepoType | None:
    expression = _find_argument(call, ("repo_type",), positional_index=None)
    if expression is None:
        return default
    value = _resolve_optional_string(expression, resolve_name)
    if value == RepoType.MODEL.value:
        return RepoType.MODEL
    if value == RepoType.DATASET.value:
        return RepoType.DATASET
    return None


def _is_obvious_local_path(repo_id: str) -> bool:
    normalized = repo_id.strip()
    lowered = normalized.lower()
    return (
        normalized in {".", "..", "~"}
        or normalized.startswith(("./", "../", ".\\", "..\\", "/", "\\", "~/", "~\\"))
        or lowered.startswith("file:")
        or re.match(r"^[a-zA-Z]:(?:[\\/]|[^/\\])", normalized) is not None
    )
