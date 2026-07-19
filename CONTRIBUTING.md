# Contributing to hf-freeze

Thanks for helping improve this early prototype. `hf-freeze` is a small local CLI
for discovering and locking supported Hugging Face Hub references in Python source.
Please keep changes narrowly focused, deterministic, and clear about unsupported
cases.

## Local setup

Install [uv](https://docs.astral.sh/uv/), then from a clone run:

```bash
uv sync --all-groups
uv run hf-freeze --help
```

## Verification

Run a focused test while changing a specific behavior, for example:

```bash
uv run pytest tests/test_scan.py -q
```

Before proposing a change, run the fast default suite and checks:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The default suite must stay fast and use no network. Use small fixtures and fake
Hub resolvers; never download model weights or execute model code in tests.
Optional manual metadata checks belong outside the default suite.

## Prototype scope

The scanner intentionally supports only documented Python call shapes and reports
dynamic values instead of guessing. Avoid adding runtime execution, artifact
downloads, broad compatibility promises, or unrelated abstractions without a
concrete product need. See the public [README](README.md) for current behavior
and limitations.

## Private context

Do not add private planning, local notes, credentials, tokens, or machine-specific
context to tracked files, issues, or pull requests. Public documentation should
describe only verified public product behavior.
