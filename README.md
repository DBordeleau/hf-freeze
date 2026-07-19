# hf-freeze

`hf-freeze` is a local Python CLI for the gap between a package lockfile and a
Hugging Face Hub call such as `AutoModel.from_pretrained("org/model")`. It finds
supported literal references, resolves a moving revision to an immutable commit,
writes a deterministic `hf.lock`, and helps you review and apply exact source
pins. That makes the Hub snapshot your project expects visible and reviewable in
Git—without running your project or downloading model weights to create the lock.

## 60-second quickstart

From the root of a Python project with supported Hugging Face Hub calls, start
with these three commands:

```bash
hf-freeze scan
hf-freeze lock
hf-freeze pin
```

Review the pin diff, then explicitly write and verify it without network access:

```bash
hf-freeze pin --write
hf-freeze check --frozen
```

`scan` discovers supported literal calls; `lock` resolves their tracking
revisions and writes `hf.lock`; dry-run `pin` shows the source diff. Only after
reviewing that diff should `pin --write` make the accepted immutable SHAs
explicit in supported source calls.

## Install

Python 3.10+ is required.

Install directly from GitHub with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install git+https://github.com/DBordeleau/hf-freeze.git
```

Or install from a source checkout:

```bash
git clone https://github.com/DBordeleau/hf-freeze.git
cd hf-freeze
uv tool install .
```

## From source to `hf.lock`, then a reviewed update

Run these commands at the root of the Python project whose Hub calls you want to
track:

```bash
hf-freeze scan
hf-freeze lock
hf-freeze pin
hf-freeze pin --write
hf-freeze check --frozen
hf-freeze diff org/model
hf-freeze update org/model
hf-freeze update org/model --write
hf-freeze pin
hf-freeze pin --write
hf-freeze check --frozen
```

`scan` shows supported findings and unresolved dynamic values. `lock` resolves
the current requested revisions and writes deterministic JSON to `hf.lock`.
`pin` previews the minimal source changes by default; use `pin --write` only
after reviewing that diff. Later, `diff` is review-only: it compares the locked
snapshot with the current candidate for its stored tracking revision. `update`
shows the same review and is a dry run by default; only `update --write` accepts
the reviewed snapshot into `hf.lock`. It deliberately does not edit source, so
run the separate `pin --write` step and then the offline `check --frozen` guard.

`hf.lock` complements, rather than replaces, other tools. Python lockfiles such
as `uv.lock`, Poetry lockfiles, and pip requirement locks lock Python packages,
not Hugging Face Hub repositories. The Hugging Face cache stores local artifacts,
but is neither a project-level dependency declaration nor a reviewed acceptance
record. `hf.lock` records the accepted Hub repository snapshots and their
tracking revisions for the project.

## Commands and safety

| Command | Purpose | Network |
| --- | --- | --- |
| `hf-freeze scan [PATH]` | Statically discover supported Python calls. | No |
| `hf-freeze lock [PATH]` | Resolve known revisions and atomically write `hf.lock`. | Hub metadata only; no weights |
| `hf-freeze check [PATH] --frozen` | Check source/lock coverage for CI. | No |
| `hf-freeze diff REPO_ID [--revision REV]` | Compare the locked SHA to a candidate revision. | Hub metadata; only allowlisted small JSON files may be read for semantic comparison |
| `hf-freeze update REPO_ID [--revision REV] [--write]` | Preview a repository update; `--write` atomically accepts it into `hf.lock` only. | Hub metadata; only allowlisted bounded small JSON files may be read for semantic comparison; no weights |
| `hf-freeze pin [PATH] [--write]` | Preview, then optionally atomically apply, exact source pins. | No |

`scan` succeeds when it can report findings, including unresolved ones. `lock`
refuses to write when supported findings are unresolved or conflict. `check
--frozen` and a `pin` run with skipped unsafe targets exit nonzero so CI cannot
mistake incomplete coverage for success. `diff` reports changes as information;
repository, lockfile, or Hub errors are failures. `pin` is dry-run by default and
does not edit files unless `--write` is explicit. `update` is also dry-run by
default; it changes only `hf.lock` when `--write` is explicit, never source.

### Supported call shapes

The prototype recognizes literal strings and simple same-scope string constants
in these forms:

- `*.from_pretrained("repo/id", ...)` (including common Diffusers forms)
- `load_dataset("repo/id", ...)`
- `hf_hub_download(repo_id="repo/id", ...)`
- `snapshot_download(repo_id="repo/id", ...)`
- `SentenceTransformer("repo/id", ...)`
- `PeftModel.from_pretrained(base_model, "repo/id", ...)` or `model_id=`
- `pipeline(..., model="repo/id", ...)`

It scans source; it does not import or execute your project. Dynamic IDs,
interpolated strings, imported configuration, and unsupported pipeline forms are
reported rather than resolved silently.

See [representative project compatibility](docs/compatibility.md) for the
methodology and exact-commit results from five public repositories. That table
records observed behavior and does not imply broad framework support.

## Complete lifecycle demo

This compact terminal walkthrough uses synthetic repository metadata from the
offline fake-backed lifecycle test. It demonstrates the supported review-first
flow; it is not output from the real example below, and it does not execute a
model or download weights.

![Synthetic terminal output showing dry-run pin before each explicit source write, offline frozen checks, and dry-run update before explicit lock acceptance.](docs/demo.svg)

## Immutable demo: tiny-random-bert

[`examples/tiny-random-bert`](examples/tiny-random-bert) is a real, intentionally
floating call site. Its committed `hf.lock` records the historical Hub commit
`9b8c223d42b2188cb49d29af482996f9d0f3e5a6`; that mismatch is deliberate until
you apply the preview from `pin`.

```bash
cd examples/tiny-random-bert
hf-freeze scan .
hf-freeze diff hf-internal-testing/tiny-random-bert --revision 8fc97e155588266e09c9f37d4a9608e1a65a279e
hf-freeze pin .
```

The exact revision comparison is reproducible: metadata reports the candidate's
added `model.safetensors` weight artifact (520,212 bytes). It does not execute a
model or download the weight/artifact. Because the source intentionally floats,
`hf-freeze check . --frozen` is expected to fail until the reviewed pin is
applied; do not commit the example after applying that preview.

## Limitations

- Discovery is static, Python-only, and intentionally incomplete; it is not a
  behavioral or security analysis.
- It does not provide complete transitive dependency discovery, notebook support,
  a strict file manifest, artifact mirroring, or runtime enforcement.
- Metadata operations avoid weight downloads. `diff` may read only an allowlisted
  small configuration JSON file when it needs a bounded semantic comparison.
- A pinned Hub commit identifies a snapshot, but this prototype does not claim
  broad library compatibility, model quality, security, or future Hub retention.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under [Apache-2.0](LICENSE).
