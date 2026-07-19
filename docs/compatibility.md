# Representative project compatibility

This page records a bounded manual validation of the `hf-freeze` 0.1.0 wheel.
It is evidence about the exact source revisions below, not a claim of complete
framework or ecosystem support.

## Method

On 2026-07-19, each public repository was shallow-cloned into a disposable
directory and pinned to the full Git commit shown below. The wheel was built
from this repository, launched with `uv tool run --isolated --from <wheel>`, and
used to scan every Python file in each checkout. No target project was imported,
installed, or executed.

A repository-wide `lock` was then attempted in each disposable checkout. Where
the full tree correctly refused to lock because findings were unresolved or
conflicted, a source file containing only audited literal findings was also
locked when available. These focused lock runs queried public Hub metadata; they
did not download model weights or execute remote code.

## Observed results

| Repository and exact commit | Observed call shapes | Full-tree scan | Lock outcome |
| --- | --- | --- | --- |
| [`tloen/alpaca-lora` at `8bb8579e403dc78e37fe81ffbb253c413007323f`](https://github.com/tloen/alpaca-lora/tree/8bb8579e403dc78e37fe81ffbb253c413007323f) | Transformers `from_pretrained`, Datasets `load_dataset`, and PEFT adapter loading | Exit 0; 2 supported literal PEFT findings and 15 actionable unresolved findings | Refused unresolved findings; no `hf.lock` written |
| [`louisbrulenaudet/tsdae` at `82b47f5e324636d8ba8b02a46f2dde44a7c86066`](https://github.com/louisbrulenaudet/tsdae/tree/82b47f5e324636d8ba8b02a46f2dde44a7c86066) | Dynamic Datasets loading; a non-Hub `SentenceTransformer(modules=...)` construction | Exit 0; 2 actionable unresolved dataset findings; the non-Hub constructor was ignored | Refused 2 unresolved findings; no `hf.lock` written |
| [`cloneofsimo/lora` at `d84074b3e3496f1cfa8a3f49b8b9972ef463b483`](https://github.com/cloneofsimo/lora/tree/d84074b3e3496f1cfa8a3f49b8b9972ef463b483) | Diffusers and Transformers component `from_pretrained` calls driven by command-line values | Exit 0; 42 actionable unresolved findings | Refused 42 unresolved findings; no `hf.lock` written |
| [`huggingface/transformers-bloom-inference` at `62698bf4b75a105e0774ff77b43a4ee572d7b3da`](https://github.com/huggingface/transformers-bloom-inference/tree/62698bf4b75a105e0774ff77b43a4ee572d7b3da) | Transformers `from_pretrained` and `snapshot_download` | Exit 0; 1 supported literal finding and 19 actionable unresolved findings | Full tree refused 19 unresolved findings; focused `ui.py` lock succeeded with 1 dependency |
| [`huggingface/peft` at `cea8213158c8b682acc0839405c2062d57fdf867`](https://github.com/huggingface/peft/tree/cea8213158c8b682acc0839405c2062d57fdf867) | Transformers, Datasets, PEFT adapter loading, `hf_hub_download`, and `snapshot_download` | Exit 0; 403 supported literal findings and 828 actionable unresolved findings, including 2 `hf://` dataset inputs | Full tree refused conflicting revisions before writing; focused `examples/arrow_multitask/arrow_phi3_mini.py` lock succeeded with 6 dependencies |

The supported-file lock checks covered one literal Transformers dependency and
a second file containing literal Transformers and Datasets dependencies. The
full scans collectively exercised Transformers, Datasets, PEFT, Diffusers
component loading, and direct Hub download/snapshot call families.

## Manual audit checklist

- [x] Confirmed each checkout's full commit SHA before scanning.
- [x] Confirmed all five full-tree scans exited successfully with no parse or
  read diagnostics.
- [x] Compared literal PEFT findings in `alpaca-lora` with the corresponding
  `PeftModel.from_pretrained` source calls.
- [x] Compared the literal `bigscience/bloom` finding and dynamic
  `snapshot_download` diagnostics with `transformers-bloom-inference` source.
- [x] Compared representative PEFT repository findings for namespaced datasets,
  Transformers models, adapters, and direct `hf_hub_download` calls with source.
- [x] Confirmed dynamic Diffusers/Transformers arguments in `cloneofsimo/lora`
  were reported as unresolved instead of guessed.
- [x] Confirmed `SentenceTransformer(modules=[...])` and Datasets builders with
  confidently local `data_files` are not Hub dependencies; confirmed recognizable
  `hf://` and `https://huggingface.co/...` inputs remain visible as actionable
  unresolved findings instead of fake `json` or `parquet` repository IDs.
- [x] Found no sampled high-impact false positive that the final wheel could
  silently lock. Repository-wide incomplete coverage failed before writing.
- [x] Confirmed focused lock runs created only external disposable `hf.lock`
  files and used no credentials, project execution, remote code, or weight
  downloads.

## Limitations

- This is a five-repository snapshot at exact commits, not a compatibility
  guarantee for later commits or every version of the named libraries.
- The scanner is static and Python-only. Runtime values, imported configuration,
  interpolated IDs, and cross-module data flow remain unresolved by design.
- The repository totals include tests and examples because the full checkout was
  scanned. A narrower application path can produce a smaller, lockable set.
- Generic `from_pretrained` matching is based on call shape rather than complete
  import provenance. The audit sampled both supported and unresolved findings;
  it was not an exhaustive proof over all 1,312 reported call sites.
- Sentence Transformers model loading did not receive a positive literal finding
  in this set. The observed `SentenceTransformer(modules=...)` call was local
  construction and is intentionally not treated as a Hub dependency.
- Locking is strict: one unresolved finding or incompatible revision set prevents
  a repository-wide lockfile. This avoids silently claiming complete coverage.
