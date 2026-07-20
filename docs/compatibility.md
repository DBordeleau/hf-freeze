# Representative project compatibility

This page records bounded manual validation of the `hf-freeze` 0.1.0 wheel.
It is evidence about the exact source revisions and scopes below, not a claim
of complete framework or ecosystem support.

## Headline results

- **Broad safety cohort:** 5/5 repository-wide scans completed without a crash,
  but **0/5 full repository trees produced a lockfile**. Strict locking refused
  unresolved or conflicting dependencies instead of claiming incomplete
  coverage.
- **Product-fit lifecycle cohort:** **3/3 natural project scopes completed**
  `scan` -> `lock` -> pin preview -> pin write -> frozen check. Each started
  with floating literal Hub dependencies and produced a reviewed source-pin
  diff.
- Across both cohorts, eight public repositories were evaluated at exact
  commits. The positive cohort exercised three supported call families:
  Sentence Transformers, direct Hub download, and Datasets loading.

## Validation method

On 2026-07-19, each public repository was shallow-cloned into a disposable
directory and pinned to the full Git commit shown below. The wheel was built
from this repository and invoked as an isolated tool:

```text
uv tool run --isolated --from <built-wheel> hf-freeze <command>
```

No target project was installed, imported, or executed. Initial scans ran
before any generated lockfile or source edit. Locking queried public Hub
metadata only; it did not download model weights or execute remote code.
Generated `hf.lock` files and source edits remained in disposable checkouts
outside the `hf-freeze` repository.

## Successful product-fit lifecycles

These repositories are consumer applications or research projects that use
supported Hub libraries; none is a Hugging Face framework repository or a
broad test/example collection. Each scope is the repository root, which is the
natural application or documented command location rather than a hand-picked
source file.

| Repository and exact commit | Natural scope and call family | Initial scan | Lock and reviewed pin | Write and frozen check |
| --- | --- | --- | --- | --- |
| [`luvris2/streamlit_chatbot` at `baf2ca43e0c3bc8971c0b944e268c1985ad0149d`](https://github.com/luvris2/streamlit_chatbot/tree/baf2ca43e0c3bc8971c0b944e268c1985ad0149d) | Repository root; the project identifies `app.py` as its main Streamlit application. Sentence Transformers constructor. | Exit 0; 1 floating literal finding for `jhgan/ko-sroberta-multitask`. | Exit 0; 1 dependency. Pin preview added resolved SHA `8fca7c9c98c26599be0e14b9916b11a756a26f19` to the constructor. | `pin --write` exit 0 changed `app.py`; `check --frozen` exit 0. |
| [`fun-research/TiTok` Space at `c08e40ef451698ea3f2f831ca35f415d235bbf79`](https://huggingface.co/spaces/fun-research/TiTok/tree/c08e40ef451698ea3f2f831ca35f415d235bbf79) | Repository root; Space metadata declares root `app.py` as the Gradio entry point. Direct `hf_hub_download`. | Exit 0; 2 floating literal call sites for `fun-research/TiTok`. | Exit 0; 1 merged dependency. Pin preview added resolved SHA `ab646ed225080a3acb7c78440a574d7f67f16fa7` to both download calls. | `pin --write` exit 0 changed `app.py`; `check --frozen` exit 0. |
| [`nik-dim/tall_masks` at `3b1b2ee365b8d734d84cf90f363d265fe3e54cdf`](https://github.com/nik-dim/tall_masks/tree/3b1b2ee365b8d734d84cf90f363d265fe3e54cdf) | Repository root; the README documents root-level training and evaluation commands. Datasets `load_dataset`. | Exit 0; 2 floating literal call sites for `Jeneral/fer-2013`. | Exit 0; 1 merged dependency. Pin preview added resolved SHA `3a46cbfae3f5b348449335f300666a0ae330f121` to the train and test loads. | `pin --write` exit 0 changed `src/datasets/fer2013.py`; `check --frozen` exit 0. |

For every row, the exact command sequence and exit codes were:

```text
hf-freeze scan <scope>             # 0
hf-freeze lock <scope>             # 0
hf-freeze pin <scope>              # 0
hf-freeze pin <scope> --write      # 0
hf-freeze check <scope> --frozen   # 0
```

The reviewed diffs inserted only literal `revision="<resolved-sha>"` arguments:
one call in `streamlit_chatbot`, two calls in `TiTok`, and two calls in
`tall_masks`. No other target source changed.

## Broad safety cohort

The original five-project cohort remains the broader stress test. Its scopes
included whole framework, test, and example trees to expose unsupported and
conflicting patterns. Those scans provide safety evidence, not successful
compatibility claims: **0/5 full trees locked**.

| Repository and exact commit | Observed call shapes | Full-tree scan | Full-tree lock outcome |
| --- | --- | --- | --- |
| [`tloen/alpaca-lora` at `8bb8579e403dc78e37fe81ffbb253c413007323f`](https://github.com/tloen/alpaca-lora/tree/8bb8579e403dc78e37fe81ffbb253c413007323f) | Transformers `from_pretrained`, Datasets `load_dataset`, and PEFT adapter loading | Exit 0; 2 supported literal PEFT findings and 15 actionable unresolved findings | Refused unresolved findings; no `hf.lock` written |
| [`louisbrulenaudet/tsdae` at `82b47f5e324636d8ba8b02a46f2dde44a7c86066`](https://github.com/louisbrulenaudet/tsdae/tree/82b47f5e324636d8ba8b02a46f2dde44a7c86066) | Dynamic Datasets loading; a non-Hub `SentenceTransformer(modules=...)` construction | Exit 0; 2 actionable unresolved dataset findings; the non-Hub constructor was ignored | Refused 2 unresolved findings; no `hf.lock` written |
| [`cloneofsimo/lora` at `d84074b3e3496f1cfa8a3f49b8b9972ef463b483`](https://github.com/cloneofsimo/lora/tree/d84074b3e3496f1cfa8a3f49b8b9972ef463b483) | Diffusers and Transformers component `from_pretrained` calls driven by command-line values | Exit 0; 42 actionable unresolved findings | Refused 42 unresolved findings; no `hf.lock` written |
| [`huggingface/transformers-bloom-inference` at `62698bf4b75a105e0774ff77b43a4ee572d7b3da`](https://github.com/huggingface/transformers-bloom-inference/tree/62698bf4b75a105e0774ff77b43a4ee572d7b3da) | Transformers `from_pretrained` and `snapshot_download` | Exit 0; 1 supported literal finding and 19 actionable unresolved findings | Refused 19 unresolved findings; no full-tree `hf.lock` written |
| [`huggingface/peft` at `cea8213158c8b682acc0839405c2062d57fdf867`](https://github.com/huggingface/peft/tree/cea8213158c8b682acc0839405c2062d57fdf867) | Transformers, Datasets, PEFT adapter loading, `hf_hub_download`, and `snapshot_download` | Exit 0; 403 supported literal findings and 828 actionable unresolved findings, including 2 `hf://` dataset inputs | Refused conflicting revisions; no full-tree `hf.lock` written |

Focused diagnostic runs in this first cohort showed that `ui.py` in
`transformers-bloom-inference` could lock 1 literal dependency and
`examples/arrow_multitask/arrow_phi3_mini.py` in PEFT could lock 6. They were
useful scanner checks, but they are deliberately not counted as successful
project lifecycles because they were hand-picked files rather than natural
application scopes.

## Manual audit checklist

- [x] Confirmed every checkout's full commit SHA before scanning.
- [x] Confirmed the five broad full-tree scans and three product-fit root scans
  exited successfully with no parse or read diagnostics.
- [x] Confirmed all three product-fit scopes were clean before their initial
  scan and lock.
- [x] Reviewed each generated lock entry and pin preview before applying it.
- [x] Reviewed the resulting Python diff and completed an offline frozen check
  for every positive lifecycle.
- [x] Compared literal PEFT findings in `alpaca-lora` with the corresponding
  `PeftModel.from_pretrained` source calls.
- [x] Compared the literal `bigscience/bloom` finding and dynamic
  `snapshot_download` diagnostics with `transformers-bloom-inference` source.
- [x] Compared representative PEFT findings for namespaced datasets,
  Transformers models, adapters, and direct Hub calls with source.
- [x] Confirmed dynamic Diffusers/Transformers arguments in `cloneofsimo/lora`
  were reported as unresolved instead of guessed.
- [x] Confirmed `SentenceTransformer(modules=[...])` and Datasets builders with
  confidently local `data_files` are not Hub dependencies; recognizable
  `hf://` and `https://huggingface.co/...` inputs remain actionable unresolved
  findings instead of fake `json` or `parquet` repository IDs.
- [x] Found no sampled high-impact false positive that the wheel could silently
  lock. Incomplete full-tree coverage failed before writing.
- [x] Used no credentials, project execution, remote code, or weight downloads.

## Failures and limitations

- The broad cohort's **0/5 full-tree lock result is unresolved by design**, not
  converted into success evidence. One unresolved finding or incompatible
  revision set still prevents a lockfile.
- These results are snapshots at eight exact commits, not guarantees for later
  commits or every version of the named libraries.
- The scanner is static and Python-only. Runtime values, imported configuration,
  interpolated IDs, and cross-module data flow remain unresolved by design.
- Generic `from_pretrained` matching is based on call shape rather than complete
  import provenance. The broad audit sampled supported and unresolved findings;
  it was not an exhaustive proof over all 1,312 reported call sites.
- The positive cohort contains deliberately product-fit natural scopes. It does
  not demonstrate that arbitrary large repositories with mixed examples and
  tests can produce one complete lockfile.
- Validation used public Hub metadata at one point in time. It did not test
  private repositories, authenticated access, project runtime compatibility,
  model quality, or artifact retention.
