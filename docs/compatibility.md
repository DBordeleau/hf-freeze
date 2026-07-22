# Representative project compatibility

This page records bounded validation of the `hf-freeze` 0.1.0 wheel. It is
evidence about the exact source revisions and scopes below, not a claim of
complete framework or ecosystem support.

## Slice 18 expanded-contract headline

- **12/12 exact-commit full-tree scans completed** without importing or
  executing target code. Only the three earlier positive projects were
  unconfigured full-tree lock candidates; the other nine retained **918 visible
  unresolved call sites** rather than claiming incomplete coverage.
- **11/12 configured application scopes produced non-empty deterministic locks
  and source-pin diffs.** Their 31 lockable sites produced 21 reviewed lock
  entries and 31 exact source pins.
- **1/12, TSDAE, was an intentionally dynamic warning-only lifecycle** with two
  acknowledged calls, an empty lock, and an empty pin diff. Its successful
  command sequence is not evidence that dependencies were frozen.
- All 12 configured command sequences exited successfully through `scan` ->
  `lock` -> pin preview -> pin write -> `check --frozen`. Across those scopes,
  coverage was 15 static, 4 environment-bound, 12 annotation-bound, 2
  acknowledged-dynamic, and 0 unresolved call sites.
- Two production Spaces proved that a committed environment binding is
  authoritative: scan classifications and `hf.lock` bytes were identical with
  `MODEL_ID` absent and set to a conflicting ambient value.

The configured audited base and actual source HEAD were both
`2ca7646c4ebc510b71ad905d202e216183da1ae9`. The built wheel SHA-256 was
`86fabe4e814a82f01c643491e126afb723c9241c01e653912ce9e54d3686616a`.
The exact bounded input is the [Slice 18 manifest](../validation/slice18/manifest.json),
and the reviewed normalized output is [results.json](../validation/slice18/results.json).
Raw command logs and disposable checkouts remained outside the tracked tree.

## Slice 18 method

The validator fetched each full commit into a new disposable checkout, ran an
unconfigured full-tree scan, applied the tracked overlay, and ran the configured
lifecycle through the built wheel:

```text
hf-freeze scan .
hf-freeze lock .
hf-freeze pin .
hf-freeze pin . --write
hf-freeze check . --frozen
```

The tool was invoked through isolated `uv tool run` environments refreshed from
the exact wheel bytes. Public Hub metadata resolution was allowed; target
packages, target source, remote code, and model weights were never installed,
imported, or executed. A fresh empty `HF_HOME` and a credential-stripped process
environment kept ambient values and cached authentication out of dependency
truth. The deterministic validator and its offline tests are tracked in
[`scripts/validate_slice18.py`](../scripts/validate_slice18.py) and
[`tests/test_slice18_validation.py`](../tests/test_slice18_validation.py).

Coverage tuples below use the stable order **static / environment / annotation /
acknowledged / unresolved** and count source call sites, not deduplicated lock
rows.

## Slice 18 project matrix

| Repository and exact commit | Unconfigured full tree | Configured scope and exact overlay | Configured coverage | Lifecycle |
| --- | --- | --- | --- | --- |
| [`tloen/alpaca-lora` at `8bb8579e403dc78e37fe81ffbb253c413007323f`](https://github.com/tloen/alpaca-lora/tree/8bb8579e403dc78e37fe81ffbb253c413007323f) | `2/0/0/0/15`; unsupported expressions remain visible. | README-documented export workflow; scope plus two dependency annotations for `BASE_MODEL`. [Overlay](../validation/slice18/overlays/alpaca-lora.patch) | `1/0/2/0/0` | Complete; 2 lock entries, 3 pinned sites, frozen check exit 0. |
| [`louisbrulenaudet/tsdae` at `82b47f5e324636d8ba8b02a46f2dde44a7c86066`](https://github.com/louisbrulenaudet/tsdae/tree/82b47f5e324636d8ba8b02a46f2dde44a7c86066) | `0/0/0/0/2`; caller-selected datasets are intentionally dynamic. | Packaged `src/tsdae` scope plus per-call ignore reasons. [Overlay](../validation/slice18/overlays/tsdae.patch) | `0/0/0/2/0` | Warning-complete; empty lock, empty pin diff, frozen check exit 0 with both calls visible. |
| [`cloneofsimo/lora` at `d84074b3e3496f1cfa8a3f49b8b9972ef463b483`](https://github.com/cloneofsimo/lora/tree/d84074b3e3496f1cfa8a3f49b8b9972ef463b483) | `0/0/0/0/42`; command- and function-selected models remain dynamic. | Documented preprocessing utility plus declarations and five dependency annotations. [Overlay](../validation/slice18/overlays/cloneofsimo-lora.patch) | `0/0/5/0/0` | Complete; 3 lock entries, 5 pinned sites, frozen check exit 0. |
| [`huggingface/transformers-bloom-inference` at `62698bf4b75a105e0774ff77b43a4ee572d7b3da`](https://github.com/huggingface/transformers-bloom-inference/tree/62698bf4b75a105e0774ff77b43a4ee572d7b3da) | `1/0/0/0/19`; the mixed server tree is not one lockable application. | Repository-documented root `ui.py` client. [Overlay](../validation/slice18/overlays/transformers-bloom-inference.patch) | `1/0/0/0/0` | Complete; 1 lock entry and pin, frozen check exit 0. |
| [`huggingface/peft` at `cea8213158c8b682acc0839405c2062d57fdf867`](https://github.com/huggingface/peft/tree/cea8213158c8b682acc0839405c2062d57fdf867) | `403/0/0/0/828`; framework examples/tests remain a broad safety stress test. | Complete `arrow_multitask` example application. [Overlay](../validation/slice18/overlays/peft.patch) | `8/0/0/0/0` | Complete; 6 lock entries, 8 pinned sites, frozen check exit 0. |
| [`luvris2/streamlit_chatbot` at `baf2ca43e0c3bc8971c0b944e268c1985ad0149d`](https://github.com/luvris2/streamlit_chatbot/tree/baf2ca43e0c3bc8971c0b944e268c1985ad0149d) | `1/0/0/0/0`; already lockable. | README-identified Streamlit `app.py`. [Overlay](../validation/slice18/overlays/streamlit-chatbot.patch) | `1/0/0/0/0` | Complete; 1 lock entry and pin, frozen check exit 0. |
| [`fun-research/TiTok` at `c08e40ef451698ea3f2f831ca35f415d235bbf79`](https://huggingface.co/spaces/fun-research/TiTok/tree/c08e40ef451698ea3f2f831ca35f415d235bbf79) | `2/0/0/0/0`; already lockable. | Space metadata `app.py`. [Overlay](../validation/slice18/overlays/titok.patch) | `2/0/0/0/0` | Complete; 1 lock entry, 2 pinned sites, frozen check exit 0. |
| [`nik-dim/tall_masks` at `3b1b2ee365b8d734d84cf90f363d265fe3e54cdf`](https://github.com/nik-dim/tall_masks/tree/3b1b2ee365b8d734d84cf90f363d265fe3e54cdf) | `2/0/0/0/0`; already lockable. | Dataset adapter used by documented training/evaluation entry points. [Overlay](../validation/slice18/overlays/tall-masks.patch) | `2/0/0/0/0` | Complete; 1 lock entry, 2 pinned sites, frozen check exit 0. |
| [`sahil147/sentiment-api` at `59775cbd780dbfd1901990b100bca2e9a0beb405`](https://huggingface.co/spaces/sahil147/sentiment-api/tree/59775cbd780dbfd1901990b100bca2e9a0beb405) | `0/0/0/0/4`; app and maintenance scripts use dynamic model selection. | Docker Space FastAPI `app` package; the module-level environment value used inside a function requires a call annotation. [Overlay](../validation/slice18/overlays/sentiment-api.patch) | `0/0/1/0/0` | Complete; 1 lock entry and pin, frozen check exit 0. |
| [`likhonsheikhdev/docker-model-runner` at `ab0cf4fb45b9962951619c33983a0b1836f43d29`](https://huggingface.co/spaces/likhonsheikhdev/docker-model-runner/tree/ab0cf4fb45b9962951619c33983a0b1836f43d29) | `0/0/0/0/4`; all deployed models are environment-selected. | Docker Space root FastAPI `main.py`; outer-scope environment values require four call annotations. [Overlay](../validation/slice18/overlays/docker-model-runner.patch) | `0/0/4/0/0` | Complete; 3 lock entries, 4 pinned sites, frozen check exit 0. |
| [`warshanks/medgemma-4b-it` at `e2b9b1aaabf0e513351dee752e305ec4df7f6e60`](https://huggingface.co/spaces/warshanks/medgemma-4b-it/tree/e2b9b1aaabf0e513351dee752e305ec4df7f6e60) | `0/0/0/0/2`; no committed binding exists upstream. | Space metadata `app.py` plus committed `MODEL_ID` binding. [Overlay](../validation/slice18/overlays/medgemma-space.patch) | `0/2/0/0/0` | Complete; 1 lock entry, 2 pinned sites, frozen check exit 0; absent/conflicting ambient scans and locks equivalent. |
| [`Marcel0123/emotie-herkennen-gezichten-mens` at `5efa22b5f79bd9e0fadb284aec803cb506fadccc`](https://huggingface.co/spaces/Marcel0123/emotie-herkennen-gezichten-mens/tree/5efa22b5f79bd9e0fadb284aec803cb506fadccc) | `0/0/0/0/2`; no committed binding exists upstream. | Space metadata `app.py` plus committed `MODEL_ID` binding. [Overlay](../validation/slice18/overlays/emotion-space.patch) | `0/2/0/0/0` | Complete; 1 lock entry, 2 pinned sites, frozen check exit 0; absent/conflicting ambient scans and locks equivalent. |

## Defects found and fixed

The cohort exposed two high-impact correctness defects, both covered by focused
regressions:

- Chained calls such as `from_pretrained(...).to(device)` were safely rewritten
  but also falsely reported as skipped because the wrapper and inner call share
  a source position. Pin now treats only the matching inner call as the target.
- A Windows legacy console encoding could make pin preview fail when untouched
  source context contained Unicode. Preview now retries exact output through a
  UTF-8 stdout stream.

## Release-decision matrix

| Remaining limitation | Type | Observed impact | Recommendation |
| --- | --- | --- | --- |
| One lock still cannot represent an entire mixed framework/test/example tree when any call is unresolved or revisions conflict. | Intentional safety behavior; application-scope decision | Nine unconfigured trees retained unresolved findings; every declared application scope completed. | **Non-blocker** for release preparation. Keep strict refusal and describe scope as the guarantee boundary. |
| TSDAE accepts arbitrary caller-selected datasets. | Intentionally dynamic design | Two calls pass only as visible acknowledged warnings and create no lock entries. | **Non-blocker** when the warning is acceptable; users needing a fixed deployment should annotate a declaration instead. |
| Module-level environment references consumed inside a function are outside the initial same-scope binding contract. | Tool limitation / unsupported expression | Two production services needed explicit per-call annotations despite committed environment tables. | **Non-blocker for 0.1**, because the canonical annotation is deterministic; reconsider only with repeated alpha demand. |
| `os.environ.get("NAME", None)` is not one of the approved environment-expression forms. | Tool limitation / unsupported expression | Alpaca-LoRA needed two dependency annotations for its export workflow. | **Non-blocker**; do not broaden expression semantics from one example. |
| Multiline calls without an existing trailing comma can receive awkward but valid revision formatting. | Tool limitation | Reviewed diffs remained valid, limited to the call, and passed frozen checks, but some should be formatter-reviewed before commit. | **Non-blocker**, with dry-run review retained; consider a focused follow-up only if users report friction. |
| Generic `from_pretrained` discovery is call-shape based rather than full import provenance. | Static-analysis limitation | Strict unresolved handling prevented silent freezing, but the 1,329 full-tree call sites were not an exhaustive semantic proof. | **Non-blocker** for the bounded claim; do not market complete framework coverage. |
| Hub metadata and repository availability can change after validation. | External network/retention limitation | All 21 entries resolved during this run; no runtime compatibility or retention claim was tested. | **Non-blocker**; preserve exact commits, wheel hash, and point-in-time wording. |

Recommendation: the remaining results do **not** block moving to an explicit
human release-preparation decision. They do block any claim that arbitrary
whole repositories, excluded files, ignored calls, deployment environment
values, or runtime compatibility are frozen. The manager and user retain the
final blocker/non-blocker decision.

## Historical Slice 11 evidence

The later application-scope, named-declaration, committed environment-binding,
source-directive, and acknowledged-dynamic mechanisms were not used in these
eight historical validations. Slice 18 does not retroactively turn the **0/5**
broad result into historical success or expand the historical **3/3**
product-fit cohort. The original evidence remains below unchanged in substance.

## Historical validation method

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
