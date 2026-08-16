# boyle

[![ci](https://github.com/beatakouchnir/boyle/actions/workflows/ci.yml/badge.svg)](https://github.com/beatakouchnir/boyle/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/boyle)](https://pypi.org/project/boyle/)
[![license](https://img.shields.io/pypi/l/boyle)](https://github.com/beatakouchnir/boyle/blob/main/LICENSE)

**Run the model you want at the memory pressure you specify.**

Declare a memory budget; boyle runs mixture-of-experts models inside it on
Apple silicon — including models far larger than RAM — with decode outputs
**bit-identical** to the fully-resident model, a **speed forecast before you
download anything**, and an OpenAI- and Ollama-compatible server your
existing tools connect to.

```bash
boyle predict mlx-community/Qwen3.5-397B-A17B-4bit --budget 90GB   # before downloading
boyle serve   mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --budget 12GB
```

![boyle predict: a 224 GB model forecast from headers alone, nothing downloaded](https://raw.githubusercontent.com/beatakouchnir/boyle/main/docs/media/predict.gif)

*Named for Robert Boyle: PV = k. What you trade for pressure here is speed,
and the exchange rate is measured.*

> **Status: v0.1.** Working today: `predict`, `run`, `serve`, `bench`,
> `build`. Landing in v0.2: `trace` (routing capture that adds unmeasured
> families to `predict`'s curves and orders stores by co-activation).

## What a budget buys you — measured on real hardware

Two machines: an M5 Max (128 GB) and a 2021 M1 Pro MacBook Pro (32 GB) —
the second bought nothing but a `git clone`, a forecast, and a `bench`
that landed 1.4% from it.

| model | on disk | budget | decode | how verified |
|---|---|---|---|---|
| Qwen3-30B-A3B-4bit | 17 GB | 12 GB | ~18 tok/s | real OpenCode session; warm agent turn **3.3 s** (cold 28.7 s) |
| Qwen3-30B-A3B-4bit, **2021 M1 Pro 32 GB** | 17 GB | 12 GB | **14.5 tok/s** | `bench` vs a forecast made before the machine was ever measured: predicted 14.7 — off by 1.4%. At 20 GB: 17.5 vs 19.3 predicted, in band |
| Qwen3-235B-A22B-4bit | 132 GB | 70 GB | **11.7 tok/s** | `bench`, within the pre-run forecast band (12.5 ± 25%) |
| Qwen3-235B-A22B-4bit | 132 GB | 90 GB | ~15.5 tok/s | research-record anchor |
| Qwen3.5-397B-A17B-4bit | 224 GB | 90 GB | **7.2 tok/s** | live agent tool-exchange behind `serve`; **load 1.5 s**; forecast band 7.6–11.9 |

Decode is bit-identical to the resident model at any budget (asserted
token-by-token in the test suite); over-capacity prefill is
rounding-equivalent (same math, different batching — text has matched
resident output on every model measured). Accuracy is therefore a property
of the *model*, not the budget: an exact-offload 397B at 90 GB scored 0.96
on gsm8k (n=100) because that is what the model scores.

The load time is real: boyle wraps expert layers *before* weights
materialize, so a 224 GB checkpoint is serving requests ~2 seconds after
you hit enter — the first request then pays the expert fill (~35 s on the
397B; forecast up front by `predict`).

## `predict` — know before you download

```
$ boyle predict mlx-community/Qwen3.5-397B-A17B-4bit --budget 90GB --max-context 16384
boyle predict — mlx-community/Qwen3.5-397B-A17B-4bit
  budget 90.00 GB: FITS (fraction 0.36, slots 77.29 GB, resident 6.43 GB)
  decode ~9.5 tok/s (band 7.6–11.9) — expert hit rate ~83% [qwen3_5_moe curve, measured]
  first request after load: up to ~37 s (cold expert fill; load itself is seconds)
  context: 16384 guaranteed at this budget (headroom to ~18566)
  disk: 223.86 GB checkpoint
  accuracy [measured]: gsm8k (answer mode) = 0.96 (n=100)
```

Reads only the checkpoint *headers* (a few hundred KB over ranged HTTP —
never the weights), resolves your budget against exact tensor shapes,
applies a routing curve distilled from measured traces, and calibrates to
your disk with a one-time cold-read probe. `boyle bench` then measures the
truth on your machine and tells you whether it landed in the band —
the 235B row above is exactly that loop, closed at a fraction nobody had
measured before.

Forecasts are honest about their provenance: measured family curve vs
flat-routing prior, compute anchor vs I/O-only upper bound — the output
says which you're getting. **Accuracy is never forecast**; the accuracy
line is lookup into measured rows, or silence.

## The rest of the CLI

`run` — one-shot or scripted generation, with the honest footer:

```
$ boyle run mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --budget 12GB \
    -p "In one sentence: what does a hash table do?" --max-tokens 40
[boyle] fraction=0.387 slots=6.24 GB max_context=8192
A hash table stores key-value pairs and uses a hash function to quickly map
keys to indices in an array, enabling fast data retrieval. [...]
[boyle] 40 tokens in 1.8s (21.8 tok/s) — expert cache hit rate 78.9%
```

`bench` — the trust loop, measured on this machine vs the forecast
(output below is the real run from a 2021 M1 Pro 32 GB):

```
$ boyle bench mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --budget 12GB
[bench] predicted 14.7 tok/s (band 11.8–18.4); loading...
[bench] measured 14.5 tok/s steady (TTFT 2.8s, hit rate 87.3%) — WITHIN the predicted band 11.8–18.4
```

`build` — a colocated expert store: one contiguous read per cache miss
instead of nine scattered ones (+13% on the measured serving ceiling;
outputs verified token-identical to direct checkpoint reads):

```
$ boyle build mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
[build] layer 48/48: 16.3 GB written
[build] colo store: 16.31 GB -> ~/.cache/boyle/stores/mlx-community--Qwen3-30B-A3B-Instruct-2507-4bit
$ boyle serve mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --budget 12GB --colo ~/.cache/boyle/stores/...
```

Stacked checkpoints only (Qwen, gemma lineage); per-expert-scheme
checkpoints (OLMoE) are read directly by the runtime and need no store.

## Works with your tools

`boyle serve` exposes **two API surfaces from one model**: OpenAI-compatible
(`/v1`, SSE streaming, tool calls) and Ollama-compatible (`/api/*`, NDJSON,
real timing fields so UIs show true tok/s). It binds port 11434 when free,
so Ollama-first apps discover it with zero config; if a real Ollama is
running it politely falls back and prints the URL.

![the official ollama CLI talking to boyle with zero config](https://raw.githubusercontent.com/beatakouchnir/boyle/main/docs/media/ollama.gif)

| harness | connect via | config |
|---|---|---|
| **Ollama CLI & Python library** | native | zero config — `ollama list/ps/show/run` and `ollama.chat(...)` (incl. tools) verified against boyle |
| **OpenCode** | OpenAI-compatible | provider block below |
| **Cline / Continue** (VS Code) | OpenAI-compatible | base URL `http://127.0.0.1:11434/v1`, any API key |
| **Open WebUI** | Ollama connector | zero config when boyle holds port 11434 |
| **SillyTavern** | Custom OpenAI | API URL `http://127.0.0.1:11434/v1` |
| aider, Zed, Goose, LibreChat, LangChain, … | OpenAI-compatible | same base URL |

OpenCode (`~/.config/opencode/opencode.json`):

```json
{
  "provider": {
    "boyle": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "boyle (local)",
      "options": { "baseURL": "http://127.0.0.1:11434/v1" },
      "models": {
        "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit": {
          "name": "Qwen3-30B via boyle",
          "limit": { "context": 32768, "output": 4096 }
        }
      }
    }
  }
}
```

Tool calls are parsed for the Qwen family — both dialects (Qwen3 hermes
JSON and Qwen3.5/Coder XML blocks); other families stream text through
untouched, and the matrix below says which is which. Conversations are
prefix-cached, aligned against the chat template's own history rendering:
an agent's warm turns re-prefill only the new suffix.

## Support matrix

| tier | models | tool calls | prefix cache |
|---|---|---|---|
| measured | Qwen3-30B/235B (4/8-bit), Qwen3.5-397B (4-bit), gemma-4-26B MoE, OLMoE | Qwen: parsed | full |
| measured, hybrid-cache | Qwen3.5 family | parsed | warm within a user turn; each *new* user turn re-prefills once (~35 s at 397B) — hybrid attention caches cannot rewind |
| expected-works | other Qwen3-MoE-family variants | parsed | full |
| experimental | GLM-4.x/5.x MoE | passthrough | blocked on upstream mlx-lm support |
| out of scope (v1) | dense models, CUDA/Linux, multi-user batching | | |

**Untested models are announced, not undefined**: `serve` probes every model
at startup (template roundtrip, cache rewindability) and tells you which
prefix-cache class you're getting; `predict` labels measured curves vs
priors. The exact tested list, what "tested" means, and the one-command
qualification procedure for new releases live in
[COMPATIBILITY.md](https://github.com/beatakouchnir/boyle/blob/main/COMPATIBILITY.md) — new notable MoE releases get
qualified promptly, and a release needing code (new tool dialect, new
cache type) gets a tracking issue.

## Honest limits

- **Single-stream by design.** Diverse-prompt batching is drive-bound
  (~9.5 tok/s aggregate regardless of batch size — measured); concurrency
  would move latency around, not create throughput. Requests queue FIFO.
- **The speed floor is architectural**: per-layer expert residency requires
  a sync per MoE layer per token (~50 ms/token at 397B scale). Polling,
  event tricks, and speculative decoding were measured and lost — the
  research record has the receipts.
- **Small-expert models** (records under ~2 MB, e.g. OLMoE) are per-read
  latency-bound; forecasts there are upper bounds, and `predict` says so.
- Capture-quality quantization matters: 4-bit is the measured sweet spot;
  the cliff to 3-bit is severe on some tasks (see the accuracy notes
  `predict` prints).

## Why it works — the 30-second version

Expert routing is *flat*: across three model families there is no hot set —
LFU loses to LRU everywhere, and a clairvoyant cache beats LRU by 0.07 hit
rate. That kills clever prefetching, but it makes speed a function of two
numbers only: budget fraction (via one reusable hit curve) and bytes per
miss. That is why a forecast from checkpoint headers plus a 10-second disk
probe lands within a ±25% band, and why the levers that survived
measurement are exactly three: direct I/O with parallel installs, a
colocated expert store, and expert-major prefill. The full research record
— every lever tried, every dead end, every number — is in
[docs/report.md](https://github.com/beatakouchnir/boyle/blob/main/docs/report.md).

## Lineage

The runtime descends from the expert-offload patch developed for
[omlx](https://github.com/jundot/omlx) (PR #2595, Apache-2.0 — see NOTICE),
by way of a measurement program whose adopted levers this package ships.
Related upstream work: mlx PR #4249 (GPU-visible mmap weights),
mlx issue #2878.

## License

Apache-2.0. Portions derive from omlx — see NOTICE.
