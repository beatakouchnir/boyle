# boyle

[![PyPI](https://img.shields.io/pypi/v/boyle)](https://pypi.org/project/boyle/)
[![license](https://img.shields.io/pypi/l/boyle)](https://github.com/beatakouchnir/boyle/blob/main/LICENSE)

> **Archived, October 2026.** boyle is no longer maintained. Its expert-offload runtime now lives in [oMLX](https://github.com/jundot/omlx); to run a mixture-of-experts model larger than memory on Apple silicon, use oMLX's [MoE expert offload](https://github.com/jundot/omlx/blob/main/docs/MoE_Expert_Offload.md). This repository remains as the measurement record: what each memory budget bought on real hardware, how well the speed forecasts held, and the research report behind both.

boyle ran mixture-of-experts models inside a declared memory budget on Apple silicon, keeping a fraction of each layer's experts resident and streaming the rest from the checkpoint, with decode outputs bit-identical to the fully resident model. It also forecast decode speed for a budget from checkpoint headers alone, before any weights were downloaded. The name is from Boyle's law (PV = k): memory is traded for speed at a measured rate.

## Where each part went

| boyle | upstream | status (October 2026) |
|---|---|---|
| expert offload runtime (resident slots, bit-identical decode) | oMLX [#2595](https://github.com/jundot/omlx/pull/2595) | merged |
| positional expert reads (DeepSeek V4.1) | oMLX [#3628](https://github.com/jundot/omlx/pull/3628) | merged |
| expert-major over-capacity prefill | oMLX [#3654](https://github.com/jundot/omlx/pull/3654) | merged |
| GLM DSA MoE offload | oMLX [#3696](https://github.com/jundot/omlx/pull/3696) | merged |
| budget → residency fit (`predict`'s FITS line) | oMLX [#3676](https://github.com/jundot/omlx/pull/3676) | open |
| live expert-cache hit rate | oMLX [#3910](https://github.com/jundot/omlx/pull/3910) | open |
| decode speed forecast (`predict`, `bench`, `trace`, routing curves) | oMLX [Discussion #4288](https://github.com/jundot/omlx/discussions/4288) | proposal |
| `serve` (OpenAI- and Ollama-compatible server) | oMLX's own server | not ported; oMLX already covers the OpenAI-compatible side |
| colocated expert store (`build`) | none | not upstreamed |

## The last release

The final release, 0.2.0, still installs and runs; no further fixes will follow.

```bash
uv tool install boyle==0.2.0
boyle predict mlx-community/Qwen3.5-397B-A17B-4bit --budget 90GB   # headers only, nothing downloaded
boyle bench   mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --budget 12GB
```

| command | what it does |
|---|---|
| `predict` | fit, decode-speed band, cold-fill time and context limit for a budget, from checkpoint headers plus a one-time disk probe |
| `bench` | measures decode on this machine and reports whether it landed inside `predict`'s band |
| `trace` | records expert routing during decode and distills it into a hit-rate curve for `predict` |
| `run` | one-shot generation under a budget |
| `serve` | OpenAI-compatible (`/v1`) and Ollama-compatible (`/api/*`) server under a budget |
| `build` | colocated expert store: one contiguous read per cache miss instead of nine (+13% on the measured serving ceiling) |

Supported models and how each was qualified are listed in [COMPATIBILITY.md](https://github.com/beatakouchnir/boyle/blob/main/COMPATIBILITY.md). Qwen3.8-Flash-Next needs mlx-lm with `qwen4_exp` (PR #1788).

## What a budget bought, measured

Two machines: an M5 Max (128 GB) and a 2021 M1 Pro (32 GB).

| model | on disk | budget | decode | how verified |
|---|---|---|---|---|
| Qwen3-30B-A3B-4bit | 17 GB | 12 GB | ~18 tok/s | OpenCode session; warm agent turn 3.3 s (cold 28.7 s) |
| Qwen3-30B-A3B-4bit, M1 Pro 32 GB | 17 GB | 12 GB | 14.5 tok/s | `bench`; forecast 14.7 (band 11.8–18.4). At 20 GB: 17.5 vs 19.3 forecast, in band |
| Qwen3-235B-A22B-4bit | 132 GB | 70 GB | 11.7 tok/s | `bench`; forecast 12.5 ± 25% |
| Qwen3-235B-A22B-4bit | 132 GB | 90 GB | ~15.5 tok/s | research-record anchor |
| Qwen3.5-397B-A17B-4bit | 224 GB | 90 GB | 7.2 tok/s | agent tool exchange through `serve`; load 1.5 s; below the forecast band of 7.6–11.9 |
| Qwen3.8-Flash-Next-4bit | 112 GB | 88 GB | 17.1 tok/s | `bench`; load 0.8 s, peak 79 GB; against mlx-vlm on the same weights, median KL 0.005 nats |

Decode is bit-identical to the resident model at every budget (asserted token by token in the test suite); over-capacity prefill is rounding-equivalent (the same math, batched differently). Accuracy is therefore a property of the model, not the budget: the 397B at 90 GB scored 0.96 on gsm8k (n=100). Loads take seconds because expert layers are wrapped before weights materialize; the first request then pays the cold expert fill (~35 s on the 397B).

### Qwen3.8-Flash-Next across budgets

Flash-Next spends 32 GB of its 111.5 GB checkpoint on a hashed n-gram embedding table that is read 16 rows per token. boyle streamed that table by row, so the table, not expert I/O, sets the decode floor, and shrinking the expert budget cost nothing until the experts became very tight (`bench`, M5 Max 128 GB):

| budget | experts resident | decode |
|---|---|---|
| 88 GB | 100% | 16.4 tok/s |
| 64 GB | 70% | 18.1 tok/s |
| 48 GB | 49% | 18.1 tok/s |
| 40 GB | 38% | 17.8 tok/s |
| 32 GB | 27% | 16.5 tok/s |
| 24 GB | 17% | 11.8 tok/s |

With a routing curve captured by `boyle trace` and one bench anchor, every budget above fell inside `predict`'s band, 24 GB included.

## How well the forecasts held

`predict` output for the 397B row above:

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

Out of sample, the forecast landed at 12.5 vs 11.7 tok/s measured (235B, 70 GB) and 14.7 vs 14.5 (30B on an M1 Pro never measured before; 19.3 vs 17.5 at 20 GB, in band). The known miss is the 397B: 7.2 tok/s through the HTTP server against a band of 7.6–11.9, because the model does not include serving overhead. Trace replay predicted live hit rates within 1–3 points on four configurations, and routing curves were identical at 4-bit and 8-bit. `predict` never forecast accuracy; its accuracy line is a lookup into measured rows.

## Limits that were measured

- **Single stream.** Diverse-prompt batching is drive-bound at ~9.5 tok/s aggregate regardless of batch size.
- **A sync per MoE layer per token** (~50 ms/token at 397B scale), because the router's output decides which weights must be present. Polling, event tricks and speculative decoding were measured and did not beat it.
- **Small-expert models** (records under ~2 MB, e.g. OLMoE) are bound by per-read latency; forecasts there are upper bounds.
- **4-bit was the best measured trade-off**; the drop to 3-bit is severe on some tasks.

## Why speed was forecastable

Expert routing is flat: across three model families there is no hot set, and LFU loses to LRU at every budget. Speed is then a function of two numbers, the resident fraction (through one hit-rate curve per family) and bytes per miss, which is why a forecast from checkpoint headers plus a short disk probe landed within a ±25% band. The full record, including every lever tried and every dead end, is in [docs/report.md](https://github.com/beatakouchnir/boyle/blob/main/docs/report.md).

## Lineage

The runtime descends from the expert-offload patch developed for [oMLX](https://github.com/jundot/omlx) (PR #2595, Apache-2.0; see NOTICE), by way of a measurement program whose adopted levers this package shipped. Related upstream work: mlx PR #4249 (GPU-visible mmap weights, closed), mlx issue #2878.

## License

Apache-2.0. Portions derive from oMLX; see NOTICE.
