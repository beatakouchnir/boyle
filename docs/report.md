# The measured map: lossless MoE streaming on Apple silicon

*The research record behind boyle. One machine (Apple M5 Max, 128 GB unified
memory), August 2026. Every claim below comes from a pre-registered
experiment with committed raw results; the underlying lab record is being
prepared for separate publication. boyle ships exactly the levers that
survived.*

**The question:** how fast can mixture-of-experts models larger than memory
run on a consumer Apple-silicon machine, with outputs bit-identical to a
resident model?

**The answer, measured to closure:** a 397B-class model decodes at ~9 tok/s
and a 235B model at ~15.5 in a 90 GB budget; under the GPU's wired ceiling,
streaming is free (resident speed at every bit-rate); above it, one law
governs everything — and every remaining millisecond sits at a measured
floor.

## 1 · The two regimes: the wired wall divides the world

Metal wires the *entire* buffer on first kernel use — a 3 MB one-expert
gather faults its full 1.2 GB tensor — and the wired ceiling counts mapped
buffers, so a 217 GB model demand-mapped through stock kernels dies at
~112 GB touched. Transparent bigger-than-RAM GPU streaming therefore cannot
exist on this platform. Two architectures survive, one per regime:

| | under the wall (≤ ~110 GB) | over the wall |
|---|---|---|
| mechanism | zero-copy mapped views | expert slot-cache, fetch-on-miss (boyle) |
| speed | = resident, every bit-rate (4-bit 132 tok/s; 8-bit 100.2%; bf16 101.2% of resident twins) | the capacity law (§2) |
| memory | wires in full while generating | bounded: ~84–90 GB for 132–418 GB models |
| outputs | bit-identical | bit-identical (decode / within-capacity prefill) |

## 2 · The capacity law (three families, one curve)

Expert routing is **flat** in every family measured — gemma-lineage
(128 experts), Qwen (512), DeepSeek-lineage GLM (256, sigmoid + shared
expert): LFU ≪ LRU at every budget; a clairvoyant (Belady-optimal) cache
beats LRU by only **+0.07 hit rate**; per-domain expert unions cover 42–46%
of all (layer, expert) pairs. **No hot set exists.**

Consequently, over-wall speed is a function of two numbers only: the
budget-to-expert-mass ratio (via one reusable hit curve) and bytes-per-miss.
A trace-driven simulator built on that curve predicted live hit rates within
1–3 points on four model/quant configurations — and the same curves, shipped
inside boyle, predicted 12.5 tok/s for a 235B configuration nobody had
measured; the live bench read 11.7. The curves are quant-independent:
distilling the 4-bit and 8-bit routing traces of the same model produces
identical curves. And the forecast transfers across machines: on a 2021
M1 Pro (32 GB — different GPU class, different disk, probed locally), the
same machinery predicted 14.7 tok/s for a 30B model in a 12 GB budget;
the live bench measured 14.5.

Measured single-stream decode (bit-identical outputs, 128 GB machine):

| model | budget fraction | decode |
|---|---|---|
| Qwen3-235B-4bit | 0.64 | 15.5 tok/s |
| Qwen3-235B-4bit | 0.45 | 11.7 tok/s (out-of-sample forecast check) |
| Qwen3.5-397B-4bit | 0.36 | 8.8 tok/s (7.2 behind the full server) |
| Qwen3-235B-8bit | 0.34 | 3.16 tok/s |
| GLM-class 418 GB-4bit | 0.19 | 1.72 tok/s |

## 3 · The quality–speed ladder

Streaming converts the quality–memory trade into quality–speed. Under the
wall, bits are speed-free; over it, each doubling of bits costs roughly the
curve twice (halved fraction × doubled miss bytes ≈ 4.9× per doubling):

| bit-rate | 26B-class under-wall | 235B-class over-wall |
|---|---|---|
| 4-bit | 132 tok/s (= resident) | 15.5 tok/s |
| 8-bit | 92 tok/s (100.2% of resident) | 3.16 tok/s |
| bf16 | 56.7 tok/s (101.2% of resident) | — (out of range) |

Quantization, not offload, is the axis that costs accuracy — and
task-dependently: under the same 4-bit quant, factual QA loses ~1.5 points
where structured generation loses 18–21. Exact offload changes *when*
weights are read, never *which* expert runs; an offloaded model's benchmark
score is the model's score.

## 4 · The levers: adopted, capped, and dead — all with mechanisms

### Adopted (these are boyle)

| lever | effect | mechanism |
|---|---|---|
| direct I/O miss fetches (F_NOCACHE pread, parallel installs) | 4.0× decode | the memmap path paid GIL-held copies + page-cache double-buffering |
| colocated expert records, co-activation-ordered | +13% serving ceiling | 9 scattered reads per miss → 1 contiguous ~7 MB read |
| pooled expert-major prefill | −47% fill-heavy time-to-first-token | prefill finally exercises NVMe parallelism (12.6 GB/s parallel-random) |
| wired-limit rule | 88× (pathology removal) | custom decode loops must pin the Metal wired limit or residency churns per step |

### Measured ceilings

| dimension | ceiling | bound by |
|---|---|---|
| identical-prompt batching | 6.2× aggregate | compute — the sharing upper bound |
| diverse-prompt batching | ~9.5 tok/s aggregate at any batch size | drive bandwidth; no cross-user expert sharing exists |
| per-layer sync | ~50 ms/token at 397B scale | architectural: router output gates which weights must exist |
| fetch | ~12.6 GB/s | the NVMe itself (parallel-random beats sequential) |

### Dead, with cause of death

| idea | verdict |
|---|---|
| learned / temporal prefetch | structurally starved — the LRU slots already are the recency predictor (confirmed on three families) |
| eviction-policy cleverness | optimal caching caps the win at +0.07 hit rate; LFU loses everywhere |
| speculative decoding | +11% best and decays — consecutive tokens share only ~27% of experts; flat routing denies sharing in time exactly as it does across users |
| sentinel-polled sync | 14% *regression* — the sync cost was never readback overhead |
| userspace second-level cache behind the slots | 3 hits in 19,594 — a victim cache behind an LRU sees only the reuse-poor rejected tail |
| weight compression at 4-bit | quantized weights are near max-entropy |
| per-layer capacity allocation / hybrid splits | flat routing + concave hit curve ⇒ uniform allocation dominates (Jensen) |

## 5 · Thinking-mode economics under streaming

Per-token throughput is mode-blind — thinking ran 37% *faster* per token
(cache warmth) — but costs ~6× the tokens: **3.4–4.4× slower per answer,
≥4.3× worse time-per-correct** even crediting every censored item (gsm8k,
n=100/mode). Corollary adopted throughout: short benchmark cells understate
steady-state speed; long generation runs are the standard design.

## 6 · Method — what kept this honest

- **Pre-registration:** every experiment declared predictions and decision
  rules before data. Several predictions were beaten, several confirmed,
  and five hypotheses were killed by their own written criteria.
- **Refuse-to-fake asserts:** wrap counts, mechanism counters, sentinel
  checks — two A/B rounds were voided by silently-inactive mechanisms and
  caught only by counting invocations.
- **Bit-identity as the correctness bar** wherever the contract promises
  it, at three quantization tiers.
- **Physical verification over accounting:** per-process memory ledgers are
  blind to mapped pages; the only honest eviction test is a timed re-read.
- **A measured noise floor** (~1.5% step time), so nulls are nulls.

## 7 · Provenance and upstream work

The runtime descends from the expert-offload patch developed for
[omlx PR #2595](https://github.com/jundot/omlx/pull/2595) (Apache-2.0 — see
NOTICE), by way of the measurement program above. Related public artifacts:
[mlx PR #4249](https://github.com/ml-explore/mlx/pull/4249) (GPU-visible
mmap weights; closed on scope policy, implementation public) and the
measurements posted to
[mlx issue #2878](https://github.com/ml-explore/mlx/issues/2878).
