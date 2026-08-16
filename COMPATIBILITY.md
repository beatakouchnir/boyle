# Compatibility

boyle's guarantees are **measured, not assumed**. This file says exactly
which models have been qualified, what "qualified" means, and what to
expect from a model nobody has tested yet.

## Explicitly tested models

The machine-readable version of this table ships in the package
(`boyle/data/tested_models.json`) — `boyle serve` tells you at startup if
your exact model is not on it.

| model | verified | what was checked |
|---|---|---|
| mlx-community/OLMoE-1B-7B-0125-Instruct-4bit | 2026-08-14 | bit-identity vs resident (token-exact); full server suite, both APIs; prefix cache: full |
| mlx-community/gemma-4-26b-a4b-it-4bit | 2026-08-14 | expert-major prefill text-identity vs resident |
| mlx-community/Qwen3-4B-Instruct-2507-4bit | 2026-08-14 | dense passthrough (0 MoE layers); tool-call exchange (hermes dialect) |
| mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit | 2026-08-15 | real OpenCode session (write/bash/edit); warm agent turn 3.3 s; overflow refusal; **cross-machine on a 2021 M1 Pro 32 GB**: qualify clean (identity exact, tools parsed, prefix 94%), bench 14.5 vs 14.7 forecast |
| mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit | 2026-08-14 | `bench` vs `predict` at an out-of-sample fraction (11.7 vs 12.5, in band) |
| mlx-community/Qwen3.5-397B-A17B-4bit | 2026-08-14 | serve agent tool exchange (XML dialect); decode in forecast band (7.2); load 1.5 s; hybrid-cache class |

## Untested models: what happens

An untested model is not undefined behavior — it is *announced* behavior.
At startup, `boyle serve` probes (tokenizer-only, before any traffic):

- **Template roundtrip** — whether the chat template re-renders assistant
  history the same way it renders generation prompts. The Qwen3.5 think-block
  quirk was exactly this class; the probe catches it for any new template.
- **Cache rewindability** — hybrid-attention caches cannot trim, which
  limits the prefix cache to within-user-turn reuse. Detected up front.
- **Family curve** — `predict` labels its forecast "measured curve" vs
  "flat-routing prior" and widens the band accordingly.

What the probes *cannot* verify automatically: new tool-call dialects
(an unrecognized format streams through as text — visible, not silent),
new checkpoint layouts (unsupported layers are skipped with a log line and
run resident), and behavioral quality under offload. Those are what
qualification is for.

## Qualifying a new model (the maintenance procedure)

When a new family or generation lands, one command runs the standard
battery and prints a row for this file:

```bash
BOYLE_LOCAL_MODELS=1 uv run python tests/qualify.py <hf-repo-id> --budget <B>
```

The battery, in order:

1. **Anatomy + plan** — headers parse, MoE layers recognized, budget resolves.
2. **Template roundtrip + cache class** — the startup probes, reported.
3. **Bit-identity** — short-prompt token equality vs the resident model
   (skipped with a note when the model exceeds RAM; the contract is then
   carried by the family's smaller members).
4. **Tool-dialect probe** — a forced tool call; flags *unparsed tool
   syntax* (dialect gap — file an issue with the raw sample) vs *parsed*
   vs *model declined*.
5. **Prefix-cache warm check** — second-turn cached-token ratio.
6. **`bench` vs `predict`** — a live out-of-sample point; outside-band is
   a bug report by construction.

Maintenance promise: new notable MoE releases get qualified promptly and
this file updated; a release that needs code (new tool dialect, new cache
type, new layout) gets an issue tracking it. If you qualify a model on
your machine, a PR adding the row (with the qualify output) is welcome.
