# boyle

**Run the model you want at the memory pressure you specify.**

Budgeted mixture-of-experts inference for Apple silicon: declare a memory
budget, and boyle runs the model inside it — including models far larger
than RAM — with outputs bit-identical to the resident model on the decode
path, a speed forecast before you download anything, and an
OpenAI-compatible server for local coding harnesses.

*Named for Robert Boyle: PV = k. What you trade for pressure here is speed,
and the exchange rate is measured.*

> **Status: pre-release.** The runtime is a port of a measured research
> program (capacity law across three model families, validated speed
> simulator, bit-identity contract); `predict`, `run`, `bench`, and `serve`
> work today. Not yet on PyPI.

## Works with your tools

`boyle serve` exposes **two API surfaces from one model**: OpenAI-compatible
(`/v1`, SSE streaming, tool calls) and Ollama-compatible (`/api/*`, NDJSON,
real timing fields so UIs show true tok/s). It binds port 11434 when free,
so Ollama-first apps discover it with zero config; if a real Ollama is
running it politely falls back and prints the URL.

```bash
boyle serve mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --budget 20GB
```

| harness | connect via | config |
|---|---|---|
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
      "models": { "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit": {} }
    }
  }
}
```

Tool calls are parsed for the Qwen family (hermes format); other families
stream text through untouched — the support matrix says which is which.
Conversations are prefix-cached: an agent's warm turns re-prefill only the
new suffix, not the whole history.

## License

Apache-2.0. Portions derive from [omlx](https://github.com/jundot/omlx) —
see NOTICE.
