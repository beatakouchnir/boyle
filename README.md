# boyle

**Run the model you want at the memory pressure you specify.**

Budgeted mixture-of-experts inference for Apple silicon: declare a memory
budget, and boyle runs the model inside it — including models far larger
than RAM — with outputs bit-identical to the resident model on the decode
path, a speed forecast before you download anything, and an
OpenAI-compatible server for local coding harnesses.

*Named for Robert Boyle: PV = k. What you trade for pressure here is speed,
and the exchange rate is measured.*

> **Status: pre-release scaffold.** The runtime is a port of a measured
> research program (capacity law across three model families, validated
> speed simulator, bit-identity contract); the packaging, CLI, and server
> are landing now. Not yet on PyPI.

## License

Apache-2.0. Portions derive from [omlx](https://github.com/jundot/omlx) —
see NOTICE.
