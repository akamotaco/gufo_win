# Gemma 4 for Gufo

A Gufo source extension for Google's Gemma 4 on Linux x86-64 / Strix Halo
`gfx1151`:

- Gemma 4 31B (Unsloth `UD-Q4_K_XL`, `UD-Q8_K_XL`)
- Gemma 4 31B QAT (Q4_0)
- Gemma 4 26B-A4B mixture of experts (`UD-Q4_K_XL`, `UD-Q6_K_XL`, `UD-Q8_K_XL`)

with `gemma4-assistant` MTP drafting, prompt lookup, the BF16 vision sidecar,
batched sessions, native tool calls and the shared serving cache.

**Status:** in progress. The engine is moving here from the
[franzmoca/gufo](https://github.com/franzmoca/gufo) `feat/gemma` branch
([gufo-org/gufo#426](https://github.com/gufo-org/gufo/pull/426)). This revision
registers provider metadata only; use `feat/gemma` to run the models until the
port is complete.

Maintainer: Francesco Moca ([@franzmoca](https://github.com/franzmoca)).

## Build with Gufo

```sh
cmake --preset release -DGUFO_EXTENSION_DIRS=/path/to/gemma4-gufo-extension
cmake --build --preset release --parallel 4
./build/release/gufo diagnose --section extensions
```

With Nix, `nix build` in this repository builds the pinned Gufo revision in
`flake.nix` with this extension.

## License

The extension source is MIT licensed ([LICENSE](LICENSE)). Gemma 4 weights are
downloaded separately under their own license.
