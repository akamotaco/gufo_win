# Tools

The scripts need Python 3 with NumPy. The oracle, `dspark_reference.py` and
`vision_reference.py` also need PyTorch, and `vision_reference.py` Pillow.
`gufo/gguf.py` is Gufo's GGUF reader, copied here so the scripts run outside a Gufo
checkout.

## Model files

| Script | Use |
| --- | --- |
| `ds41/engram_split.py` | Set the shard an Engram file declares, on a copy or in place, to share one table across builds |
| `ds41/direction_adapter.py` | Convert mako `.abl` files and rank-1 abliteration LoRAs to direction adapters for `--adapter`, or inspect one |
| `ds41/pack_expert_tails.py` | Pack every routed expert's tail into one file for a second drive |
| `ds41/import_expert_rank.py` | Convert Xolotl's published expert ranking to the built-in `runtime/expert_rank.inc` |
| `ds41/import_pruned_experts.py` | Convert an expert-pruned release's kept-experts list to a built-in table |

## Expert streaming

The engine writes an expert trace with `DS41_EXPERT_TRACE=PATH`. Gufo's
`tools/expert_stream/` reads it (the format is in its README) and holds the
model-neutral replays: `prefetch_sim.py` (prefetch policies), `evict_sim.py`
(eviction rules, LRU and the offline optimum) and `trace_compare.py` (two
traces record the same cache decisions). The scripts below import that reader:
set `GUFO_SRC` to a Gufo checkout.

| Script | Use |
| --- | --- |
| `ds41/prefetch_fit.py` | Fit the simulator's costs to measured A/B ratios |
| `ds41/replay_sim.py` | Read stalls of the decoder's bounded replay |
| `ds41/rank_eval.py` | Evaluate the built-in expert ranking |
| `ds41/route_occupancy.py` | Routed tile occupancy per layer |
| `ds41/expert_histogram.py` | Build an expert-cache seed from oracle routes |
| `ds41/stream_io_bench.py` | Read-only SSD benchmark over real expert spans |
| `ds41/engram_row_bench.py` | Cold random Engram row reads |

## References

| Script | Use |
| --- | --- |
| `ds41/reference.py` | CPU oracle over the GGUF weights (official `model.py` and `engram.py`) |
| `ds41/quants.py`, `ds41/gguf_model.py` | Dequantization and split-GGUF access for the oracle |
| `ds41/dspark_reference.py` | CPU oracle for one DSpark draft |
| `ds41/vision_reference.py` | CPU reference for the vision encoder |
| `ds41/chat_goldens.py` | Chat-template goldens from the official encoder (`tests/fixtures/encoding-1017.json`) |
| `ds41/compare_logits.py` | Perplexity, NLL difference, KL and argmax agreement between `ds41_load_probe` logits dumps |

`dspark_reference.py` and `vision_reference.py` compare against dumps from the GPU
probes; [QUALITY.md](../docs/models/deepseek-v4.1-flash/QUALITY.md) describes the checks.

Gufo's development shell has PyTorch for the GPU, built against an older ROCm than the
shell's own, whose variables make it crash at the first GPU operation. Clear them for the
oracle:

```sh
nix develop github:gufo-org/gufo -c bash -c '
  export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr : "\n" | grep -v rocm-runtime | paste -sd:)
  for v in $(env | grep -o "^\(HIP\|HSA\|ROCM\)[A-Z_]*"); do unset "$v"; done
  python3 tools/ds41/reference.py run --model MODEL-00001-of-N.gguf --tokens "$(cat ids)" --score --output DIR'
```

A 3K-token score takes about 6 minutes and an 18.5K one 12.

## Probes

`probe/` holds the programs that run a model outside the server. Build them from a Gufo
checkout configured as in [AGENTS.md](../AGENTS.md) (they are not part of `nix build`):

```sh
nix develop -c cmake --build build/ds41 --target deepseek41flash_probes
```

| Program | Use |
| --- | --- |
| `ds41_load_probe` | Memory plan; `--decode IDS [--batch] [--score] [--out DIR]` teacher-forced logits or scores; `--spec-check`, `--joint-check`, `--replay-check`, `--dspark-dump` exactness checks |
| `ds41_mmq_bench` | Batched kernels on synthetic or recorded routes (`--f16 1 --counts FILE`, `--dense`) |
| `ds41_gemv_bench` | Single-token GEMVs against their references |
| `ds41_vision_probe` | Vision encoder dumps for `vision_reference.py` |

A change that moves outputs is gated on the 3K teacher-forced logits of the build before
and after it, batched and token by token:

```sh
ds41_load_probe MODEL --stream-experts --decode "$(cat ids)" --batch --out new-batch
python3 tools/ds41/compare_logits.py ids base-batch new-batch
```

## Kernel benchmarks

`bench/` holds standalone `gfx1151` benchmarks of the decoder's kernels. Each one
includes the production source from `kernels/rocm/`. Build them into `/tmp` from a
Gufo development shell:

```sh
nix develop github:gufo-org/gufo -c tools/bench/build.sh ds41_expert_gemv_tune
```
