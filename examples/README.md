# Example trials

One archived trial per domain, laid out exactly as `--specification-root`
expects, so this directory is directly usable as a replay source:

```
examples/<domain>/<variant>/vlm/
    fm_diagnostics/fm_call_001.json   the raw foundation-model response
    structural_sanitization.json      after dangling-reference repair
    functional_specification.json     compiled: wording to fixed vocabulary
    functional_requirement_graph.json G_F
    graph_grounding_result.json       phi : G_F to G_O
    symbolic_problem.json             what A* was given
    run_manifest.json                 frozen search contract, phase timing
    result.json                       terminal status and validation
```

```bash
./run_demo.sh workshop W2
```

**Re-running may not reproduce the archived `result.json`, and that is
expected.** `raw-replay` recompiles from the model's raw words, so it applies
today's canonicalization rather than the canonicalization in force when the
trial was first run. The region alias table was later corrected against the
scene geometry, which changes how region proposals resolve and therefore which
regions get inspected. The archived `result.json` is the original observation;
a fresh replay is today's pipeline reading the same model output.

This is the reason to use `raw-replay` rather than `replay` for any change to
canonicalization: `replay` loads the already-compiled specification and would
carry the old mapping forward unchanged, which silently reproduces the baseline
and reads as a clean null result.

The frozen tables are scored from the original run artifacts, not from replays.
