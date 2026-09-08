# Operational handoff, 2026-09-07 night

Written before launching the Kitchen grid so that a crash costs minutes, not
hours. Read this first, then `SESSION_CONTEXT_20260907.md` for the reasoning
behind each fix.

## 1. Infrastructure

**Inference: the Blackwell host only.** gvlab2 (RTX 5090) was handed to Naren
and is no longer serving.

- Host `user1@10.4.25.63`, RTX PRO 5000 Blackwell, 48 GB, 128 GB RAM.
- vLLM **pinned to `0.27.2rc1.dev122+g8efa13b70`**, identical to what gvlab2
  ran, so every scene shares one server version. Pinned with:
  `uv pip install "vllm==0.27.2rc1.dev122+g8efa13b70" --extra-index-url
  https://wheels.vllm.ai/8efa13b700f1836657699cae2503dc2feab27fa0/ --torch-backend=auto`
- Served `--max-model-len 65536 --max-num-seqs 16`, KV pool 701,385 tokens,
  `10.70x` concurrency at full window. 65536 is required by ViLaIn Kitchen,
  whose prompt runs 24-31k with a 16384-token output budget on top.
- **The tunnel is opened by the researcher**, interactively, because this
  machine's key is not on the box:
  `ssh -L 18000:127.0.0.1:8000 user1@10.4.25.63` -- leave that window open.
  Port 18000 is not negotiable: ViLaIn's transport hard-asserts it.

Health check: `curl -s http://127.0.0.1:18000/v1/models`

## 2. Guardrails (learned from two OOM cascades tonight)

- **`loginctl enable-linger longhorizon` is set.** Without it, logging out
  stops `user@1002.service` and kills every `--user` unit. That is what ended
  the 21:10 session.
- **All legs run in `lh.slice`** (`~/.config/systemd/user/lh.slice`) with a
  *single* ceiling for the group: `MemoryHigh=16G`, `MemoryMax=20G` on a 31 GB
  host. Per-unit caps were the earlier mistake -- 11G+11G+12G summed to 34G,
  and one 12-worker leg had no cap at all, so the OOM killer took dbus and
  gnome-shell and logged the user out.
- **Never launch a grid with `nohup` from an editor terminal.** That puts it in
  the VS Code snap cgroup, which an OOM kills wholesale. Use:
  `systemd-run --user --unit=lh-<name> --slice=lh.slice --same-dir
  --property=StandardOutput=append:<log> --property=StandardError=append:<log>
  <script>`
- **Never run two `--resume` batch runners against one `--output-root`.** The
  second relocates the first's in-flight episodes as `.interrupted_NNN`.

## 3. Where things are

- Status: `scratchpad/status.sh` (see below for the path).
- Scratchpad, scripts and logs:
  `/tmp/claude-1002/-home-longhorizon-Documents-LH-Extension-V1/d803e54c-28e0-42f4-9520-69502b05ef8d/scratchpad`
- The *previous* session's transcript and scratchpad survived its crash at
  `~/.claude/projects/-home-longhorizon-Documents-LH-Extension/dced8008-*.jsonl`
  and `/tmp/claude-1002/-home-longhorizon-Documents-LH-Extension/dced8008-*/scratchpad`.
  Transcripts are keyed by cwd, so a session run from a parent directory does
  not appear in the editor's list for a child directory. Nothing is lost.

## 4. Completed and comparable

| scene | vlm+owl | retrieval |
|---|---|---|
| Workshop | **200/200** | **10/10** |
| Living Room | **200/200** | **10/10** |
| Kitchen | 0/240 (launching) | **12/12** |

Final numbers, from `baseline_common.summarize_execution_batch`:

| scene | method | n | outcome correct | feasible success | infeasible rejection |
|---|---|---|---|---|---|
| Living Room | VLM-TAMP | 100 | 60.0% | 52/60 | 8/40 |
| Living Room | OWL-TAMP | 100 | 73.0% | 43/60 | 17/40 |
| Living Room | Retrieval | 10 | 100.0% | 0/6 | 4/4 |
| Workshop | VLM-TAMP | 99 | 12.1% | 8/79 | 4/20 |
| Workshop | OWL-TAMP | 100 | 13.0% | 0/80 | 13/20 |
| Workshop | Retrieval | 10 | 80.0% | 1/8 | 0/2 |
| Kitchen | Retrieval | 12 | 50.0% | 0/6 | 6/6 |

The consistent finding across two scenes: **OWL-TAMP rejects infeasible
variants far better than VLM-TAMP while solving fewer feasible ones**
(17/40 vs 8/40 on Living Room, 13/20 vs 4/20 on Workshop). Its single-shot
"inspect everything" sketch is what `infeasibility_proven()` needs. Do not
summarise OWL as uniformly worse.

## 5. Not yet run

| leg | episodes | note |
|---|---|---|
| Kitchen vlm+owl | 240 | launching now |
| ROBUST-TAMP x3 scenes | 320 | wired and live-verified tonight |
| ViLaIn x3 scenes | 320 | Kitchen's 120 need the 65536 window |

ViLaIn has **0** artifacts: its earlier partial run was archived as
`vilain_20260907.pre_clean_tree_2016` because those episodes recorded a dirty
tree, and the surviving sample was selected by a provenance crash rather than
by the method. **Its 0% result is NOT established** -- the episodes that
crashed were the ones that got furthest.

**ViLaIn's provenance guard rejects both an uncommitted tracked change and a
HEAD change mid-run.** Stop its legs before any edit or commit, or in-flight
episodes die at the execution boundary.

## 6. Fixes tonight, all committed

Six harness faults, five of one class -- infrastructure failure billed to a
method's planning budget:

1. **Kitchen published the oracle's region names** (`B1/C1/C2/D1/D2`) to the
   model; the leakage guard rejected all 240 episodes. Now anonymised as
   `region_0001..0005`. Infeasible variants also aborted before their first
   call and now run, since they are scored on rejection.
2. **ViLaIn truncation ended the episode** with no retry and no artifact.
3. **ViLaIn's transport deadline was 300 s** against 600 s for the methods it
   is tabled against, while making the slowest calls in the grid.
4. **ROBUST-TAMP charged transport faults to its planning budget** --
   `ModelTransportError` subclasses `PlanningError`.
5. **ViLaIn bypassed the prompt-leakage guard entirely**, the only
   model-driven method with none. Its Kitchen episodes have not run.
6. **Retrieval Kitchen launched an interactive MuJoCo viewer** per episode
   (`--headless` parsed and dropped), throttling stepping to render rate, and
   read a frame directory Kitchen never writes. That cell had *no* results and
   a 240 h estimate; it now completes in ~3 minutes.

Also: `make_paper_tables.py` labelled ROBUST-TAMP as "Ours (single FM call)"
and bolded it. It is now a baseline row; `functional_tamp` is reserved and
bolded for the proposed method.

## 7. Deliberately not changed

- **OWL-TAMP does not retry truncation.** It does label it
  `MODEL_OUTPUT_TRUNCATED`, so it is excluded from planning-failure counts,
  which is what `BASELINE_FIDELITY.md` requires. Measured impact: 2 of 200
  episodes. Changing it now would make Kitchen OWL differ from the completed
  Workshop and Living Room OWL episodes.
- **`--max-replans` for ROBUST-TAMP is 5**, following the grid's planning
  budget rather than OWL's default of 8. The published setting is 10, which
  becomes the ablation.

## 8. Test baseline on this machine

**57 failed, ~1700 passed, 12 errors.** `CLAUDE.md`'s "expect exactly five
failures" is stale -- it predates the Naren merge. The 12 errors are all one
missing fixture, `runs/integrated_no_pot_clearance_seed19_20260807`, part of
the 5 GB `runs/` archive never copied here. Run as:
`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest`

---

# 2026-09-08 morning: Kitchen complete, region leaks found in two methods

## Kitchen finished 06:22 — the baseline table is complete (771 episodes)

| scene | method | n | outcome | feas succ | infeas rej | calls | reqs | acts | mean s |
|---|---|---|---|---|---|---|---|---|---|
| Living Room | VLM-TAMP | 100 | 60.0% | 52/60 | 8/40 | 3.1 | 5.9 | 8.7 | 598 |
| Living Room | OWL-TAMP | 100 | 73.0% | 43/60 | 17/40 | 1.0 | 8.9 | 6.5 | 643 |
| Living Room | Retrieval | 10 | 100.0% | 0/6 | 4/4 | 0 | 0 | 6.0 | 100 |
| Workshop | VLM-TAMP | 100 | 12.0% | 8/80 | 4/20 | 4.8 | 9.6 | 3.1 | 974 |
| Workshop | OWL-TAMP | 100 | 13.0% | 0/80 | 13/20 | 1.0 | 3.5 | 2.2 | 213 |
| Workshop | Retrieval | 10 | 80.0% | 1/8 | 0/2 | 0 | 0 | 5.1 | 131 |
| Kitchen | VLM-TAMP | 119 | 13.4% | 0/60 | 16/59 | 5.0 | 9.6 | 4.6 | 1291 |
| Kitchen | OWL-TAMP | 120 | 0.0% | 0/60 | 0/60 | 1.0 | 12.1 | 0.0 | 1088 |
| Kitchen | Retrieval | 12 | 50.0% | 0/6 | 6/6 | 0 | 0 | 5.0 | 55 |

Feasible success is monotone in scene difficulty: 87/72% -> 10/0% -> 0/0%.
**Neither model-driven baseline solves any feasible Kitchen variant.**

**OWL-TAMP's Kitchen 0.0% needs its caveat stated.** `acts = 0.0` across all
120 episodes: it terminated at `NO_SYMBOLIC_PLAN` having executed nothing.
`infeasibility_proven()` requires every storage region to have been physically
inspected, so an episode that executes nothing cannot earn a rejection whatever
it concluded. The 0/60 is structural, not evidence it cannot discriminate --
and it inverts its Living Room (17/40) and Workshop (13/20) behaviour.

Kitchen is 239/240; `vlm_tamp K10 seed 1` is re-running (lost to the residual
D1 leak below).

## Region-identity leaks in ROBUST-TAMP and ViLaIn (both legs stopped)

`FORBIDDEN_CANONICAL_REGION_TOKENS` in
`functional_tamp_pipeline/audit.py` lists only Kitchen (`D1 D2 C2 B1 C1`) and
Workshop (`LEFT_DRAWER RIGHT_DRAWER TOOL_CABINET`) tokens. **It has no Living
Room tokens**, so a Living Room leak is silent while the same defect on
Workshop fails loudly. That asymmetry hid two real defects:

**ROBUST-TAMP Living Room.** Its observation's `regions` list is anonymised,
but `known_regions[].category` carries the canonical role name:
`{"id": "region_0001", "category": "personal_table_left", ...}`. VLM-TAMP's
Living Room observation exposes only `{id, inspected, state}` -- no category at
all. Since the task instruction is "each of two fixed *personal side tables*
... the fixed *shared coffee table*", the category hands ROBUST-TAMP the
object-to-region answer the other methods must infer. Source is the
`category` field of `ObservedRegion`/`ObservedEntity` in
`mujoco_scenes/tamp/state.py`. Workshop was clean (its categories are the
legitimate lowercase annotation labels, not canonical tokens).

**ViLaIn-TAMP, both scenes.** Two independent sources:
- Kitchen/Workshop: `baselines/vilain_tamp/observations.py:24`
  `FIXED_INSPECTION_ORDERS` hardcodes `("D1","D2","C2","B1","C1")` and
  `("LEFT_DRAWER","RIGHT_DRAWER","TOOL_CABINET")`. Caught: 194 errors,
  every Workshop episode died.
- Living Room: `identity.py:302` and `live_fixed_evidence.py:30` key off
  `personal_table_left` / `personal_table_right` / `shared_table`. **Silent.**

ViLaIn is the harder fix: these identifiers run end-to-end through its PDDL
`:init`/`:goal`, inspection traces, identity map and fixed-evidence tables, so
anonymising only the prompt would desynchronise the PDDL from the prompt.

**Extend the forbidden list with the Living Room tokens** so this class cannot
hide again.

## Fixes committed this morning

- `2ff16ca6` the leakage guard now quotes a bounded excerpt around each
  forbidden token. It identified the ViLaIn Workshop source on the first
  occurrence, after the previous D1 refusal cost an hour of blind reading.
- `7943d374` ROBUST-TAMP Living Room now runs headless. The shared `common`
  block adds `--headless` for Kitchen only, on the grounds that "the Living
  Room physical runtime is constructed headless" -- true of vlm/owl's runner
  but not of `run_living_room_discovery_replanning.py`, which owns the flag and
  defaults to viewer-on. Measured while wrong: 1 `mujoco.viewer` process, 37
  GPU/DRI handles, **1.4-1.6% CPU** per episode against 22-25% headless. Third
  appearance of this bug; now asserted as a parity property (the grid passes
  `--headless` exactly when the runner defines it).
- `3cf652bd`, `b2f2047c` two end-of-episode faults in
  `vlm_tamp_baseline/run_kitchen.py`: `expected` bound only under
  `--planning-only`, then `arguments.variant` used where only
  `--physical-variant` is set. Each killed episodes *after* 20+ minutes of
  work. No test calls `main()` or exercises the executing path for any scene
  runner -- that gap is why both shipped, and an end-to-end executing-path
  test with a mocked planner and runtime is still worth adding.

## State

Running: `lh-k10` (one Kitchen episode), `lh-rt-workshop` (clean, headless).
Stopped: both ViLaIn legs, ROBUST-TAMP Living Room -- all for region leaks.
Discarded: ~100 viewer-throttled ROBUST-TAMP Living Room episodes, and both
ViLaIn output roots (0 artifacts).

gvlab2 is free again if inference becomes the bottleneck. It is not currently:
the Blackwell server runs `--max-num-seqs 16` and sits at 7 concurrent with 0
queued, so worker count rather than server capacity is the limit.
