# SVI-Bench

SVI-Bench is the runnable recording and evaluation pipeline for streaming
vision-language interaction. It records a time-aligned interaction trace for
each system and task, then grades every recording with a label-blind,
five-stage automatic Judge.

This repository contains the current formal workflow. Benchmark media,
workbooks, model weights, credentials, generated results, and paper source are
distributed separately.

## What runs

The checked-in campaign evaluates 75 tasks on five systems in parallel:

- JoyAI-VL-Interaction
- Doubao Seed 2.1 Pro
- Mage-VL
- MOSS-VL-Realtime
- MiniCPM-O-4.5-9B

All systems receive the same local MP4 inputs through the configured WebUI.
Each session allows one in-flight frame inference; frames are skipped while the
model is busy rather than queued. The Judge then executes five stages for every
model/task pair: first pass, two independent reviews, adjudication, and final
review.

## Repository layout

| Path | Purpose |
| --- | --- |
| `config/` | Formal recording campaign and five-model registry |
| `scripts/run_all.sh` | End-to-end recording-then-Judge entrypoint |
| `scripts/record_all.sh` | Recording-only entrypoint |
| `scripts/judge_all.sh` | Judge-only entrypoint |
| `prompts/` | The five fixed Judge prompts |
| `tests/` | Offline protocol, scoring, resume, and audit tests |
| `moss_mage_api_service/` | Optional local MOSS/Mage serving adapter |
| `third_party/` | Pinned JoyAI adapter source and provenance |

## Install

Requirements are Python 3.10+, Node.js 18+, `ffmpeg`, `ffprobe`, Chromium,
and `cloudflared` when the WebUI must reach locally launched model adapters.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm ci
npx playwright install chromium
```

MOSS and Mage checkpoints require separate environments; see
[`moss_mage_api_service/README.zh-CN.md`](moss_mage_api_service/README.zh-CN.md).

## Configure

Copy the environment template and keep the populated file private:

```bash
cp .env.example .env
chmod 600 .env
```

Fill the WebUI URL and credentials, the five model endpoints/keys, and the
OpenAI-compatible Judge endpoint/key. No endpoint or credential is embedded in
the checked-in configuration.

Place the external workbook and MP4 files as described in [DATA.md](DATA.md),
then build the recording manifest and MiniCPM query audio:

```bash
python scripts/build_tasks_from_xlsx.py \
  --xlsx data/SVIBench-开源表.xlsx \
  --sheet 题目池 \
  --video-dir data/interaction-75题 \
  --out data/recording_tasks_75.jsonl \
  --report data/recording_tasks_75.report.json

python scripts/prepare_minicpmo_query_audio.py \
  --tasks data/recording_tasks_75.jsonl \
  --out-dir data/minicpmo_query_audio
```

If paths or runtime policy differ locally, copy
`config/recording_campaign.json` to the ignored
`config/recording_campaign.local.json`, edit that copy, and set
`VL_INTERACTION_CAMPAIGN_CONFIG` in `.env`.

## One-command run

Validation is local and does not start recording or call the Judge API:

```bash
bash scripts/run_all.sh validate
```

Start the complete workflow with one command:

```bash
bash scripts/run_all.sh start
```

The supervisor runs recording first, waits for all five 75-task recording sets
to pass their checks, builds label-free Judge manifests, and then starts all
five five-stage Judge jobs. Interrupted work is resumed from validated task and
stage artifacts.

```bash
bash scripts/run_all.sh status
bash scripts/run_all.sh stop
```

For a new independent run, set a new `PIPELINE_OUTPUT_ROOT` and
`JUDGE_OUTPUT_ROOT`, or archive the previous generated `outputs/` directory.

## Run stages separately

The two phases can also be operated independently:

```bash
bash scripts/record_all.sh dry-run
bash scripts/record_all.sh start
bash scripts/record_all.sh status
```

After recording completes, point `JUDGE_RECORDING_RUN` at that run directory:

```bash
export JUDGE_RECORDING_RUN=outputs/recording_campaign/runs/RUN_ID
bash scripts/judge_all.sh validate
bash scripts/judge_all.sh start
bash scripts/judge_all.sh status
```

## Codex Skill

The repository includes an agent-facing InteractFlow Skill at
`skills/interactflow/`. Install or link that directory into the skills location
used by your Codex environment, then invoke `$interactflow` when asking Codex
to validate, run, monitor, resume, diagnose, or audit the recording-plus-Judge
workflow. The Skill delegates execution to the supported scripts above and
does not contain credentials, datasets, model weights, or a second copy of the
pipeline.

For example, ask: `Use $interactflow to start the complete benchmark.` The
Skill first runs a secret-safe setup check. If anything is missing, it reports
the required variable names, files, or installation commands and starts
nothing. Once setup is complete, the same request prepares the manifest and
MiniCPM query audio, launches recording, hands the completed recordings to the
five-stage Judge, and monitors the workflow through its completion audit.

The deterministic front door used by the Skill is also available directly:

```bash
python3 skills/interactflow/scripts/interactflow.py check
python3 skills/interactflow/scripts/interactflow.py start
python3 skills/interactflow/scripts/interactflow.py status
```

## Scores and outputs

The public workbook expresses its anchors as G/F/P. The manifest builder maps
these labels deterministically to the validated Judge protocol's G/S/B labels
(F to S and P to B), which map to `1/0.5/0`. Within each task, applicable D1
and D2 are combined as one component using their minimum; a lone D1 or D2 is retained.
That component is averaged with the other applicable dimensions, and Overall
is the unweighted mean of all 75 task scores on a 0--100 scale. Raw
five-dimension means remain diagnostic only.

Generated artifacts are written below `outputs/` and ignored by Git. The Judge
campaign writes per-model `judge_results_summary.json` files and a combined
`leaderboard.json` after all five systems complete.

## Tests

These checks do not call external model APIs:

```bash
npm test
```

No repository-level license is currently declared. Add the approved license
before announcing the repository as open source.
