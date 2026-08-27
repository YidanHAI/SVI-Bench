# Supported entrypoints

## Complete workflow

`run_all.sh` is the public end-to-end interface:

```bash
bash scripts/run_all.sh validate
bash scripts/run_all.sh start
bash scripts/run_all.sh status
bash scripts/run_all.sh stop
```

It runs the recording campaign to completion and then starts the five-model,
five-stage Judge campaign. Both phases preserve validated artifacts and resume
after process interruption.

## Recording only

`record_all.sh` is the recording campaign interface:

```bash
bash scripts/record_all.sh dry-run
bash scripts/record_all.sh start
bash scripts/record_all.sh status
bash scripts/record_all.sh stop
```

The launcher reads `config/recording_campaign.json`, schedules all enabled
models, applies bounded retries, and records durable state. Supporting modules
implement the browser capture, model adapters, video validation, task wall
budgets, and atomic artifact promotion.

## Judge only

Set exactly one recording source, then use `judge_all.sh`:

```bash
export JUDGE_RECORDING_RUN=outputs/recording_campaign/runs/RUN_ID
# Alternatively: export JUDGE_RECORDING_INDEX=/path/to/recording_manifest.jsonl

bash scripts/judge_all.sh prepare
bash scripts/judge_all.sh validate
bash scripts/judge_all.sh start
bash scripts/judge_all.sh status
bash scripts/judge_all.sh stop
```

`prepare_judge_manifests.py` binds the workbook rubric and validated recording
artifacts into five label-free manifests. `judge_pipeline.py` executes first
pass, review 1, review 2, adjudication, and final review. The audit modules
verify request blindness, evidence identity, stage provenance, retry ledgers,
and final deliverability.

The public workbook uses G/F/P anchor labels. During manifest construction F
and P are normalized to the already validated Judge labels S and B, so the
language model still emits only G/S/B grades. `judge_scoring.py`
deterministically derives the raw dimension mean, the official item-level
D1/D2-coupled score, and the diagnostic score without D3.
