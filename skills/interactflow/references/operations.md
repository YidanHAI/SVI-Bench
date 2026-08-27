# InteractFlow operations

## Contents

1. Preflight
2. Prepare external inputs
3. Run and monitor
4. Resume and recovery
5. Judge existing recordings
6. Completion audit
7. Prompt or rubric changes

## 1. Preflight

Work from the resolved repository root.

Confirm tools without exposing secrets:

```bash
python3 --version
node --version
ffmpeg -version
ffprobe -version
test -x scripts/run_all.sh
test -x scripts/record_all.sh
test -x scripts/judge_all.sh
```

Require Python 3.10+, Node.js 18+, Chromium for Playwright, and the packages in
`requirements.txt` and `package-lock.json`. Install them using the repository's
README when missing.

Verify `.env` permissions are `600`. Check only the presence of required
variables; never print their values. The five-model workflow normally needs:

- `JOYVL_WEB_URL`, `JOYVL_WEB_USERNAME`, `JOYVL_WEB_PASSWORD`
- `JOYAI_API_BASE`, `JOYAI_API_KEY`
- `DOUBAO_API_BASE`, `ARK_API_KEY`
- `MAGE_API_BASE`, `MAGE_REALTIME_API_BASE`, `MAGE_API_KEY`
- `MOSS_API_BASE`, `MOSS_REALTIME_API_BASE`, `MOSS_API_KEY`
- `MODELBEST_API_BASE`, `MODELBEST_REALTIME_API_BASE`, `MODELBEST_API_KEY`
- `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `JUDGE_MODEL=GPT-5.5`

Treat `config/recording_campaign.json` as the single source of truth. Confirm
that it specifies upload input, the direct network route, foreground scheduling
with `max_in_flight=1`, `busy_policy=skip`, and `queue_capacity=0`, and exactly
the five expected enabled models.

Run offline regression tests after code or prompt changes:

```bash
npm test
```

## 2. Prepare external inputs

Require:

```text
data/SVIBench-开源表.xlsx
data/interaction-75题/
```

If every video is named `<task-id>.mp4`, build the task manifest directly:

```bash
python scripts/build_tasks_from_xlsx.py \
  --xlsx data/SVIBench-开源表.xlsx \
  --sheet 题目池 \
  --video-dir data/interaction-75题 \
  --out data/recording_tasks_75.jsonl \
  --report data/recording_tasks_75.report.json
```

If videos have descriptive names, require `data/media_index.jsonl` and add:

```text
--video-map data/media_index.jsonl
```

Do not continue unless the report says 75 selected rows, 75 written tasks, and
zero missing videos. Review duplicate matches and duration mismatches rather
than choosing a file heuristically.

Prepare deterministic MiniCPM query audio:

```bash
python scripts/prepare_minicpmo_query_audio.py \
  --tasks data/recording_tasks_75.jsonl \
  --out-dir data/minicpmo_query_audio
```

Require a non-empty `data/minicpmo_query_audio/manifest.json` whose source task
hash matches the current task manifest.

## 3. Run and monitor

Validate before starting:

```bash
bash scripts/run_all.sh validate
```

Validation must not be described as a complete Judge validation when no
completed recording set exists. Independently resolve any workbook-contract
error before spending time on recording.

Start the complete workflow only on explicit request:

```bash
bash scripts/run_all.sh start
```

Inspect progress with:

```bash
bash scripts/run_all.sh status
bash scripts/record_all.sh status
bash scripts/judge_all.sh status
```

Use the state files and per-model logs named by status output. During active
monitoring, compare progress across all models and stages. A live PID without
increasing validated counts is not evidence of progress. Check recent log
timestamps, task/stage counters, retry ledgers, and the recorded error.

## 4. Resume and recovery

The supervisors resume validated artifacts automatically. Prefer rerunning the
same supported `start` command over deleting state or forcing individual tasks.

For recording recovery, use the wrapper's supported resume options:

```bash
bash scripts/record_all.sh start --resume-current
```

Do not change the WebUI, prompt, model, manifest, or output-root identity during
resume. Use `--allow-webui-migration` only when the user explicitly authorizes a
WebUI migration and the recording launcher accepts it.

Classify failures before intervening:

- transient API/network failure: allow bounded retry or supported resume;
- missing/invalid input: stop and repair the input contract;
- hash/provenance mismatch: use a new output root;
- exhausted task or launcher budget: report the exact model, task, stage, and
  last error before requesting a scope-changing decision.

Never delete partial outputs to make a status look clean.

## 5. Judge existing recordings

Set exactly one source:

```bash
export JUDGE_RECORDING_RUN=outputs/recording_campaign/runs/RUN_ID
```

or:

```bash
export JUDGE_RECORDING_INDEX=/absolute/path/to/recording_manifest.jsonl
```

Then run:

```bash
bash scripts/judge_all.sh prepare
bash scripts/judge_all.sh validate
bash scripts/judge_all.sh start
bash scripts/judge_all.sh status
```

Manifest preparation must bind exactly 375 recording rows, five expected model
IDs, 75 unique tasks per model, the workbook hash, evidence hashes, and timing
summaries. A formal manifest must declare `contains_human_labels: false`.

## 6. Completion audit

Do not rely only on `campaign_state.json`. Cross-check:

- each recording's capture summary and MP4 integrity;
- 75 final predictions per model and 375 stage predictions per model;
- stage identities, prompt hashes, request blindness, evidence hashes, and
  retry ledgers;
- each `judge_results_summary.json` for 75 successes and zero failures;
- `leaderboard.json` for all five model IDs and the official scoring metric.

If any check fails, report the campaign as incomplete even when a supervisor
state says `complete`.

## 7. Prompt or rubric changes

Treat checked-in formal prompts as immutable for an existing run. For an
authorized new prompt version:

1. create a new named/versioned prompt and output root;
2. keep task IDs, expert grades, aggregate targets, and historical predictions
   out of all model inputs;
3. run a label-blind pilot alignment evaluation separately;
4. record prompt and script hashes;
5. promote the candidate only after the declared MAE criterion is met;
6. run all 75 tasks only after the pilot gate passes.

Never tune against formal-model scores or insert task-specific answers.
