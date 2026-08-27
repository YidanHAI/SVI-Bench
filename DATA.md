# External data

Data and generated artifacts are intentionally excluded from Git. The default
formal configuration expects this local layout:

```text
data/
├── SVIBench-开源表.xlsx
├── interaction-75题/
│   └── <task-id>.mp4
├── media_index.jsonl          # optional when media are not named by task id
├── recording_tasks_75.jsonl
├── recording_tasks_75.report.json
└── minicpmo_query_audio/
    ├── manifest.json
    └── audio/
        └── *.f32le
```

The workbook's `题目池` sheet defines exactly 75 formal tasks and the
`评测维度` sheet defines the Judge rubric. `interaction-75题/` contains the
corresponding source MP4 files. Name each file with its task id when possible,
for example `A1001.mp4`. If media retain descriptive filenames, provide an
optional `media_index.jsonl` with one mapping per line:

```json
{"id":"A1001","video":"interaction-75题/加油站起火.mp4"}
```

Build the two derived inputs with:

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

Add `--video-map data/media_index.jsonl` to the first command when using the
optional mapping. The builder treats every non-empty id row as a formal task;
the public workbook intentionally has no internal selection-status column.

The task manifest may contain machine-local absolute paths. This is expected:
it is generated on the machine that performs recording and remains ignored by
Git. The formal recording path is local MP4 upload through the configured
WebUI; users do not provide RTSP sources.

## Generated layout

The default workflow writes:

```text
outputs/
├── recording_campaign/
│   ├── current.json
│   └── runs/RUN_ID/
└── judge_campaign/
    └── runs/RUN_ID/
        ├── manifests/
        ├── models/
        ├── campaign_state.json
        └── leaderboard.json
```

Recording and Judge manifests contain paths and cryptographic hashes that bind
each result to its input files. Do not edit generated manifests or completed
artifacts in place. Create a new output root for a new benchmark run.

Human labels are not required to run the released benchmark Judge and must not
be placed in a formal Judge manifest or prompt.
