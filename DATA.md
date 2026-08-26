# External data

Data and generated artifacts are intentionally excluded from Git. The default
formal configuration expects this local layout:

```text
data/
├── JoyAI-VL-Interaction评测.xlsx
├── interaction-75题/
│   └── *.mp4
├── recording_tasks_75.jsonl
├── recording_tasks_75.report.json
└── minicpmo_query_audio/
    ├── manifest.json
    └── audio/
        └── *.f32le
```

The workbook's `V1_题目池` sheet defines the selected tasks and the
`评测维度` sheet defines the Judge rubric. `interaction-75题/` contains the
corresponding source MP4 files. Build the two derived inputs with:

```bash
python scripts/build_tasks_from_xlsx.py \
  --xlsx data/JoyAI-VL-Interaction评测.xlsx \
  --video-dir data/interaction-75题 \
  --out data/recording_tasks_75.jsonl \
  --report data/recording_tasks_75.report.json

python scripts/prepare_minicpmo_query_audio.py \
  --tasks data/recording_tasks_75.jsonl \
  --out-dir data/minicpmo_query_audio
```

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
