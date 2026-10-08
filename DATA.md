# SVI-Bench data

The benchmark data are hosted separately at
[Danmel02/SVI-bench](https://huggingface.co/datasets/Danmel02/SVI-bench).
Pin a dataset commit with `--revision` when reproducing reported results.

The dataset is governed by the [SVI-Bench Research-Only Data Terms](DATA_TERMS.md).
It is limited to non-commercial academic research and benchmark evaluation.
The source videos remain the property of their respective rights holders. The
software in this repository is licensed separately under Apache-2.0.

SVI-Bench has **75 tasks backed by 71 unique source videos**. Four videos are
intentionally reused by two tasks each because the same visual trajectory is
evaluated under different interaction requirements. The released
`media_index.jsonl` is the authoritative task-to-video mapping.

## Download

Download a pinned snapshot into the repository's ignored `data/` directory:

```bash
hf download Danmel02/SVI-bench \
  --repo-type dataset \
  --revision DATASET_COMMIT \
  --local-dir data
```

The snapshot layout is:

```text
data/
├── DATA_TERMS.md
├── SVIBench-开源表.xlsx
├── media_index.jsonl          # 75 task references -> 71 unique files
├── annotations/
│   └── SVI-Pilot-human-ratings.xlsx
└── interaction-75题/
    └── *.mp4                  # 71 source videos
```

The workbook's `题目池` sheet defines the 75 formal tasks, and `评测维度`
defines their dimension-specific grading anchors. Video containers in the
release package are remuxed without re-encoding to remove nonessential editor,
device, user, and creation metadata.

## Build recording inputs

Generate machine-local task paths from the released workbook and mapping:

```bash
python scripts/build_tasks_from_xlsx.py \
  --xlsx data/SVIBench-开源表.xlsx \
  --sheet 题目池 \
  --video-dir data/interaction-75题 \
  --video-map data/media_index.jsonl \
  --out data/recording_tasks_75.jsonl \
  --report data/recording_tasks_75.report.json
```

The report must contain 75 selected rows, 75 written tasks, 71 unique source
paths, and zero missing videos. The generated manifest may contain absolute
paths because it remains local and is ignored by Git.

MiniCPM's native video mode receives each textual Query as synthesized audio.
For exact MiniCPM reproduction, use the frozen audio cache associated with the
reported run. If the cache is regenerated with the command below, the complete
Query text is sent to the Microsoft Edge online TTS service:

```bash
python scripts/prepare_minicpmo_query_audio.py \
  --tasks data/recording_tasks_75.jsonl \
  --out-dir data/minicpmo_query_audio
```

Do not use online TTS for private task text without authorization. A local TTS
implementation may instead produce the same 16 kHz mono float32 files and a
hash-valid manifest.

## Generated evaluation artifacts

The default workflow writes recordings and Judge outputs below `outputs/`,
which remains ignored by Git. Manifests contain cryptographic hashes binding
each result to its inputs. Do not edit completed artifacts in place; use a new
output root for a new benchmark run.

Human pilot labels are calibration data, not formal Judge inputs, and must not
be placed in a formal Judge manifest or prompt.
