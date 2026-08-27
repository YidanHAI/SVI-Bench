---
name: interactflow
description: Operate the SVI-Bench InteractFlow automation for local-MP4 recording and label-blind five-stage judging. Use when Codex needs to configure, validate, prepare, start, monitor, resume, diagnose, stop, or audit the repository's five-model recording-plus-Judge workflow, or report its progress and deliverability.
---

# InteractFlow

Treat the repository scripts as the execution engine. Do not recreate per-model
commands or fork a second orchestration path.

## Locate the repository

Resolve the project root in this order:

1. Use the current workspace when it contains `scripts/run_all.sh`,
   `scripts/record_all.sh`, and `scripts/judge_all.sh`.
2. Use `INTERACTFLOW_ROOT` when it points to such a directory.
3. Ask for the clone path only if neither works.

Read a repository-level `AGENTS.md` before acting. Then inspect `README.md`,
`DATA.md`, `.env.example`, and `config/recording_campaign.json`. Never treat a
path stored inside this Skill as the user's project root.

## Honor the one-invocation contract

When the user asks to start the full workflow, use the bundled front door:

```bash
python3 skills/interactflow/scripts/interactflow.py start
```

The command performs a secret-safe preflight, prepares the 75-task manifest
and MiniCPM query audio, validates the formal configuration, and delegates to
`bash scripts/run_all.sh start`. It must either start the complete
recording-then-Judge supervisor or start nothing.

If preflight reports `needs_configuration`, tell the user only the missing
variable names, files, executables, or packages and the reported setup
commands. Never ask them to paste secret values into chat. After the user says
the setup is complete, rerun the same `start` command; do not make them invoke
the preparation stages manually.

After a successful start, continue monitoring with:

```bash
python3 skills/interactflow/scripts/interactflow.py status
```

Keep monitoring through recording and all five Judge stages until the
completion audit passes or a genuine blocker requires user action. The
background repository supervisor performs the recording-to-Judge handoff;
the Skill remains the intelligent operator that checks progress, diagnoses
stalls, and invokes supported resume behavior.

## Interpret the request

- For status, inspection, explanation, or diagnosis, perform read-only checks.
  Do not start, stop, resume, rerecord, or rejudge anything.
- For validation or setup, prepare local derived inputs and run checks without
  calling model APIs unless the user explicitly requests a live smoke test.
- For a read-only setup report, run
  `python3 skills/interactflow/scripts/interactflow.py check`.
- For a full run, prefer the bundled `interactflow.py start` front door. It
  calls `bash scripts/run_all.sh start` only after preflight and preparation
  pass.
- For recording-only or Judge-only work, use the corresponding supported
  wrapper. Do not assemble ad hoc per-model commands.
- Stop a live campaign only when the user explicitly requests it.
- When asked to monitor, continue checking until completion or a genuine
  blocker. Diagnose stalled work and use the supported resume behavior; do not
  merely launch and disengage.

Read [references/operations.md](references/operations.md) before preparing,
running, resuming, or diagnosing a campaign.

## Preserve formal invariants

- Use the campaign-configured WebUI over its configured route and local MP4
  upload mode. Do not substitute another endpoint or convert the source to
  RTSP.
- Keep one in-flight frame inference per session, skip frames while busy, and
  never enable a per-frame backlog or API concurrency.
- Keep the formal model set at five systems and the task set at 75 unless the
  user explicitly requests a non-formal experiment.
- Use GPT-5.5 and the checked-in five fixed prompts for the formal Judge. The
  stages are first pass, review 1, review 2, adjudication, and final review.
- Keep human pilot labels out of formal manifests, prompts, requests, and
  evidence. Human labels are only for separate alignment evaluation.
- Preserve item-level scoring: combine applicable D1 and D2 with their minimum,
  average that component with the other applicable dimensions, then average
  the 75 task scores without weighting.
- Never edit generated manifests or completed artifacts in place. Use a new
  output root for an independent run and resume only hash-compatible artifacts.

## Handle credentials and external data

Read credentials only from environment variables or the private `.env` file.
Never print, summarize, commit, or persist credential values. Report only which
variable names are missing.

Keep workbooks, source videos, model weights, generated media, logs, and result
artifacts outside Git. Use `data/SVIBench-开源表.xlsx` as the task and rubric
source. Do not use the human pilot workbook as a formal Judge input.

## Report completion

Require all of the following before calling a formal run complete:

- 375 validated recordings: five models times 75 tasks;
- 1,875 successful Judge stage predictions: 375 tasks times five stages;
- five complete per-model result summaries with zero failed tasks;
- a generated combined `leaderboard.json`;
- passing blindness, provenance, evidence-integrity, and resume checks.

Report partial counts by model and stage. Name the exact failed validation or
missing artifact when incomplete; never round progress up or infer success from
a launcher process alone.
