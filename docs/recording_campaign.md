# Formal recording campaign

The supported recording entrypoint is `bash scripts/record_all.sh`. Its single
source of truth is `config/recording_campaign.json`; use an ignored local copy
selected by `VL_INTERACTION_CAMPAIGN_CONFIG` for machine-specific changes.

## Protocol invariants

- Inputs are local MP4 files uploaded through the configured HTTPS WebUI.
- WebUI credentials come from `JOYVL_WEB_USERNAME` and
  `JOYVL_WEB_PASSWORD`.
- Every system/item pair uses a fresh stateful session.
- Each session permits one in-flight frame inference. Frames arriving while
  busy are skipped and no ordinary-frame backlog is accumulated.
- Each timed Query is delivered exactly once and is bound to the displayed
  frame at dispatch. Query-frame work remains FIFO.
- Recording starts only after endpoint identity, session readiness, and target
  upload checks pass.
- A healthy non-response is a capability outcome. Network, identity, capture,
  and protocol failures are technical failures handled by bounded retries.
- Accepted recordings must pass task identity, video integrity, Query
  delivery, backend identity, and protocol audits.

The public workflow does not require users to create or supply an RTSP stream.

## Configuration

Store credentials and endpoints only in a private environment file:

```bash
cp .env.example .env
chmod 600 .env
```

The launcher refuses an environment file readable by group or other users.
Do not pass credentials on the command line. External inputs and their default
layout are documented in [../DATA.md](../DATA.md).

## Lifecycle

```bash
bash scripts/record_all.sh dry-run
bash scripts/record_all.sh start
bash scripts/record_all.sh status
bash scripts/record_all.sh stop
```

The optional `--models id,...`, `--task-ids id,...`, and `--run-id NAME`
arguments are intended for bounded operational runs. Resume an interrupted
formal campaign with `--resume-current`; completed recordings are reused only
when their task, source media, protocol, and validation records still match.

After completion, verify the current run with:

```bash
node scripts/verify_recording_campaign.mjs --run-id RUN_ID
```

Add `--full-decode` to decode every output video. Verification is read-only.
