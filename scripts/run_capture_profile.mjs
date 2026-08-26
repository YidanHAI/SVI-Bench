#!/usr/bin/env node
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { recordingWebUrl } from './recording_config.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');
const RECORDING_WEB_URL = recordingWebUrl();

function timestamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return [
    d.getFullYear(),
    pad(d.getMonth() + 1),
    pad(d.getDate()),
    '_',
    pad(d.getHours()),
    pad(d.getMinutes()),
    pad(d.getSeconds()),
  ].join('');
}

function quoteArg(value) {
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(value)) return value;
  return `'${value.replaceAll("'", "'\\''")}'`;
}

function printHelp() {
  console.log(`Usage:
  npm run capture:rtsp -- [capture args]
  npm run capture:upload -- [capture args]
  npm run capture:profile -- --mode rtsp|upload [capture args]

Profiles:
  rtsp    configured WebUI + per-task ffmpeg RTMP publish / RTSP playback relay
  upload  configured WebUI + /api/video/upload + /api/video/start

Common examples:
  npm run capture:rtsp -- --tasks tasks.local_71.jsonl --out outputs/rtsp_run_001 --skip-existing
  npm run capture:upload -- --tasks tasks.local_71.jsonl --out outputs/upload_run_001 --skip-existing
  npm run capture:rtsp -- --limit 1 --print

Any extra argument is passed to scripts/capture.mjs. Explicit arguments override profile defaults when the
same option name is provided.`);
}

function parseWrapperArgs(argv) {
  let mode = 'rtsp';
  let printOnly = false;
  const captureArgs = [];

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    }
    if (arg === '--mode') {
      mode = argv[i + 1] || '';
      i += 1;
      continue;
    }
    if (arg.startsWith('--mode=')) {
      mode = arg.slice('--mode='.length);
      continue;
    }
    if (arg === '--print' || arg === '--dry-run') {
      printOnly = true;
      continue;
    }
    captureArgs.push(arg);
  }

  if (!['rtsp', 'upload'].includes(mode)) {
    throw new Error(`--mode must be "rtsp" or "upload", got ${mode || '(empty)'}`);
  }

  return { mode, printOnly, captureArgs };
}

function hasOption(args, option) {
  return args.some((arg) => arg === option || arg.startsWith(`${option}=`));
}

function addDefault(args, option, value = null) {
  if (hasOption(args, option)) return;
  args.push(option);
  if (value !== null) args.push(String(value));
}

function redactArgsForLog(args) {
  const result = [...args];
  const secretOptions = new Set(['--web-password', '--vlm-api-key']);
  for (let index = 0; index < result.length; index += 1) {
    if (secretOptions.has(result[index]) && index + 1 < result.length) {
      result[index + 1] = '[REDACTED]';
    } else if (result[index].startsWith('--web-password=')) {
      result[index] = '--web-password=[REDACTED]';
    } else if (result[index].startsWith('--vlm-api-key=')) {
      result[index] = '--vlm-api-key=[REDACTED]';
    }
  }
  return result;
}

function buildArgs(mode, captureArgs) {
  const args = [];

  addDefault(args, '--tasks', 'tasks.local_71.jsonl');
  addDefault(args, '--out', `outputs/${mode}_capture_${timestamp()}`);
  addDefault(args, '--process-interval-s', 1);
  addDefault(args, '--frames-per-batch', 1);

  if (mode === 'upload') {
    addDefault(args, '--web-url', RECORDING_WEB_URL);
    addDefault(args, '--web-username', process.env.JOYVL_WEB_USERNAME || '');
    addDefault(args, '--local-video-mode', 'upload');
    addDefault(args, '--task-gap-s', 8);
    addDefault(args, '--task-retries', 5);
    addDefault(args, '--retry-gap-s', 20);
    addDefault(args, '--health-retries', 20);
    addDefault(args, '--health-interval-s', 5);
    addDefault(args, '--stream-ready-timeout-s', 180);
    addDefault(args, '--video-upload-timeout-s', 900);
    addDefault(args, '--post-video-response-timeout-s', 180);
    addDefault(args, '--stream-start-retries', 1);
    addDefault(args, '--stream-start-retry-gap-s', 5);
  } else {
    addDefault(args, '--web-url', RECORDING_WEB_URL);
    addDefault(args, '--web-username', process.env.JOYVL_WEB_USERNAME || '');
    addDefault(args, '--local-video-mode', 'rtsp');
    addDefault(args, '--task-gap-s', 10);
    addDefault(args, '--task-retries', 5);
    addDefault(args, '--retry-gap-s', 20);
    addDefault(args, '--health-retries', 20);
    addDefault(args, '--health-interval-s', 5);
    addDefault(args, '--stream-ready-timeout-s', 90);
    addDefault(args, '--stream-start-retries', 4);
    addDefault(args, '--stream-start-retry-gap-s', 5);
    addDefault(args, '--stream-reconnects', 0);
    addDefault(args, '--stream-reconnect-timeout-s', 60);
    addDefault(args, '--attempt-timeout-margin-s', 7200);
    if (!hasOption(captureArgs, '--fresh-browser-per-attempt')) {
      args.push('--fresh-browser-per-attempt');
    }
    addDefault(args, '--local-rtsp-relay-host', '127.0.0.1');
    addDefault(args, '--local-rtsp-relay-port', 8554);
    addDefault(args, '--local-relay-publish-protocol', 'rtmp');
    addDefault(args, '--local-rtmp-relay-port', 1935);
    addDefault(args, '--local-rtsp-playback-host', '127.0.0.1');
    addDefault(args, '--local-rtsp-playback-port', 8554);
    addDefault(args, '--local-rtsp-relay-prefix', 'vl_local');
    addDefault(args, '--local-rtsp-warmup-ms', 3000);
    addDefault(args, '--local-rtsp-preroll-s', 60);
    addDefault(args, '--local-rtsp-postroll-s', 30);
    addDefault(args, '--local-rtsp-connect-lead-s', 45);
    addDefault(args, '--local-rtsp-reconnect-warmup-ms', 3000);
    addDefault(args, '--local-rtsp-fps', 12);
    addDefault(args, '--local-rtsp-gop-s', 2);
    addDefault(args, '--local-rtsp-scale-long-edge', 480);
    addDefault(args, '--local-rtsp-crf', 32);
    addDefault(args, '--local-rtsp-maxrate', '400k');
    addDefault(args, '--local-rtsp-bufsize', '800k');
    if (!hasOption(captureArgs, '--local-rtsp-encode')) {
      args.push('--local-rtsp-encode');
    }
  }

  return [...args, ...captureArgs];
}

async function main() {
  const { mode, printOnly, captureArgs } = parseWrapperArgs(process.argv.slice(2));
  const capturePath = path.join(ROOT, 'scripts', 'capture.mjs');
  const finalArgs = [capturePath, ...buildArgs(mode, captureArgs)];
  const command = [process.execPath, ...redactArgsForLog(finalArgs)].map(quoteArg).join(' ');

  console.log(`[capture-profile] mode=${mode}`);
  console.log(`[capture-profile] ${command}`);

  if (printOnly) return;

  const child = spawn(process.execPath, finalArgs, {
    cwd: ROOT,
    stdio: 'inherit',
  });

  child.on('exit', (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 1);
  });
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
