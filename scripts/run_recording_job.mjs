#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  DEFAULT_VLM_CONFIG,
  getVlmProfile,
  loadVlmRegistry,
  profileFingerprint,
} from './vlm_profiles.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function printHelp() {
  console.log(`Usage:
  node scripts/run_recording_job.mjs --profile ID --tasks FILE [options] [-- capture-options]

Options:
  --profile ID       Required VLM profile id
  --tasks FILE       Required JSONL/CSV task manifest
  --mode MODE        upload or rtsp; default upload
  --vlm-config FILE  Model registry; default config/vlm_models.json
  --out-root DIR     Stable matrix output root; default is content-addressed
  --allow-direct     Explicitly permit a non-formal direct API profile
  --dry-run          Print the resolved command only

The default output path includes hashes of the task manifest, model profile, mode, and forwarded
capture options. Running the same command resumes that exact job; changing any input creates a new
output path. Secrets must be supplied through environment variables, never capture options.`);
}

function parseArgs(argv) {
  const args = {
    profile: '',
    tasks: '',
    mode: 'upload',
    vlmConfig: DEFAULT_VLM_CONFIG,
    outRoot: '',
    allowDirect: false,
    dryRun: false,
    captureArgs: [],
  };
  let forwarding = false;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (forwarding) {
      args.captureArgs.push(arg);
      continue;
    }
    const next = () => {
      if (index + 1 >= argv.length) throw new Error(`Missing value for ${arg}`);
      index += 1;
      return argv[index];
    };
    if (arg === '--') forwarding = true;
    else if (arg === '--profile') args.profile = next();
    else if (arg.startsWith('--profile=')) args.profile = arg.slice('--profile='.length);
    else if (arg === '--tasks') args.tasks = next();
    else if (arg.startsWith('--tasks=')) args.tasks = arg.slice('--tasks='.length);
    else if (arg === '--mode') args.mode = next();
    else if (arg.startsWith('--mode=')) args.mode = arg.slice('--mode='.length);
    else if (arg === '--vlm-config') args.vlmConfig = next();
    else if (arg.startsWith('--vlm-config=')) args.vlmConfig = arg.slice('--vlm-config='.length);
    else if (arg === '--out-root') args.outRoot = next();
    else if (arg.startsWith('--out-root=')) args.outRoot = arg.slice('--out-root='.length);
    else if (arg === '--allow-direct') args.allowDirect = true;
    else if (arg === '--dry-run' || arg === '--print') args.dryRun = true;
    else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown job option ${arg}; put capture options after --`);
    }
  }
  if (!args.profile) throw new Error('Missing --profile');
  if (!args.tasks) throw new Error('Missing --tasks');
  if (!['upload', 'rtsp'].includes(args.mode)) throw new Error('--mode must be upload or rtsp');
  if (args.captureArgs.some((arg) => (
    arg === '--web-password'
    || arg.startsWith('--web-password=')
    || arg === '--vlm-api-key'
    || arg.startsWith('--vlm-api-key=')
  ))) {
    throw new Error('Supply WebUI and VLM secrets through environment variables');
  }
  return args;
}

function safeName(value) {
  return String(value).replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^_+|_+$/g, '') || 'job';
}

function stableOutputRoot(args, profile, tasksPath) {
  const taskBytes = fs.readFileSync(tasksPath);
  const identity = JSON.stringify({
    task_sha256: crypto.createHash('sha256').update(taskBytes).digest('hex'),
    profile_sha256: profileFingerprint(profile),
    mode: args.mode,
    capture_args: args.captureArgs,
  });
  const jobHash = crypto.createHash('sha256').update(identity).digest('hex').slice(0, 12);
  const taskName = safeName(path.basename(tasksPath, path.extname(tasksPath)));
  return path.join(ROOT, 'outputs', `${taskName}_${safeName(profile.id)}_${jobHash}`);
}

function runChild(command, commandArgs) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, commandArgs, { cwd: ROOT, stdio: 'inherit', env: process.env });
    const forward = (signal) => {
      if (!child.killed) child.kill(signal);
    };
    const onSigint = () => forward('SIGINT');
    const onSigterm = () => forward('SIGTERM');
    process.once('SIGINT', onSigint);
    process.once('SIGTERM', onSigterm);
    child.once('error', reject);
    child.once('exit', (code, signal) => {
      process.removeListener('SIGINT', onSigint);
      process.removeListener('SIGTERM', onSigterm);
      if (signal) reject(new Error(`Recording job terminated by ${signal}`));
      else resolve(code ?? 1);
    });
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const tasksPath = path.resolve(ROOT, args.tasks);
  if (!fs.existsSync(tasksPath)) throw new Error(`Task manifest does not exist: ${tasksPath}`);
  const registry = loadVlmRegistry(args.vlmConfig);
  const profile = getVlmProfile(registry, args.profile);
  if (profile.native_video) {
    throw new Error(
      `Profile ${profile.id} requires ${profile.input_transport} (${profile.native_video_schema}); `
      + 'the frame-stream recording job cannot be used for this model',
    );
  }
  const formalProfile = profile.formal_eval && profile.route === 'joyai_adapter';
  const explicitlyAllowedDirectProfile = args.allowDirect && profile.route === 'direct';
  if (!formalProfile && !explicitlyAllowedDirectProfile) {
    throw new Error(
      `Profile ${profile.id} is not a formal shared-adapter profile; `
      + 'pass --allow-direct to run a clearly labelled raw-API job',
    );
  }
  if (args.mode === 'upload' && !process.env.JOYVL_WEB_PASSWORD && !args.dryRun) {
    throw new Error('JOYVL_WEB_PASSWORD is required by the upload WebUI');
  }

  const outRoot = path.resolve(args.outRoot || stableOutputRoot(args, profile, tasksPath));
  const matrixPath = path.join(ROOT, 'scripts', 'run_model_matrix.mjs');
  const commandArgs = [
    matrixPath,
    '--mode', args.mode,
    '--profiles', profile.id,
    '--vlm-config', registry.path,
    '--tasks', tasksPath,
    '--out-root', outRoot,
    '--stop-on-error',
    '--skip-existing',
    ...args.captureArgs,
  ];
  if (args.dryRun) commandArgs.push('--dry-run');

  console.log(`[recording-job] profile=${profile.id}`);
  console.log(`[recording-job] evaluation_mode=${formalProfile ? 'formal-adapter' : 'raw-direct-api'}`);
  console.log(`[recording-job] tasks=${tasksPath}`);
  console.log(`[recording-job] output=${outRoot}`);
  const code = await runChild(process.execPath, commandArgs);
  if (code !== 0) throw new Error(`Recording job failed with exit code ${code}; rerun the same command to resume`);
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
