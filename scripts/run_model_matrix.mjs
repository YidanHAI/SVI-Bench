#!/usr/bin/env node
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  DEFAULT_VLM_CONFIG,
  loadVlmRegistry,
  safeProfileSnapshot,
  selectVlmProfiles,
} from './vlm_profiles.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

function timestamp() {
  const date = new Date();
  const pad = (value) => String(value).padStart(2, '0');
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}`
    + `_${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}

function printHelp() {
  console.log(`Usage:
  npm run capture:matrix -- --profiles all --tasks tasks.jsonl [capture args]

Matrix options:
  --profiles SELECTOR   all, comma-separated profile ids, or tag:NAME; default all
  --vlm-config PATH     Model registry, default ${DEFAULT_VLM_CONFIG}
  --mode MODE           upload or rtsp; default upload
  --out-root DIR        Matrix output root; each model gets its own subdirectory
  --stop-on-error       Stop after the first failed model; default continues
  --dry-run             Print the planned sequential commands without running them

All remaining options are forwarded to scripts/run_capture_profile.mjs. --tasks is required.
Use --skip-existing to resume only results whose profile fingerprint and backend identity match.`);
}

function optionPresent(args, option) {
  return args.some((arg) => arg === option || arg.startsWith(`${option}=`));
}

function optionValue(args, option) {
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === option) return args[index + 1] || '';
    if (args[index].startsWith(`${option}=`)) return args[index].slice(option.length + 1);
  }
  return '';
}

function parseArgs(argv) {
  const args = {
    profiles: 'all',
    vlmConfig: DEFAULT_VLM_CONFIG,
    mode: 'upload',
    outRoot: `outputs/model_matrix_${timestamp()}`,
    stopOnError: false,
    dryRun: false,
    captureArgs: [],
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = () => {
      if (index + 1 >= argv.length) throw new Error(`Missing value for ${arg}`);
      index += 1;
      return argv[index];
    };
    if (arg === '--profiles') args.profiles = next();
    else if (arg.startsWith('--profiles=')) args.profiles = arg.slice('--profiles='.length);
    else if (arg === '--vlm-config') args.vlmConfig = next();
    else if (arg.startsWith('--vlm-config=')) args.vlmConfig = arg.slice('--vlm-config='.length);
    else if (arg === '--mode') args.mode = next();
    else if (arg.startsWith('--mode=')) args.mode = arg.slice('--mode='.length);
    else if (arg === '--out-root') args.outRoot = next();
    else if (arg.startsWith('--out-root=')) args.outRoot = arg.slice('--out-root='.length);
    else if (arg === '--stop-on-error') args.stopOnError = true;
    else if (arg === '--dry-run' || arg === '--print') args.dryRun = true;
    else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    } else {
      args.captureArgs.push(arg);
    }
  }
  if (!['upload', 'rtsp'].includes(args.mode)) {
    throw new Error('--mode must be upload or rtsp');
  }
  if (!optionPresent(args.captureArgs, '--tasks')) {
    throw new Error('Model matrix requires an explicit --tasks PATH');
  }
  for (const forbidden of ['--out', '--vlm-profile', '--vlm-model', '--vlm-api-base', '--vlm-route']) {
    if (optionPresent(args.captureArgs, forbidden)) {
      throw new Error(`${forbidden} is controlled by the model matrix`);
    }
  }
  return args;
}

function quoteArg(value) {
  const text = String(value);
  if (/^[A-Za-z0-9_./:=@+-]+$/.test(text)) return text;
  return `'${text.replaceAll("'", "'\\''")}'`;
}

function redactArgs(args) {
  const redacted = [...args];
  for (let index = 0; index < redacted.length; index += 1) {
    if (['--web-password', '--vlm-api-key'].includes(redacted[index]) && index + 1 < redacted.length) {
      redacted[index + 1] = '[REDACTED]';
    } else if (/^--(?:web-password|vlm-api-key)=/.test(redacted[index])) {
      redacted[index] = `${redacted[index].split('=')[0]}=[REDACTED]`;
    }
  }
  return redacted;
}

function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function reusableCompletedRun(outputDir, fingerprint) {
  const snapshot = readJson(path.join(outputDir, 'model_profile.json'));
  const summary = readJson(path.join(outputDir, 'run_summary.json'));
  if (snapshot?.fingerprint !== fingerprint || summary?.status !== 'complete') return false;
  if (!Array.isArray(summary.task_results) || !summary.task_results.length) return false;
  return summary.task_results.every((task) => {
    if (!['ok', 'skipped'].includes(task.status) || task.vlm_backend_identity_ok !== true) {
      return false;
    }
    const taskSummaryPath = String(task.summary_path || '');
    const taskMp4Path = String(task.task_mp4 || '');
    if (!taskSummaryPath || !taskMp4Path) return false;
    if (!fs.existsSync(taskSummaryPath) || !fs.existsSync(taskMp4Path)) return false;
    const taskSummary = readJson(taskSummaryPath);
    return taskSummary?.status === 'ok'
      && taskSummary?.vlm_backend_identity?.ok === true
      && taskSummary?.files?.task_mp4 === taskMp4Path;
  });
}

function assertCompatibleOutput(outputDir, fingerprint) {
  const snapshotPath = path.join(outputDir, 'model_profile.json');
  if (!fs.existsSync(snapshotPath)) return;
  const existing = readJson(snapshotPath);
  if (!existing || existing.fingerprint !== fingerprint) {
    throw new Error(
      `Output ${outputDir} belongs to a different model profile; use a new --out-root`,
    );
  }
}

function runChild(command, argv) {
  return new Promise((resolve) => {
    const child = spawn(command, argv, { cwd: ROOT, stdio: 'inherit' });
    const forwardSignal = (signal) => {
      if (!child.killed) child.kill(signal);
    };
    const onSigint = () => forwardSignal('SIGINT');
    const onSigterm = () => forwardSignal('SIGTERM');
    process.once('SIGINT', onSigint);
    process.once('SIGTERM', onSigterm);
    child.once('error', (error) => {
      process.removeListener('SIGINT', onSigint);
      process.removeListener('SIGTERM', onSigterm);
      resolve({ code: 1, signal: '', error: error.message });
    });
    child.once('exit', (code, signal) => {
      process.removeListener('SIGINT', onSigint);
      process.removeListener('SIGTERM', onSigterm);
      resolve({ code: code ?? 1, signal: signal || '', error: '' });
    });
  });
}

function csvCell(value) {
  return `"${String(value ?? '').replaceAll('"', '""')}"`;
}

async function writeMatrixSummary(args, profiles, results, startedAt, status) {
  const summary = {
    status,
    started_at: startedAt,
    finished_at: status === 'running' ? null : new Date().toISOString(),
    output_root: path.resolve(args.outRoot),
    mode: args.mode,
    config: path.resolve(args.vlmConfig),
    selector: args.profiles,
    tasks: path.resolve(optionValue(args.captureArgs, '--tasks')),
    capture_args: redactArgs(args.captureArgs),
    profiles_total: profiles.length,
    profiles_finished: results.length,
    counts: results.reduce((counts, result) => {
      counts[result.status] = (counts[result.status] || 0) + 1;
      return counts;
    }, {}),
    results,
  };
  await fsp.writeFile(
    path.join(args.outRoot, 'matrix_summary.json'),
    `${JSON.stringify(summary, null, 2)}\n`,
  );
  const headers = [
    'profile_id', 'model', 'route', 'formal_eval', 'status', 'exit_code',
    'tasks_total', 'tasks_ok', 'tasks_failed', 'elapsed_s', 'output_dir', 'error',
  ];
  const lines = [
    headers.join(','),
    ...results.map((result) => headers.map((header) => csvCell(result[header])).join(',')),
  ];
  await fsp.writeFile(path.join(args.outRoot, 'matrix_summary.csv'), `${lines.join('\n')}\n`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const registry = loadVlmRegistry(args.vlmConfig);
  args.vlmConfig = registry.path;
  const profiles = selectVlmProfiles(registry, args.profiles);
  if (!profiles.length) throw new Error(`No enabled profiles selected by ${args.profiles}`);

  const wrapperPath = path.join(ROOT, 'scripts', 'run_capture_profile.mjs');
  console.log(`[model-matrix] ${profiles.length} sequential profile(s): ${profiles.map((item) => item.id).join(', ')}`);
  for (const profile of profiles) {
    const outputDir = path.resolve(args.outRoot, profile.id);
    const commandArgs = [
      wrapperPath,
      '--mode', args.mode,
      '--vlm-config', registry.path,
      '--vlm-profile', profile.id,
      '--out', outputDir,
      ...args.captureArgs,
    ];
    console.log(`[model-matrix] ${profile.id}: ${[process.execPath, ...redactArgs(commandArgs)].map(quoteArg).join(' ')}`);
  }
  if (args.dryRun) return;

  await fsp.mkdir(args.outRoot, { recursive: true });
  const startedAt = new Date().toISOString();
  const results = [];
  await writeMatrixSummary(args, profiles, results, startedAt, 'running');

  let stopped = false;
  for (let index = 0; index < profiles.length; index += 1) {
    const profile = profiles[index];
    const snapshot = safeProfileSnapshot(profile);
    const outputDir = path.resolve(args.outRoot, profile.id);
    await fsp.mkdir(outputDir, { recursive: true });
    assertCompatibleOutput(outputDir, snapshot.fingerprint);
    if (optionPresent(args.captureArgs, '--skip-existing')
      && reusableCompletedRun(outputDir, snapshot.fingerprint)) {
      const runSummary = readJson(path.join(outputDir, 'run_summary.json'));
      results.push({
        profile_id: profile.id,
        model: profile.model,
        route: profile.route,
        formal_eval: profile.formal_eval,
        status: 'skipped_existing',
        exit_code: 0,
        tasks_total: runSummary.tasks_total || 0,
        tasks_ok: runSummary.task_results?.length || 0,
        tasks_failed: 0,
        elapsed_s: 0,
        output_dir: outputDir,
        error: '',
      });
      await writeMatrixSummary(args, profiles, results, startedAt, 'running');
      continue;
    }

    await fsp.writeFile(
      path.join(outputDir, 'model_profile.json'),
      `${JSON.stringify({ ...snapshot, config_path: registry.path }, null, 2)}\n`,
    );
    const commandArgs = [
      wrapperPath,
      '--mode', args.mode,
      '--vlm-config', registry.path,
      '--vlm-profile', profile.id,
      '--out', outputDir,
      ...args.captureArgs,
    ];
    console.log(`[model-matrix] [${index + 1}/${profiles.length}] starting ${profile.id}`);
    const startedMs = Date.now();
    const childResult = await runChild(process.execPath, commandArgs);
    const runSummary = readJson(path.join(outputDir, 'run_summary.json'));
    const taskResults = Array.isArray(runSummary?.task_results) ? runSummary.task_results : [];
    const failedTasks = taskResults.filter((task) => !['ok', 'skipped'].includes(task.status));
    const ok = childResult.code === 0
      && runSummary?.status === 'complete'
      && taskResults.length > 0
      && failedTasks.length === 0
      && taskResults.every((task) => task.vlm_backend_identity_ok === true);
    const error = childResult.error
      || (childResult.signal ? `terminated by ${childResult.signal}` : '')
      || (!ok ? `capture exit=${childResult.code}, run_status=${runSummary?.status || 'missing'}` : '');
    const result = {
      profile_id: profile.id,
      model: profile.model,
      route: profile.route,
      formal_eval: profile.formal_eval,
      status: ok ? 'ok' : 'failed',
      exit_code: childResult.code,
      tasks_total: runSummary?.tasks_total ?? taskResults.length,
      tasks_ok: taskResults.length - failedTasks.length,
      tasks_failed: failedTasks.length,
      elapsed_s: (Date.now() - startedMs) / 1000,
      output_dir: outputDir,
      error,
    };
    results.push(result);
    await writeMatrixSummary(args, profiles, results, startedAt, 'running');
    if (!ok && (args.stopOnError || childResult.signal)) {
      stopped = Boolean(childResult.signal);
      break;
    }
  }

  const failures = results.filter((result) => result.status === 'failed');
  const finalStatus = stopped ? 'stopped' : (failures.length ? 'complete_with_errors' : 'complete');
  await writeMatrixSummary(args, profiles, results, startedAt, finalStatus);
  if (failures.length || stopped) {
    throw new Error(`Model matrix finished with ${failures.length} failed profile(s)`);
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
