#!/usr/bin/env node

import { spawn } from 'node:child_process';

function parseArgs(argv) {
  const args = {
    url: '',
    ffmpeg: 'ffmpeg',
    retryMs: 500,
    connectTimeoutMs: 15000,
    stallTimeoutMs: 20000,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const name = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`Missing value for ${name}`);
      i += 1;
      return argv[i];
    };
    if (name === '--url') args.url = next();
    else if (name === '--ffmpeg') args.ffmpeg = next();
    else if (name === '--retry-ms') args.retryMs = Number(next());
    else if (name === '--connect-timeout-ms') args.connectTimeoutMs = Number(next());
    else if (name === '--stall-timeout-ms') args.stallTimeoutMs = Number(next());
    else throw new Error(`Unknown argument: ${name}`);
  }
  if (!args.url) throw new Error('--url is required');
  for (const [name, value] of [
    ['--retry-ms', args.retryMs],
    ['--connect-timeout-ms', args.connectTimeoutMs],
    ['--stall-timeout-ms', args.stallTimeoutMs],
  ]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be positive`);
  }
  return args;
}

function emit(type, fields = {}) {
  process.stdout.write(`${JSON.stringify({ type, ...fields })}\n`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  let stopping = false;
  let child = null;
  let attempt = 0;
  let forceKillTimer = null;

  const stop = () => {
    stopping = true;
    const activeChild = child;
    if (!activeChild || activeChild.exitCode !== null || activeChild.signalCode !== null) return;
    activeChild.kill('SIGTERM');
    if (forceKillTimer) clearTimeout(forceKillTimer);
    forceKillTimer = setTimeout(() => {
      if (child === activeChild && activeChild.exitCode === null && activeChild.signalCode === null) {
        activeChild.kill('SIGKILL');
      }
    }, 1000);
  };
  process.on('SIGTERM', stop);
  process.on('SIGINT', stop);

  while (!stopping) {
    attempt += 1;
    let ready = false;
    let lastProgressAtMs = Date.now();
    const startedAtMs = Date.now();
    emit('reader_start', { attempt, url: args.url });
    child = spawn(args.ffmpeg, [
      '-hide_banner', '-loglevel', 'info',
      '-rtsp_transport', 'tcp',
      '-i', args.url,
      '-map', '0:v:0', '-an', '-c', 'copy',
      '-f', 'null', '-',
    ], { stdio: ['ignore', 'ignore', 'pipe'] });

    let stderrTail = '';
    child.stderr.on('data', (chunk) => {
      const text = chunk.toString();
      process.stderr.write(text);
      stderrTail = `${stderrTail}${text}`.slice(-8192);
      if (/frame=\s*\d+/.test(text)) lastProgressAtMs = Date.now();
      if (!ready && /Output #0/.test(stderrTail)) {
        ready = true;
        lastProgressAtMs = Date.now();
        emit('reader_ready', {
          attempt,
          url: args.url,
          ready_ms: Date.now() - startedAtMs,
        });
      }
    });

    const watchdog = setInterval(() => {
      const now = Date.now();
      const timeoutMs = ready ? args.stallTimeoutMs : args.connectTimeoutMs;
      const referenceMs = ready ? lastProgressAtMs : startedAtMs;
      if (!stopping && now - referenceMs > timeoutMs) {
        emit(ready ? 'reader_stalled' : 'reader_connect_timeout', {
          attempt,
          url: args.url,
          stalled_ms: now - referenceMs,
        });
        child?.kill('SIGKILL');
      }
    }, 1000);

    const result = await new Promise((resolve) => {
      let settled = false;
      const finish = (value) => {
        if (settled) return;
        settled = true;
        resolve(value);
      };
      child.once('error', (error) => finish({ code: null, signal: null, error: error.message }));
      child.once('close', (code, signal) => finish({ code, signal, error: '' }));
    });
    clearInterval(watchdog);
    if (forceKillTimer) {
      clearTimeout(forceKillTimer);
      forceKillTimer = null;
    }
    child = null;
    emit('reader_exit', {
      attempt,
      url: args.url,
      ready,
      code: result.code,
      signal: result.signal,
      error: result.error || undefined,
    });
    if (!stopping) await sleep(args.retryMs);
  }
}

main().catch((error) => {
  emit('reader_error', { error: error.message });
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
