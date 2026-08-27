#!/usr/bin/env node

import { once } from 'node:events';
import { spawn } from 'node:child_process';
import readline from 'node:readline';

function parseArgs(argv) {
  const args = {
    input: '',
    output: '',
    ffmpeg: 'ffmpeg',
    seekS: 0,
    fps: 12,
    gopS: 1,
    crf: 28,
    maxrate: '',
    bufsize: '',
    startFrozen: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const name = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`Missing value for ${name}`);
      i += 1;
      return argv[i];
    };
    if (name === '--input') args.input = next();
    else if (name === '--output') args.output = next();
    else if (name === '--ffmpeg') args.ffmpeg = next();
    else if (name === '--seek-s') args.seekS = Number(next());
    else if (name === '--fps') args.fps = Number(next());
    else if (name === '--gop-s') args.gopS = Number(next());
    else if (name === '--crf') args.crf = Number(next());
    else if (name === '--maxrate') args.maxrate = next();
    else if (name === '--bufsize') args.bufsize = next();
    else if (name === '--start-frozen') args.startFrozen = true;
    else throw new Error(`Unknown argument: ${name}`);
  }
  if (!args.input || !args.output) throw new Error('--input and --output are required');
  if (!Number.isFinite(args.seekS) || args.seekS < 0) throw new Error('--seek-s must be non-negative');
  if (!Number.isFinite(args.fps) || args.fps <= 0) throw new Error('--fps must be positive');
  return args;
}

function emit(type, fields = {}) {
  process.stdout.write(`${JSON.stringify({ type, ...fields })}\n`);
}

async function probeVideo(ffmpeg, input) {
  const ffprobe = ffmpeg.includes('/')
    ? `${ffmpeg.slice(0, ffmpeg.lastIndexOf('/'))}/ffprobe`
    : 'ffprobe';
  const proc = spawn(ffprobe, [
    '-v', 'error',
    '-select_streams', 'v:0',
    '-show_entries', 'stream=width,height',
    '-of', 'json',
    input,
  ], { stdio: ['ignore', 'pipe', 'pipe'] });
  let stdout = '';
  let stderr = '';
  proc.stdout.on('data', (chunk) => { stdout += chunk.toString(); });
  proc.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
  const [code] = await once(proc, 'close');
  if (code !== 0) throw new Error(`ffprobe failed: ${stderr.trim()}`);
  const stream = JSON.parse(stdout).streams?.[0];
  const width = Number(stream?.width);
  const height = Number(stream?.height);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error(`Could not probe video dimensions for ${input}`);
  }
  return { width, height };
}

class RawFrameReader {
  constructor(stream, frameSize) {
    this.iterator = stream[Symbol.asyncIterator]();
    this.frameSize = frameSize;
    this.buffer = Buffer.alloc(0);
    this.done = false;
  }

  async next() {
    while (this.buffer.length < this.frameSize && !this.done) {
      const item = await this.iterator.next();
      if (item.done) {
        this.done = true;
        break;
      }
      this.buffer = this.buffer.length
        ? Buffer.concat([this.buffer, item.value])
        : item.value;
    }
    if (this.buffer.length < this.frameSize) return null;
    const frame = this.buffer.subarray(0, this.frameSize);
    this.buffer = this.buffer.subarray(this.frameSize);
    return frame;
  }
}

async function writeFrame(stream, frame) {
  if (stream.destroyed || !stream.writable) throw new Error('encoder stdin is not writable');
  if (stream.write(frame)) return;
  await once(stream, 'drain');
}

function waitForClose(proc) {
  if (proc.exitCode !== null || proc.signalCode !== null) {
    return Promise.resolve([proc.exitCode, proc.signalCode]);
  }
  return once(proc, 'close');
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const fps = Math.max(1, Math.round(args.fps));
  const gop = Math.max(1, Math.round(fps * args.gopS));
  const { width, height } = await probeVideo(args.ffmpeg, args.input);
  const frameSize = Math.ceil(width * height * 3 / 2);

  const decoderArgs = [
    '-hide_banner', '-loglevel', 'warning',
    ...(args.seekS > 0.001 ? ['-ss', args.seekS.toFixed(3)] : []),
    '-i', args.input,
    '-map', '0:v:0', '-an', '-sn', '-dn',
    '-pix_fmt', 'yuv420p', '-f', 'rawvideo', 'pipe:1',
  ];
  const decoder = spawn(args.ffmpeg, decoderArgs, { stdio: ['ignore', 'pipe', 'pipe'] });
  decoder.stderr.pipe(process.stderr, { end: false });

  const encoderArgs = [
    '-hide_banner', '-loglevel', 'info',
    '-re', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
    '-video_size', `${width}x${height}`, '-framerate', String(fps),
    '-i', 'pipe:0', '-an',
    '-c:v', 'libx264', '-preset', 'veryfast', '-tune', 'zerolatency',
    '-pix_fmt', 'yuv420p', '-profile:v', 'baseline',
    '-g', String(gop), '-keyint_min', String(gop), '-sc_threshold', '0',
    '-bf', '0', '-refs', '1', '-crf', String(args.crf),
    '-x264-params', 'repeat-headers=1:force-cfr=1',
    ...(args.maxrate ? ['-maxrate', args.maxrate] : []),
    ...(args.bufsize ? ['-bufsize', args.bufsize] : []),
    '-avoid_negative_ts', 'make_zero', '-f', 'flv', args.output,
  ];
  const encoder = spawn(args.ffmpeg, encoderArgs, { stdio: ['pipe', 'ignore', 'pipe'] });
  let ready = false;
  let encoderOutputReady = false;
  let encoderStderr = '';
  encoder.stderr.on('data', (chunk) => {
    const text = chunk.toString();
    process.stderr.write(text);
    encoderStderr = `${encoderStderr}${text}`.slice(-8192);
    if (/Output #0|frame=\s*\d+/.test(encoderStderr)) encoderOutputReady = true;
  });

  let stopping = false;
  let frozen = args.startFrozen;
  let eof = false;
  let sourceFrames = 0;
  let outputFrames = 0;
  let currentFrame = null;
  let lastFrameWrittenAtMs = Date.now();
  const sourcePositionS = () => args.seekS + Math.max(0, sourceFrames - 1) / fps;
  const reader = new RawFrameReader(decoder.stdout, frameSize);
  const progressWatchdog = setInterval(() => {
    const stalledMs = Date.now() - lastFrameWrittenAtMs;
    if (!stopping && ready && stalledMs > 30000) {
      emit('relay_output_stalled', {
        stalled_ms: stalledMs,
        source_position_s: sourcePositionS(),
        frozen,
      });
      encoder.kill('SIGKILL');
    }
  }, 1000);

  const commands = readline.createInterface({ input: process.stdin });
  commands.on('line', (line) => {
    const command = line.trim().toLowerCase();
    if (command === 'freeze') {
      frozen = true;
      emit('relay_frozen', { source_position_s: sourcePositionS() });
    } else if (command === 'resume') {
      frozen = false;
      emit('relay_resumed', { source_position_s: sourcePositionS() });
    } else if (command === 'stop') {
      stopping = true;
    }
  });

  const terminate = () => { stopping = true; };
  process.on('SIGTERM', terminate);
  process.on('SIGINT', terminate);

  const intervalMs = 1000 / fps;
  let nextFrameAt = Date.now();
  try {
    while (!stopping) {
      if (!currentFrame && !eof) {
        const frame = await reader.next();
        if (frame) {
          currentFrame = frame;
          sourceFrames = 1;
        } else {
          eof = true;
        }
      } else if (!frozen && !eof) {
        const frame = await reader.next();
        if (frame) {
          currentFrame = frame;
          sourceFrames += 1;
        } else {
          eof = true;
          emit('relay_source_eof', { source_position_s: sourcePositionS() });
        }
      }
      if (!currentFrame) throw new Error('decoder produced no video frame');
      await writeFrame(encoder.stdin, currentFrame);
      lastFrameWrittenAtMs = Date.now();
      outputFrames += 1;
      if (!ready && encoderOutputReady) {
        ready = true;
        emit('relay_ready', { width, height, fps, seek_s: args.seekS });
      }
      if (outputFrames % fps === 0) {
        emit('relay_progress', {
          source_position_s: sourcePositionS(),
          frozen,
          output_frames: outputFrames,
        });
      }
      nextFrameAt += intervalMs;
      const delayMs = nextFrameAt - Date.now();
      if (delayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      } else if (delayMs < -intervalMs * 2) {
        nextFrameAt = Date.now();
      }
    }
  } catch (error) {
    emit('relay_failed', {
      error: error.message,
      source_position_s: sourcePositionS(),
      frozen,
    });
    throw error;
  } finally {
    clearInterval(progressWatchdog);
    commands.close();
    encoder.stdin.end();
    decoder.kill('SIGTERM');
    const encoderClosed = waitForClose(encoder);
    const decoderClosed = waitForClose(decoder);
    const timeout = setTimeout(() => {
      if (encoder.exitCode === null && encoder.signalCode === null) encoder.kill('SIGKILL');
      if (decoder.exitCode === null && decoder.signalCode === null) decoder.kill('SIGKILL');
    }, 3000);
    await Promise.allSettled([encoderClosed, decoderClosed]);
    clearTimeout(timeout);
  }

  emit('relay_stopped', { source_position_s: sourcePositionS() });
}

main().catch((error) => {
  emit('relay_error', { error: error.message });
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
