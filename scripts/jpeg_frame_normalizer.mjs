import { spawn } from 'node:child_process';

const JPEG_SOF_MARKERS = new Set([
  0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf,
]);

export function jpegDimensions(bytes) {
  const buffer = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes);
  if (buffer.length < 4 || buffer[0] !== 0xff || buffer[1] !== 0xd8) {
    throw new Error('Input is not a JPEG image');
  }
  let offset = 2;
  while (offset + 3 < buffer.length) {
    while (offset < buffer.length && buffer[offset] !== 0xff) offset += 1;
    while (offset < buffer.length && buffer[offset] === 0xff) offset += 1;
    if (offset >= buffer.length) break;
    const marker = buffer[offset];
    offset += 1;
    if (marker === 0xd8 || marker === 0x01) continue;
    if (marker === 0xd9 || marker === 0xda) break;
    if (offset + 1 >= buffer.length) break;
    const segmentLength = buffer.readUInt16BE(offset);
    if (segmentLength < 2 || offset + segmentLength > buffer.length) break;
    if (JPEG_SOF_MARKERS.has(marker)) {
      if (segmentLength < 7) throw new Error('JPEG SOF segment is truncated');
      const height = buffer.readUInt16BE(offset + 3);
      const width = buffer.readUInt16BE(offset + 5);
      if (width <= 0 || height <= 0) throw new Error('JPEG dimensions are invalid');
      return { width, height };
    }
    offset += segmentLength;
  }
  throw new Error('JPEG dimensions were not found');
}

function decodeJpegDataUrl(value) {
  const compact = String(value || '').replace(/\s+/g, '');
  const match = compact.match(/^data:image\/(?:jpeg|jpg);base64,([A-Za-z0-9+/=]+)$/i);
  if (!match) throw new Error('Expected an inline JPEG data URL');
  const bytes = Buffer.from(match[1], 'base64');
  return { compact, bytes, dimensions: jpegDimensions(bytes) };
}

function resizeJpeg(bytes, width, height, timeoutMs) {
  return new Promise((resolve, reject) => {
    const filter = `scale=${width}:${height}:force_original_aspect_ratio=decrease,`
      + `crop=min(iw\\,${width}):min(ih\\,${height}),`
      + `pad=${width}:${height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1`;
    const child = spawn(process.env.FFMPEG_BIN || 'ffmpeg', [
      '-loglevel', 'error',
      '-f', 'image2pipe',
      '-vcodec', 'mjpeg',
      '-i', 'pipe:0',
      '-vf', filter,
      '-frames:v', '1',
      '-f', 'image2pipe',
      '-vcodec', 'mjpeg',
      '-pix_fmt', 'yuvj444p',
      '-q:v', '3',
      'pipe:1',
    ], { stdio: ['pipe', 'pipe', 'pipe'] });
    const stdout = [];
    const stderr = [];
    let outputBytes = 0;
    let settled = false;
    const finish = (error, value = null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error);
      else resolve(value);
    };
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      finish(new Error(`JPEG normalization timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    child.stdout.on('data', (chunk) => {
      outputBytes += chunk.length;
      if (outputBytes > 32 * 1024 * 1024) {
        child.kill('SIGKILL');
        finish(new Error('JPEG normalization output exceeded 32 MiB'));
        return;
      }
      stdout.push(chunk);
    });
    child.stderr.on('data', (chunk) => {
      if (stderr.reduce((total, item) => total + item.length, 0) < 64 * 1024) {
        stderr.push(chunk);
      }
    });
    child.once('error', finish);
    child.once('exit', (code, signal) => {
      if (code !== 0) {
        finish(new Error(
          `ffmpeg JPEG normalization failed: code=${code}, signal=${signal || ''}, `
          + `stderr=${Buffer.concat(stderr).toString('utf8').slice(-2000)}`,
        ));
        return;
      }
      const output = Buffer.concat(stdout);
      const dimensions = jpegDimensions(output);
      if (dimensions.width !== width || dimensions.height !== height) {
        finish(new Error(
          `JPEG normalization produced ${dimensions.width}x${dimensions.height}, `
          + `expected ${width}x${height}`,
        ));
        return;
      }
      finish(null, output);
    });
    child.stdin.once('error', (error) => {
      if (error.code !== 'EPIPE') finish(error);
    });
    child.stdin.end(bytes);
  });
}

export async function normalizeJpegDataUrlToReference(
  sourceDataUrl,
  referenceDataUrl,
  { timeoutMs = 30000 } = {},
) {
  const source = decodeJpegDataUrl(sourceDataUrl);
  const reference = decodeJpegDataUrl(referenceDataUrl);
  const sameSize = source.dimensions.width === reference.dimensions.width
    && source.dimensions.height === reference.dimensions.height;
  if (sameSize) {
    return {
      imageUrl: source.compact,
      normalized: false,
      sourceSize: source.dimensions,
      targetSize: reference.dimensions,
    };
  }
  const output = await resizeJpeg(
    source.bytes,
    reference.dimensions.width,
    reference.dimensions.height,
    timeoutMs,
  );
  return {
    imageUrl: `data:image/jpeg;base64,${output.toString('base64')}`,
    normalized: true,
    sourceSize: source.dimensions,
    targetSize: reference.dimensions,
  };
}
