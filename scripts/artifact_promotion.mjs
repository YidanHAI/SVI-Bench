import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';

async function hashFile(filePath) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = fs.createReadStream(filePath);
    stream.on('error', reject);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

function safeRelativeFile(relativePath) {
  const normalized = path.normalize(String(relativePath || ''));
  if (
    !normalized
    || path.isAbsolute(normalized)
    || normalized === '..'
    || normalized.startsWith(`..${path.sep}`)
  ) {
    throw new Error(`Artifact path must remain inside the task directory: ${relativePath}`);
  }
  return normalized;
}

export async function relocateDirectory(sourceDir, destinationDir) {
  await fsp.rm(destinationDir, { recursive: true, force: true });
  try {
    await fsp.rename(sourceDir, destinationDir);
    return { method: 'rename' };
  } catch (error) {
    if (error?.code !== 'EXDEV') throw error;
  }

  const stagingDir = `${destinationDir}.copying-${process.pid}-${Date.now()}`;
  await fsp.rm(stagingDir, { recursive: true, force: true });
  try {
    await fsp.cp(sourceDir, stagingDir, {
      recursive: true,
      force: false,
      errorOnExist: true,
    });
    await fsp.rename(stagingDir, destinationDir);
    await fsp.rm(sourceDir, { recursive: true, force: true });
    return { method: 'copy' };
  } catch (error) {
    await fsp.rm(stagingDir, { recursive: true, force: true }).catch(() => {});
    throw error;
  }
}

export async function promoteTaskDirectory({
  sourceDir,
  destinationDir,
  relativeVideoPath,
  validateVideo,
  forceCopy = false,
}) {
  const videoRelative = safeRelativeFile(relativeVideoPath);
  await fsp.rm(destinationDir, { recursive: true, force: true });

  if (!forceCopy) {
    try {
      await fsp.rename(sourceDir, destinationDir);
      return { method: 'rename', video_sha256: null, validation: null };
    } catch (error) {
      if (error?.code !== 'EXDEV') throw error;
    }
  }

  const stagingDir = `${destinationDir}.promoting-${process.pid}-${Date.now()}`;
  await fsp.rm(stagingDir, { recursive: true, force: true });
  try {
    await fsp.cp(sourceDir, stagingDir, {
      recursive: true,
      force: false,
      errorOnExist: true,
    });
    const sourceVideo = path.join(sourceDir, videoRelative);
    const stagedVideo = path.join(stagingDir, videoRelative);
    const [sourceHash, stagedHash] = await Promise.all([
      hashFile(sourceVideo),
      hashFile(stagedVideo),
    ]);
    if (sourceHash !== stagedHash) {
      throw new Error(`Copied task MP4 checksum mismatch: ${sourceHash} != ${stagedHash}`);
    }
    const validation = await validateVideo(stagedVideo);
    if (validation?.ok !== true) {
      throw new Error(`Copied task MP4 decode failed: ${validation?.error || 'unknown error'}`);
    }
    await fsp.rename(stagingDir, destinationDir);
    await fsp.rm(sourceDir, { recursive: true, force: true });
    return {
      method: 'copy-verify-rename',
      video_sha256: stagedHash,
      validation,
    };
  } catch (error) {
    await fsp.rm(stagingDir, { recursive: true, force: true }).catch(() => {});
    throw error;
  }
}
