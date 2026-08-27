#!/usr/bin/env node

import fs from 'node:fs';
import { WebSocketServer } from 'ws';

const port = Number(process.env.MOCK_REALTIME_PORT || 0);
const eventsPath = process.env.MOCK_REALTIME_EVENTS || '';
if (!Number.isInteger(port) || port <= 0) throw new Error('MOCK_REALTIME_PORT is required');
if (!eventsPath) throw new Error('MOCK_REALTIME_EVENTS is required');

let sessionSequence = 0;

function record(event) {
  fs.appendFileSync(eventsPath, `${JSON.stringify(event)}\n`);
}

const server = new WebSocketServer({ port, host: '127.0.0.1' });
server.on('connection', (socket, request) => {
  const authorization = String(request.headers.authorization || '');
  record({ event: 'connection', authorization });
  socket.send(JSON.stringify({ type: 'session.queue_done' }));
  let querySession = false;
  let replied = false;
  socket.on('message', (raw) => {
    const message = JSON.parse(raw.toString());
    if (message.type === 'session.init') {
      sessionSequence += 1;
      const instruction = String(message.payload?.system_prompt || '');
      querySession = instruction.includes('USER QUERY:');
      record({ event: 'session.init', instruction, query_session: querySession });
      socket.send(JSON.stringify({
        type: 'session.created',
        session_id: `mock-session-${sessionSequence}`,
      }));
      return;
    }
    if (message.type === 'input.append') {
      const audio = Buffer.from(String(message.input?.audio || ''), 'base64');
      const audioNonzero = audio.some((value) => value !== 0);
      record({
        event: 'input.append',
        frame_count: Array.isArray(message.input?.video_frames)
          ? message.input.video_frames.length
          : 0,
        force_listen: message.input?.force_listen === true,
        audio_present: Boolean(message.input?.audio),
        audio_bytes: audio.length,
        audio_nonzero: audioNonzero,
        query_session: querySession,
        has_text_field: Object.prototype.hasOwnProperty.call(message.input || {}, 'text'),
      });
      if ((querySession || audioNonzero) && !message.input?.force_listen && !replied) {
        replied = true;
        socket.send(JSON.stringify({
          type: 'response.output.delta',
          kind: 'text',
          text: 'mock reply',
          response_id: 'mock-response-1',
        }));
        socket.send(JSON.stringify({
          type: 'response.output.delta',
          kind: 'listen',
          response_id: 'mock-response-1',
        }));
      }
      return;
    }
    if (message.type === 'session.close') {
      record({ event: 'session.close', reason: String(message.reason || '') });
      socket.send(JSON.stringify({ type: 'session.closed', reason: message.reason || '' }));
    }
  });
});

server.on('listening', () => process.stdout.write('ready\n'));
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.once(signal, () => server.close(() => process.exit(0)));
}
