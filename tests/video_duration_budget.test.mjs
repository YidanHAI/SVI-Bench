import assert from 'node:assert/strict';
import test from 'node:test';

import { calculateMaximumTaskVideoDurationS } from '../scripts/video_quality.mjs';

test('maximum task-video duration always includes capture overhead', () => {
  assert.equal(calculateMaximumTaskVideoDurationS({
    minimumDurationS: 1503,
    responseWaitRequired: true,
    responseTimeoutS: 180,
    postResponseRecordingS: 1,
    maxExtraS: 5,
  }), 1689);

  assert.equal(calculateMaximumTaskVideoDurationS({
    minimumDurationS: 1503,
    responseWaitRequired: false,
    responseTimeoutS: 180,
    postResponseRecordingS: 1,
    maxExtraS: 5,
  }), 1508);
});
