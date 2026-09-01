import { describe, expect, it } from "vitest";
import { AudioQueue } from "../audioQueue";

/* ---- minimal fake WebAudio ---------------------------------------- */

class FakeSource {
  buffer: { duration: number } | null = null;
  startedAt: number | null = null;
  stopped = false;
  onended: (() => void) | null = null;
  connect(): void {}
  start(at: number): void {
    this.startedAt = at;
  }
  stop(): void {
    this.stopped = true;
  }
}

class FakeContext {
  currentTime = 0;
  state = "running";
  destination = {};
  sources: FakeSource[] = [];
  createGain() {
    return { connect: () => {}, gain: { value: 1 } };
  }
  createAnalyser() {
    return {
      fftSize: 512,
      connect: () => {},
      getByteTimeDomainData: () => {},
    };
  }
  createBuffer(_ch: number, length: number, rate: number) {
    return { duration: length / rate, copyToChannel: () => {} };
  }
  createBufferSource() {
    const s = new FakeSource();
    this.sources.push(s);
    return s;
  }
  resume() {
    return Promise.resolve();
  }
  decodeAudioData() {
    return Promise.reject(new Error("container decode not available in tests"));
  }
}

/** n silent PCM16 samples as base64 (24 kHz → duration n/24000 s). */
function pcmB64(samples: number): string {
  return Buffer.alloc(samples * 2).toString("base64");
}

function makeQueue() {
  const ctx = new FakeContext();
  const queue = new AudioQueue(() => ctx as unknown as AudioContext);
  return { ctx, queue };
}

/* ---- tests -------------------------------------------------------- */

describe("AudioQueue", () => {
  it("schedules out-of-order segments strictly in seq order, back-to-back", async () => {
    const { ctx, queue } = makeQueue();
    const seg = (seq: number, samples: number) => ({
      seq,
      format: "pcm",
      b64: pcmB64(samples),
      sampleRate: 24000,
    });

    // seq 1 arrives first — nothing may play yet.
    await queue.enqueue(seg(1, 12000)); // 0.5 s
    expect(ctx.sources).toHaveLength(0);

    // seq 0 arrives — both schedule, in order, gapless.
    await queue.enqueue(seg(0, 24000)); // 1.0 s
    expect(ctx.sources).toHaveLength(2);
    expect(ctx.sources[0].buffer?.duration).toBeCloseTo(1.0);
    expect(ctx.sources[1].buffer?.duration).toBeCloseTo(0.5);
    expect(ctx.sources[0].startedAt).toBeCloseTo(0);
    expect(ctx.sources[1].startedAt).toBeCloseTo(1.0);

    // a later segment continues the same timeline.
    await queue.enqueue(seg(2, 6000)); // 0.25 s
    expect(ctx.sources[2].startedAt).toBeCloseTo(1.5);
    expect(queue.isPlaying).toBe(true);
  });

  it("stop() silences scheduled audio, drops the queue, and restarts at seq 0", async () => {
    const { ctx, queue } = makeQueue();
    await queue.enqueue({ seq: 0, format: "pcm", b64: pcmB64(24000) });
    await queue.enqueue({ seq: 2, format: "pcm", b64: pcmB64(24000) }); // waits for 1

    ctx.currentTime = 0.4;
    queue.stop();
    expect(ctx.sources.every((s) => s.stopped)).toBe(true);
    expect(queue.isPlaying).toBe(false);
    expect(queue.playedSeconds).toBeCloseTo(0.4); // what the listener heard

    // a new turn starts cleanly at seq 0 from the current time.
    await queue.enqueue({ seq: 0, format: "pcm", b64: pcmB64(12000) });
    const fresh = ctx.sources[ctx.sources.length - 1];
    expect(fresh.startedAt).toBeCloseTo(0.4);
    expect(queue.isPlaying).toBe(true);
  });
});
