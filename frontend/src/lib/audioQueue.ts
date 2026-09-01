/**
 * Gapless streamed-audio player.
 *
 * Segments arrive as base64 PCM (24 kHz, 16-bit signed LE, mono by default)
 * or, as a fallback, any container `decodeAudioData` understands (e.g. mp3).
 * Buffers are scheduled back-to-back on one AudioContext timeline, strictly
 * in `seq` order — out-of-order arrivals wait for their predecessors and are
 * never reordered.
 *
 * The AudioContext must be created from a user gesture (Safari/iOS): call
 * `start()` inside the click handler before any segment arrives.
 */

export interface AudioSegment {
  seq: number;
  format: string; // "pcm" | "mp3" | ...
  b64: string;
  sampleRate?: number;
  channels?: number;
}

interface ScheduledSource {
  node: AudioBufferSourceNode;
  end: number;
  seq: number;
}

const DEFAULT_SAMPLE_RATE = 24000;

export function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

/** Int16 little-endian PCM → Float32 samples in [-1, 1). */
export function pcm16ToFloat32(bytes: Uint8Array): Float32Array {
  const usable = bytes.byteLength - (bytes.byteLength % 2);
  const view = new DataView(bytes.buffer, bytes.byteOffset, usable);
  const out = new Float32Array(usable / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = view.getInt16(i * 2, true) / 32768;
  }
  return out;
}

export class AudioQueue {
  onPlayingChange: ((playing: boolean) => void) | null = null;

  private readonly createContext: () => AudioContext;
  private ctx: AudioContext | null = null;
  private gain: GainNode | null = null;
  private analyser: AnalyserNode | null = null;

  private epoch = 0; // bumped by stop(); orphans in-flight decodes of a dead turn
  private nextSeq = 0;
  private maxPlayedSeq = -1; // highest seq that played to its natural end
  private frozenPlayedSeq = -1; // ...as of the last stop(), for cancel reporting
  private pending = new Map<number, AudioBuffer>();
  private scheduled: ScheduledSource[] = [];
  private nextStartTime = 0;
  private firstStartTime: number | null = null;
  private scheduledDuration = 0;
  private frozenPlayed = 0;
  private playing = false;

  constructor(createContext?: () => AudioContext) {
    this.createContext =
      createContext ??
      (() => {
        type Ctor = typeof AudioContext;
        const Ctx: Ctor =
          window.AudioContext ??
          (window as unknown as { webkitAudioContext: Ctor }).webkitAudioContext;
        return new Ctx();
      });
  }

  /** Create/resume the AudioContext. Call from a user gesture. */
  start(): AudioContext {
    if (!this.ctx) {
      this.ctx = this.createContext();
      this.gain = this.ctx.createGain();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 512;
      this.gain.connect(this.analyser);
      this.analyser.connect(this.ctx.destination);
    }
    if (this.ctx.state === "suspended") void this.ctx.resume();
    return this.ctx;
  }

  get context(): AudioContext | null {
    return this.ctx;
  }

  /** Node the played signal passes through — feed this to a visualizer. */
  get outputNode(): AudioNode | null {
    return this.gain;
  }

  get isPlaying(): boolean {
    return this.playing;
  }

  /** Highest segment seq the listener heard to its end; -1 when none.
   * This is what the cancel endpoint's `played_seq` expects. */
  get playedSeq(): number {
    return this.firstStartTime !== null
      ? this.maxPlayedSeq
      : this.frozenPlayedSeq;
  }

  /** Seconds of audio the listener actually heard for the current turn. */
  get playedSeconds(): number {
    if (this.firstStartTime === null || !this.ctx) return this.frozenPlayed;
    const elapsed = this.ctx.currentTime - this.firstStartTime;
    return Math.min(Math.max(elapsed, 0), this.scheduledDuration);
  }

  /** Decode and enqueue one segment; schedules every ready-in-order buffer. */
  async enqueue(segment: AudioSegment): Promise<void> {
    const ctx = this.start();
    const epoch = this.epoch;
    if (segment.seq < this.nextSeq) return; // stale (already played)
    const buffer = await this.decode(ctx, segment);
    if (epoch !== this.epoch) return; // stop() ran while decoding — dead turn
    if (buffer) this.pending.set(segment.seq, buffer);
    else if (segment.seq === this.nextSeq) this.nextSeq++; // undecodable: skip, don't stall
    this.drain(ctx);
  }

  /** The stream has ended: no missing seq will ever arrive. Play whatever is
   * buffered beyond a hole — the backend skips a seq when one sentence's TTS
   * fails, and waiting on that hole would silence the rest of the answer. */
  finish(): void {
    const ctx = this.ctx;
    if (!ctx) return;
    while (this.pending.size > 0) {
      const waiting = [...this.pending.keys()].filter((s) => s >= this.nextSeq);
      if (waiting.length === 0) break;
      this.nextSeq = Math.min(...waiting);
      this.drain(ctx);
    }
  }

  /** Stop playback now, drop queued segments, reset ordering to seq 0. */
  stop(): void {
    this.epoch++;
    this.frozenPlayed = this.playedSeconds;
    this.frozenPlayedSeq = this.maxPlayedSeq;
    this.maxPlayedSeq = -1;
    for (const s of this.scheduled) {
      s.node.onended = null;
      try {
        s.node.stop();
      } catch {
        /* not started or already stopped */
      }
    }
    this.scheduled = [];
    this.pending.clear();
    this.nextSeq = 0;
    this.nextStartTime = 0;
    this.firstStartTime = null;
    this.scheduledDuration = 0;
    this.setPlaying(false);
  }

  private async decode(
    ctx: AudioContext,
    segment: AudioSegment,
  ): Promise<AudioBuffer | null> {
    try {
      const bytes = base64ToBytes(segment.b64);
      if (segment.format === "pcm") {
        const samples = pcm16ToFloat32(bytes);
        if (samples.length === 0) return null;
        const rate = segment.sampleRate ?? DEFAULT_SAMPLE_RATE;
        const buffer = ctx.createBuffer(segment.channels ?? 1, samples.length, rate);
        buffer.copyToChannel(samples, 0);
        return buffer;
      }
      const array = bytes.buffer.slice(
        bytes.byteOffset,
        bytes.byteOffset + bytes.byteLength,
      ) as ArrayBuffer;
      return await ctx.decodeAudioData(array);
    } catch {
      return null; // skip the segment rather than stall the queue
    }
  }

  private drain(ctx: AudioContext): void {
    while (this.pending.has(this.nextSeq)) {
      const seq = this.nextSeq;
      const buffer = this.pending.get(seq)!;
      this.pending.delete(seq);
      this.nextSeq++;

      const node = ctx.createBufferSource();
      node.buffer = buffer;
      node.connect(this.gain ?? ctx.destination);

      const startAt = Math.max(ctx.currentTime, this.nextStartTime);
      if (this.firstStartTime === null) {
        this.firstStartTime = startAt;
        this.frozenPlayed = 0;
      }
      this.nextStartTime = startAt + buffer.duration;
      this.scheduledDuration += buffer.duration;

      const entry: ScheduledSource = { node, end: this.nextStartTime, seq };
      node.onended = () => {
        // stop() nulls this handler first, so reaching here means the
        // segment played to its natural end.
        this.maxPlayedSeq = Math.max(this.maxPlayedSeq, entry.seq);
        this.scheduled = this.scheduled.filter((s) => s !== entry);
        if (this.scheduled.length === 0 && this.pending.size === 0) {
          this.setPlaying(false);
        }
      };
      this.scheduled.push(entry);
      node.start(startAt);
      this.setPlaying(true);
    }
  }

  private setPlaying(playing: boolean): void {
    if (this.playing === playing) return;
    this.playing = playing;
    this.onPlayingChange?.(playing);
  }

}
