/**
 * One waveform for both directions: point it at the mic's
 * MediaStreamSource while recording, or the playback gain node while the
 * answer is spoken. Draws real analyser data on a canvas; optionally reports
 * RMS level so callers can drive CSS (mic ring).
 *
 * The stroke is the element's own `color`, so the caller's class picks the
 * direction: `.viz--input` draws coral (live), `.viz--speaking` draws blue.
 */

import { useEffect, useRef } from "react";

interface VisualizerProps {
  ctx: AudioContext | null;
  source: AudioNode | null;
  onLevel?: (rms: number) => void;
  className?: string;
}

export function Visualizer({ ctx, source, onLevel, className }: VisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onLevelRef = useRef(onLevel);
  onLevelRef.current = onLevel;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!ctx || !source || !canvas) return;

    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.75;
    source.connect(analyser);
    const data = new Uint8Array(analyser.fftSize);

    const g = canvas.getContext("2d");
    let raf = 0;

    const draw = () => {
      raf = requestAnimationFrame(draw);
      if (!g) return;
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (w === 0 || h === 0) return;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);

      analyser.getByteTimeDomainData(data);
      let sum = 0;
      g.beginPath();
      const step = Math.max(1, Math.floor(data.length / w));
      for (let x = 0; x < w; x++) {
        const v = (data[Math.min(x * step, data.length - 1)] - 128) / 128;
        sum += v * v;
        const y = h / 2 + v * (h / 2 - 1.5);
        if (x === 0) g.moveTo(x, y);
        else g.lineTo(x, y);
      }
      g.strokeStyle = getComputedStyle(canvas).color;
      g.lineWidth = 1.75;
      g.lineJoin = "round";
      g.lineCap = "round";
      g.stroke();
      onLevelRef.current?.(Math.sqrt(sum / w));
    };
    draw();

    return () => {
      cancelAnimationFrame(raf);
      try {
        source.disconnect(analyser);
      } catch {
        /* source already gone */
      }
      analyser.disconnect();
      onLevelRef.current?.(0);
    };
  }, [ctx, source]);

  return (
    <canvas
      ref={canvasRef}
      className={className ? `viz ${className}` : "viz"}
      aria-hidden="true"
    />
  );
}
