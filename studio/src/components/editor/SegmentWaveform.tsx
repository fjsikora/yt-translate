"use client";

import { useEffect, useState, useRef, useMemo, useCallback } from "react";
import { generateWaveformData, isWaveformCached } from "@/lib/waveform";

interface SegmentWaveformProps {
  /** URL of the audio file */
  audioUrl: string;
  /** Width of the waveform container in pixels */
  width: number;
  /** Height of the waveform container in pixels */
  height: number;
  /** Color of the waveform bars */
  color?: string;
  /** Background color (transparent by default) */
  backgroundColor?: string;
  /** Whether to show loading state */
  showLoading?: boolean;
  /** Number of samples (affects resolution, scales with width) */
  baseSamples?: number;
  /** Zoom level - affects sample density */
  zoom?: number;
}

/**
 * Renders an audio waveform visualization for a timeline segment
 * Uses canvas for efficient rendering and scales with zoom level
 */
export function SegmentWaveform({
  audioUrl,
  width,
  height,
  color = "currentColor",
  backgroundColor = "transparent",
  showLoading = true,
  baseSamples = 100,
  zoom = 1,
}: SegmentWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [waveformData, setWaveformData] = useState<Float32Array | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Calculate samples based on width and zoom
  // More samples for wider segments and higher zoom levels
  const samples = useMemo(() => {
    const scaledSamples = Math.max(
      50,
      Math.min(500, Math.floor(baseSamples * Math.sqrt(zoom)))
    );
    return scaledSamples;
  }, [baseSamples, zoom]);

  // Load waveform data
  useEffect(() => {
    if (!audioUrl) {
      setWaveformData(null);
      return;
    }

    // Check if already cached
    if (isWaveformCached(audioUrl, samples)) {
      setIsLoading(false);
    } else {
      setIsLoading(true);
    }

    setError(null);

    generateWaveformData(audioUrl, { samples })
      .then((data) => {
        setWaveformData(data);
        setIsLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load waveform:", err);
        setError(err.message);
        setIsLoading(false);
      });
  }, [audioUrl, samples]);

  // Draw waveform on canvas
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !waveformData || waveformData.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas resolution (handle high DPI displays)
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // Draw waveform bars
    ctx.fillStyle = color;

    const barWidth = width / waveformData.length;
    const barSpacing = Math.max(0.5, barWidth * 0.1);
    const effectiveBarWidth = barWidth - barSpacing;
    const centerY = height / 2;

    for (let i = 0; i < waveformData.length; i++) {
      const peak = waveformData[i];
      const barHeight = Math.max(2, peak * (height - 4)); // Min 2px height
      const x = i * barWidth;
      const y = centerY - barHeight / 2;

      // Draw bar with rounded corners for smoother appearance
      ctx.beginPath();
      const radius = Math.min(effectiveBarWidth / 2, 2);
      const barX = x + barSpacing / 2;

      if (radius > 0 && effectiveBarWidth > 2) {
        // Rounded rectangle
        ctx.moveTo(barX + radius, y);
        ctx.lineTo(barX + effectiveBarWidth - radius, y);
        ctx.quadraticCurveTo(
          barX + effectiveBarWidth,
          y,
          barX + effectiveBarWidth,
          y + radius
        );
        ctx.lineTo(barX + effectiveBarWidth, y + barHeight - radius);
        ctx.quadraticCurveTo(
          barX + effectiveBarWidth,
          y + barHeight,
          barX + effectiveBarWidth - radius,
          y + barHeight
        );
        ctx.lineTo(barX + radius, y + barHeight);
        ctx.quadraticCurveTo(barX, y + barHeight, barX, y + barHeight - radius);
        ctx.lineTo(barX, y + radius);
        ctx.quadraticCurveTo(barX, y, barX + radius, y);
      } else {
        // Simple rectangle for narrow bars
        ctx.rect(barX, y, effectiveBarWidth, barHeight);
      }

      ctx.fill();
    }
  }, [waveformData, width, height, color, backgroundColor]);

  // Redraw when data or dimensions change
  useEffect(() => {
    drawWaveform();
  }, [drawWaveform]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      drawWaveform();
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [drawWaveform]);

  if (error) {
    return (
      <div
        className="flex items-center justify-center text-[10px] text-muted-foreground/50"
        style={{ width, height }}
      >
        {/* Empty on error - no waveform to show */}
      </div>
    );
  }

  if (isLoading && showLoading) {
    return (
      <div
        className="flex items-center justify-center"
        style={{ width, height }}
      >
        <div className="flex gap-0.5">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className="w-0.5 bg-current opacity-30 animate-pulse"
              style={{
                height: `${20 + Math.random() * 30}%`,
                animationDelay: `${i * 0.1}s`,
              }}
            />
          ))}
        </div>
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none"
      style={{
        width: `${width}px`,
        height: `${height}px`,
        opacity: waveformData ? 1 : 0,
        transition: "opacity 0.2s ease-in-out",
      }}
    />
  );
}

export default SegmentWaveform;
