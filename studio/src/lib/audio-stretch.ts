/**
 * Audio time-stretching utilities using Tone.js
 * Provides real-time preview of audio at different playback speeds
 */

import * as Tone from "tone";

// Singleton for managing audio preview playback
let grainPlayer: Tone.GrainPlayer | null = null;
let isInitialized = false;

/**
 * Initialize Tone.js audio context
 * Must be called after user interaction (browser security requirement)
 */
export async function initAudioContext(): Promise<void> {
  if (isInitialized) return;

  await Tone.start();
  isInitialized = true;
}

/**
 * Check if the audio context is initialized
 */
export function isAudioInitialized(): boolean {
  return isInitialized;
}

/**
 * Preview audio at a specific playback rate
 * Uses Tone.js GrainPlayer for high-quality time-stretching
 *
 * @param audioUrl - URL of the audio to preview
 * @param playbackRate - Speed factor (0.5 to 2.0)
 * @param startTime - Optional start position in seconds
 * @param duration - Optional duration to play in seconds
 */
export async function previewAudioAtSpeed(
  audioUrl: string,
  playbackRate: number,
  startTime = 0,
  duration?: number
): Promise<void> {
  // Ensure audio context is initialized
  await initAudioContext();

  // Stop any existing preview
  stopPreview();

  // Clamp playback rate to valid range
  const clampedRate = Math.max(0.5, Math.min(2.0, playbackRate));

  try {
    // Create a new GrainPlayer for time-stretching
    grainPlayer = new Tone.GrainPlayer({
      url: audioUrl,
      loop: false,
      grainSize: 0.1,
      overlap: 0.05,
      playbackRate: clampedRate,
      onload: () => {
        if (grainPlayer) {
          grainPlayer.toDestination();

          // Calculate actual duration accounting for playback rate
          const adjustedDuration = duration
            ? duration / clampedRate
            : undefined;

          // Start playback
          grainPlayer.start(Tone.now(), startTime, adjustedDuration);
        }
      },
    });
  } catch (error) {
    console.error("Failed to preview audio:", error);
    throw error;
  }
}

/**
 * Stop any currently playing preview
 */
export function stopPreview(): void {
  if (grainPlayer) {
    try {
      grainPlayer.stop();
      grainPlayer.dispose();
    } catch {
      // Ignore errors during cleanup
    }
    grainPlayer = null;
  }
}

/**
 * Calculate the new speed factor when stretching a segment
 *
 * @param originalDuration - Original duration of the segment in seconds
 * @param newDuration - New duration after stretching in seconds
 * @param currentSpeedFactor - Current speed factor of the segment
 * @returns New speed factor (clamped to 0.5-2.0 range)
 */
export function calculateSpeedFactor(
  originalDuration: number,
  newDuration: number,
  currentSpeedFactor: number = 1.0
): number {
  if (originalDuration <= 0 || newDuration <= 0) {
    return currentSpeedFactor;
  }

  // Speed factor is the ratio of how much faster/slower we play
  // If we stretch the segment (make it longer), we need to play faster
  // If we shrink the segment (make it shorter), we need to play slower
  const durationRatio = originalDuration / newDuration;
  const newSpeedFactor = currentSpeedFactor * durationRatio;

  // Clamp to valid range
  return Math.max(0.5, Math.min(2.0, newSpeedFactor));
}

/**
 * Calculate the original audio duration from display duration and speed factor
 *
 * @param displayDuration - Current display duration of the segment
 * @param speedFactor - Current speed factor
 * @returns Original audio duration
 */
export function getOriginalAudioDuration(
  displayDuration: number,
  speedFactor: number
): number {
  return displayDuration * speedFactor;
}

/**
 * Calculate what the new display duration would be for a given speed factor
 *
 * @param originalAudioDuration - Original audio file duration
 * @param speedFactor - Desired speed factor
 * @returns New display duration
 */
export function calculateDisplayDuration(
  originalAudioDuration: number,
  speedFactor: number
): number {
  return originalAudioDuration / speedFactor;
}

/**
 * Format speed factor for display (e.g., "1.25x")
 */
export function formatSpeedFactor(speedFactor: number): string {
  if (Math.abs(speedFactor - 1.0) < 0.01) {
    return "1x";
  }
  return `${speedFactor.toFixed(2)}x`;
}

// Speed factor constants
export const MIN_SPEED_FACTOR = 0.5;
export const MAX_SPEED_FACTOR = 2.0;
