/**
 * Waveform generation and caching utilities
 * Uses Web Audio API to extract waveform peaks from audio files
 */

// Cache for waveform data to avoid re-processing
const waveformCache = new Map<string, Float32Array>();

// Pending requests to prevent duplicate processing
const pendingRequests = new Map<string, Promise<Float32Array>>();

export interface WaveformOptions {
  /** Number of samples to extract (affects resolution) */
  samples?: number;
  /** Audio context to reuse (optional) */
  audioContext?: AudioContext;
}

/**
 * Generate waveform peak data from an audio URL
 * Returns normalized peak values between 0 and 1
 */
export async function generateWaveformData(
  audioUrl: string,
  options: WaveformOptions = {}
): Promise<Float32Array> {
  const { samples = 200 } = options;

  // Check cache first
  const cacheKey = `${audioUrl}:${samples}`;
  const cached = waveformCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  // Check if request is already pending
  const pending = pendingRequests.get(cacheKey);
  if (pending) {
    return pending;
  }

  // Create a new request
  const request = fetchAndProcessAudio(audioUrl, samples);
  pendingRequests.set(cacheKey, request);

  try {
    const waveformData = await request;
    waveformCache.set(cacheKey, waveformData);
    return waveformData;
  } finally {
    pendingRequests.delete(cacheKey);
  }
}

/**
 * Fetch audio and extract waveform peaks
 */
async function fetchAndProcessAudio(
  audioUrl: string,
  samples: number
): Promise<Float32Array> {
  // Create audio context
  const audioContext = new (window.AudioContext ||
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (window as any).webkitAudioContext)();

  try {
    // Fetch the audio file
    const response = await fetch(audioUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch audio: ${response.status}`);
    }

    const arrayBuffer = await response.arrayBuffer();

    // Decode the audio data
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    // Extract peaks from the first channel (or mix if stereo)
    const channelData = audioBuffer.getChannelData(0);
    const peaks = extractPeaks(channelData, samples);

    return peaks;
  } finally {
    // Close the audio context to free resources
    await audioContext.close();
  }
}

/**
 * Extract peak values from audio samples
 * Downsamples the audio data to the specified number of peaks
 */
function extractPeaks(data: Float32Array, numPeaks: number): Float32Array {
  const peaks = new Float32Array(numPeaks);
  const samplesPerPeak = Math.floor(data.length / numPeaks);

  if (samplesPerPeak === 0) {
    // Audio is shorter than requested peaks, just copy what we have
    for (let i = 0; i < Math.min(data.length, numPeaks); i++) {
      peaks[i] = Math.abs(data[i]);
    }
    return peaks;
  }

  for (let i = 0; i < numPeaks; i++) {
    const start = i * samplesPerPeak;
    const end = Math.min(start + samplesPerPeak, data.length);

    // Find the maximum absolute value in this segment
    let maxPeak = 0;
    for (let j = start; j < end; j++) {
      const value = Math.abs(data[j]);
      if (value > maxPeak) {
        maxPeak = value;
      }
    }

    peaks[i] = maxPeak;
  }

  // Normalize peaks to 0-1 range
  // Find max value without spread operator (for TypeScript compatibility)
  let maxValue = 0;
  for (let i = 0; i < peaks.length; i++) {
    if (peaks[i] > maxValue) {
      maxValue = peaks[i];
    }
  }

  if (maxValue > 0) {
    for (let i = 0; i < numPeaks; i++) {
      peaks[i] = peaks[i] / maxValue;
    }
  }

  return peaks;
}

/**
 * Clear the waveform cache for a specific URL or all entries
 */
export function clearWaveformCache(audioUrl?: string): void {
  if (audioUrl) {
    // Clear all entries for this URL (any sample count)
    const keysToDelete: string[] = [];
    waveformCache.forEach((_, key) => {
      if (key.startsWith(audioUrl)) {
        keysToDelete.push(key);
      }
    });
    keysToDelete.forEach((key) => waveformCache.delete(key));
  } else {
    waveformCache.clear();
  }
}

/**
 * Check if waveform data is cached for a URL
 */
export function isWaveformCached(audioUrl: string, samples = 200): boolean {
  const cacheKey = `${audioUrl}:${samples}`;
  return waveformCache.has(cacheKey);
}

/**
 * Get cache statistics
 */
export function getWaveformCacheStats(): { size: number; urls: string[] } {
  const urls = new Set<string>();
  waveformCache.forEach((_, key) => {
    const url = key.split(":")[0];
    urls.add(url);
  });
  return {
    size: waveformCache.size,
    urls: Array.from(urls),
  };
}
