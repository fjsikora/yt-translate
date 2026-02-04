"use client";

import { useRef, useEffect, useCallback, forwardRef, useImperativeHandle } from "react";

export interface VideoPlayerProps {
  src: string | undefined;
  isPlaying: boolean;
  currentTime: number;
  onTimeUpdate: (time: number) => void;
  onDurationChange: (duration: number) => void;
  onPlayStateChange: (isPlaying: boolean) => void;
  onEnded?: () => void;
  muted?: boolean;
  playbackRate?: number;
  className?: string;
}

export interface VideoPlayerRef {
  play: () => void;
  pause: () => void;
  seek: (time: number) => void;
  togglePlayback: () => void;
}

/**
 * VideoPlayer component with spacebar play/pause and timeline sync.
 *
 * Features:
 * - Spacebar toggles play/pause (when editor is focused)
 * - Time updates sync to timeline
 * - Click-to-seek from timeline updates video position
 * - Current time displayed in MM:SS.ms format
 */
export const VideoPlayer = forwardRef<VideoPlayerRef, VideoPlayerProps>(
  function VideoPlayer(
    {
      src,
      isPlaying,
      currentTime,
      onTimeUpdate,
      onDurationChange,
      onPlayStateChange,
      onEnded,
      muted,
      playbackRate,
      className,
    },
    ref
  ) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const isSeeking = useRef(false);
    const rate = playbackRate ?? 1.0;

    // Expose methods to parent via ref
    useImperativeHandle(ref, () => ({
      play: () => {
        videoRef.current?.play();
      },
      pause: () => {
        videoRef.current?.pause();
      },
      seek: (time: number) => {
        if (videoRef.current) {
          isSeeking.current = true;
          videoRef.current.currentTime = time * rate;
          // Reset seeking flag after seek completes
          setTimeout(() => {
            isSeeking.current = false;
          }, 100);
        }
      },
      togglePlayback: () => {
        if (videoRef.current) {
          if (videoRef.current.paused) {
            videoRef.current.play();
          } else {
            videoRef.current.pause();
          }
        }
      },
    }));

    // Apply playback rate to video element
    useEffect(() => {
      if (videoRef.current) {
        videoRef.current.playbackRate = rate;
      }
    }, [rate]);

    // Sync video playback state with isPlaying prop
    useEffect(() => {
      if (!videoRef.current) return;

      if (isPlaying && videoRef.current.paused) {
        videoRef.current.play().catch(() => {
          // Autoplay may be blocked by browser
          onPlayStateChange(false);
        });
      } else if (!isPlaying && !videoRef.current.paused) {
        videoRef.current.pause();
      }
    }, [isPlaying, onPlayStateChange]);

    // Sync video position with currentTime prop (from timeline seek)
    // Convert timeline-time → video-time using playback rate
    useEffect(() => {
      if (!videoRef.current || isSeeking.current) return;

      const video = videoRef.current;
      const videoTime = currentTime * rate;
      const diff = Math.abs(video.currentTime - videoTime);

      // Only seek if difference is significant (> 0.1 seconds)
      // This prevents seeking during normal playback
      if (diff > 0.1) {
        isSeeking.current = true;
        video.currentTime = videoTime;
        setTimeout(() => {
          isSeeking.current = false;
        }, 100);
      }
    }, [currentTime, rate]);

    // Handle time update from video
    // Convert video-time → timeline-time using playback rate
    const handleTimeUpdate = useCallback(() => {
      if (videoRef.current && !isSeeking.current) {
        onTimeUpdate(videoRef.current.currentTime / rate);
      }
    }, [onTimeUpdate, rate]);

    // Handle metadata loaded (duration available)
    // Convert video duration to timeline duration using playback rate
    const handleLoadedMetadata = useCallback(() => {
      if (videoRef.current) {
        onDurationChange(videoRef.current.duration / rate);
      }
    }, [onDurationChange, rate]);

    // Handle play/pause events from video
    const handlePlay = useCallback(() => {
      onPlayStateChange(true);
    }, [onPlayStateChange]);

    const handlePause = useCallback(() => {
      onPlayStateChange(false);
    }, [onPlayStateChange]);

    const handleEnded = useCallback(() => {
      onPlayStateChange(false);
      onEnded?.();
    }, [onPlayStateChange, onEnded]);

    // Keyboard shortcuts (spacebar play/pause, arrow keys seek)
    useEffect(() => {
      const handleKeyDown = (e: KeyboardEvent) => {
        // Don't trigger if user is typing in an input
        const target = e.target as HTMLElement;
        if (
          target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable
        ) {
          return;
        }

        switch (e.code) {
          case "Space":
            // Prevent page scroll
            e.preventDefault();
            // Toggle playback
            if (videoRef.current) {
              if (videoRef.current.paused) {
                videoRef.current.play();
              } else {
                videoRef.current.pause();
              }
            }
            break;

          case "ArrowLeft":
            // Seek back 1 second (in timeline-time)
            e.preventDefault();
            if (videoRef.current) {
              const currentTimeline = videoRef.current.currentTime / rate;
              const newTimeline = Math.max(0, currentTimeline - 1);
              videoRef.current.currentTime = newTimeline * rate;
              onTimeUpdate(newTimeline);
            }
            break;

          case "ArrowRight":
            // Seek forward 1 second (in timeline-time)
            e.preventDefault();
            if (videoRef.current) {
              const currentTimeline = videoRef.current.currentTime / rate;
              const maxTimeline = (videoRef.current.duration || Infinity) / rate;
              const newTimeline = Math.min(maxTimeline, currentTimeline + 1);
              videoRef.current.currentTime = newTimeline * rate;
              onTimeUpdate(newTimeline);
            }
            break;
        }
      };

      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }, [onTimeUpdate, rate]);

    if (!src) {
      return (
        <div className={`flex items-center justify-center bg-black ${className || ""}`}>
          <span className="text-muted-foreground">No video available</span>
        </div>
      );
    }

    return (
      <video
        ref={videoRef}
        className={`h-full w-full object-contain ${className || ""}`}
        src={src}
        muted={muted}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onPlay={handlePlay}
        onPause={handlePause}
        onEnded={handleEnded}
        onSeeked={() => {
          // Emit time update after seek completes (convert video-time → timeline-time)
          if (videoRef.current) {
            onTimeUpdate(videoRef.current.currentTime / rate);
          }
        }}
        preload="metadata"
      />
    );
  }
);

export default VideoPlayer;
