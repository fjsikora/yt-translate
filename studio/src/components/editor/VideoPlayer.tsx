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
      className,
    },
    ref
  ) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const isSeeking = useRef(false);

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
          videoRef.current.currentTime = time;
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
    useEffect(() => {
      if (!videoRef.current || isSeeking.current) return;

      const video = videoRef.current;
      const diff = Math.abs(video.currentTime - currentTime);

      // Only seek if difference is significant (> 0.1 seconds)
      // This prevents seeking during normal playback
      if (diff > 0.1) {
        isSeeking.current = true;
        video.currentTime = currentTime;
        setTimeout(() => {
          isSeeking.current = false;
        }, 100);
      }
    }, [currentTime]);

    // Handle time update from video
    const handleTimeUpdate = useCallback(() => {
      if (videoRef.current && !isSeeking.current) {
        onTimeUpdate(videoRef.current.currentTime);
      }
    }, [onTimeUpdate]);

    // Handle metadata loaded (duration available)
    const handleLoadedMetadata = useCallback(() => {
      if (videoRef.current) {
        onDurationChange(videoRef.current.duration);
      }
    }, [onDurationChange]);

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
            // Seek back 1 second
            e.preventDefault();
            if (videoRef.current) {
              const newTime = Math.max(0, videoRef.current.currentTime - 1);
              videoRef.current.currentTime = newTime;
              onTimeUpdate(newTime);
            }
            break;

          case "ArrowRight":
            // Seek forward 1 second
            e.preventDefault();
            if (videoRef.current) {
              const maxTime = videoRef.current.duration || Infinity;
              const newTime = Math.min(maxTime, videoRef.current.currentTime + 1);
              videoRef.current.currentTime = newTime;
              onTimeUpdate(newTime);
            }
            break;
        }
      };

      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }, [onTimeUpdate]);

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
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onPlay={handlePlay}
        onPause={handlePause}
        onEnded={handleEnded}
        onSeeked={() => {
          // Emit time update after seek completes
          if (videoRef.current) {
            onTimeUpdate(videoRef.current.currentTime);
          }
        }}
        preload="metadata"
      />
    );
  }
);

export default VideoPlayer;
