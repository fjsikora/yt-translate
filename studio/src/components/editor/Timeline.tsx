"use client";

import { useCallback, useEffect, useRef, useMemo, useState } from "react";
import {
  TimelineProvider,
  useTimelineContext,
  TrackJSON,
} from "@twick/timeline";
import { cn } from "@/lib/utils";
import { SegmentWaveform } from "./SegmentWaveform";
import {
  calculateSpeedFactor,
  MIN_SPEED_FACTOR,
  MAX_SPEED_FACTOR,
  previewAudioAtSpeed,
  stopPreview,
} from "@/lib/audio-stretch";

// Types matching the project data structure
export interface TimelineSegment {
  id: string;
  track_id: string;
  speaker?: string;
  original_text?: string;
  translated_text?: string;
  start_time: number;
  end_time: number;
  speed_factor: number;
  audio_url?: string;
}

export interface TimelineTrack {
  id: string;
  name: string;
  type: "vocals" | "background" | "dubbed" | "video";
  muted: boolean;
  solo: boolean;
  volume: number;
  segments: TimelineSegment[];
}

// Drag state interface for segment repositioning
interface DragState {
  segmentId: string;
  trackId: string;
  initialStartTime: number;
  initialEndTime: number;
  dragStartX: number;
  currentOffset: number;
}

// Trim state interface for segment edge resizing
interface TrimState {
  segmentId: string;
  trackId: string;
  edge: "left" | "right";
  initialStartTime: number;
  initialEndTime: number;
  trimStartX: number;
  currentTimeDelta: number;
}

// Stretch state interface for segment time-stretching (Alt+drag)
interface StretchState {
  segmentId: string;
  trackId: string;
  edge: "left" | "right";
  initialStartTime: number;
  initialEndTime: number;
  initialSpeedFactor: number;
  stretchStartX: number;
  currentTimeDelta: number;
  currentSpeedFactor: number;
  audioUrl?: string;
}

// Minimum segment duration in seconds
const MIN_SEGMENT_DURATION = 0.1;

interface TimelineProps {
  tracks: TimelineTrack[];
  duration: number;
  currentTime: number;
  zoom: number;
  onSeek: (time: number) => void;
  onZoomChange: (zoom: number) => void;
  onSegmentSelect?: (segment: TimelineSegment | null) => void;
  onSegmentDrop?: (segmentId: string, startTime: number, endTime: number) => void;
  onSegmentTrim?: (segmentId: string, startTime: number, endTime: number) => void;
  onSegmentStretch?: (segmentId: string, startTime: number, endTime: number, speedFactor: number) => void;
  selectedSegmentId?: string | null;
}

// Color mapping for track types
const TRACK_COLORS: Record<string, { bg: string; border: string; hover: string }> = {
  vocals: {
    bg: "bg-blue-500/20",
    border: "border-blue-500/40",
    hover: "hover:bg-blue-500/30",
  },
  background: {
    bg: "bg-green-500/20",
    border: "border-green-500/40",
    hover: "hover:bg-green-500/30",
  },
  dubbed: {
    bg: "bg-purple-500/20",
    border: "border-purple-500/40",
    hover: "hover:bg-purple-500/30",
  },
  video: {
    bg: "bg-amber-500/20",
    border: "border-amber-500/40",
    hover: "hover:bg-amber-500/30",
  },
};

// Pixels per second at zoom level 1
const BASE_PIXELS_PER_SECOND = 50;

// Convert project tracks to Twick TrackJSON format
function convertToTwickTracks(tracks: TimelineTrack[]): TrackJSON[] {
  return tracks.map((track) => ({
    id: track.id,
    name: track.name,
    type: track.type,
    props: {
      muted: track.muted,
      solo: track.solo,
      volume: track.volume,
    },
    elements: track.segments.map((segment) => ({
      id: segment.id,
      type: "audio",
      s: segment.start_time,
      e: segment.end_time,
      t: segment.translated_text || segment.original_text || segment.speaker || "",
      props: {
        src: segment.audio_url || "",
        speaker: segment.speaker,
        original_text: segment.original_text,
        translated_text: segment.translated_text,
        speed_factor: segment.speed_factor,
      },
    })),
  }));
}

// Inner timeline component that uses the Twick context
function TimelineInner({
  tracks,
  duration,
  currentTime,
  zoom,
  onSeek,
  onZoomChange,
  onSegmentSelect,
  onSegmentDrop,
  onSegmentTrim,
  onSegmentStretch,
  selectedSegmentId,
}: TimelineProps) {
  const timelineRef = useRef<HTMLDivElement>(null);
  const tracksContainerRef = useRef<HTMLDivElement>(null);
  // Access Twick context - available for future operations (undo/redo, element manipulation)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { editor } = useTimelineContext();

  const pixelsPerSecond = BASE_PIXELS_PER_SECOND * zoom;
  const timelineWidth = Math.max(duration * pixelsPerSecond, 800);

  // Drag state for segment repositioning
  const [dragState, setDragState] = useState<DragState | null>(null);
  const isDragging = dragState !== null;

  // Trim state for segment edge resizing
  const [trimState, setTrimState] = useState<TrimState | null>(null);
  const isTrimming = trimState !== null;

  // Stretch state for segment time-stretching (Alt+drag)
  const [stretchState, setStretchState] = useState<StretchState | null>(null);
  const isStretching = stretchState !== null;

  // Handle scroll wheel zoom
  const handleWheel = useCallback(
    (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const newZoom = Math.max(0.25, Math.min(10, zoom * delta));
        onZoomChange(newZoom);
      }
    },
    [zoom, onZoomChange]
  );

  // Attach wheel listener
  useEffect(() => {
    const container = timelineRef.current;
    if (container) {
      container.addEventListener("wheel", handleWheel, { passive: false });
      return () => container.removeEventListener("wheel", handleWheel);
    }
  }, [handleWheel]);

  // Handle segment drag start
  const handleSegmentDragStart = useCallback(
    (e: React.MouseEvent, segment: TimelineSegment) => {
      e.stopPropagation();
      e.preventDefault();

      setDragState({
        segmentId: segment.id,
        trackId: segment.track_id,
        initialStartTime: segment.start_time,
        initialEndTime: segment.end_time,
        dragStartX: e.clientX,
        currentOffset: 0,
      });

      // Select the segment being dragged
      onSegmentSelect?.(segment);
    },
    [onSegmentSelect]
  );

  // Handle mouse move during drag
  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!dragState) return;

      const deltaX = e.clientX - dragState.dragStartX;
      const timeOffset = deltaX / pixelsPerSecond;

      // Calculate new start time, ensuring it doesn't go below 0
      const newStartTime = Math.max(0, dragState.initialStartTime + timeOffset);
      const segmentDuration = dragState.initialEndTime - dragState.initialStartTime;
      const newEndTime = newStartTime + segmentDuration;

      // Don't allow dragging beyond duration
      if (newEndTime > duration) {
        const adjustedStartTime = duration - segmentDuration;
        setDragState((prev) =>
          prev
            ? {
                ...prev,
                currentOffset: adjustedStartTime - prev.initialStartTime,
              }
            : null
        );
      } else {
        setDragState((prev) =>
          prev
            ? {
                ...prev,
                currentOffset: newStartTime - prev.initialStartTime,
              }
            : null
        );
      }
    },
    [dragState, pixelsPerSecond, duration]
  );

  // Handle mouse up to finish drag
  const handleMouseUp = useCallback(() => {
    if (!dragState) return;

    const segmentDuration = dragState.initialEndTime - dragState.initialStartTime;
    let newStartTime = dragState.initialStartTime + dragState.currentOffset;

    // Clamp values
    newStartTime = Math.max(0, newStartTime);
    const newEndTime = Math.min(duration, newStartTime + segmentDuration);
    newStartTime = newEndTime - segmentDuration;

    // Only trigger callback if position actually changed
    if (Math.abs(dragState.currentOffset) > 0.01) {
      onSegmentDrop?.(dragState.segmentId, newStartTime, newEndTime);
    }

    setDragState(null);
  }, [dragState, duration, onSegmentDrop]);

  // Attach drag event listeners
  useEffect(() => {
    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);

      return () => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Handle trim start (edge resize)
  const handleTrimStart = useCallback(
    (e: React.MouseEvent, segment: TimelineSegment, edge: "left" | "right") => {
      e.stopPropagation();
      e.preventDefault();

      setTrimState({
        segmentId: segment.id,
        trackId: segment.track_id,
        edge,
        initialStartTime: segment.start_time,
        initialEndTime: segment.end_time,
        trimStartX: e.clientX,
        currentTimeDelta: 0,
      });

      // Select the segment being trimmed
      onSegmentSelect?.(segment);
    },
    [onSegmentSelect]
  );

  // Handle mouse move during trim
  const handleTrimMove = useCallback(
    (e: MouseEvent) => {
      if (!trimState) return;

      const deltaX = e.clientX - trimState.trimStartX;
      const timeDelta = deltaX / pixelsPerSecond;

      // Calculate new times based on which edge is being dragged
      let newStartTime = trimState.initialStartTime;
      let newEndTime = trimState.initialEndTime;

      if (trimState.edge === "left") {
        // Dragging left edge changes start_time
        newStartTime = trimState.initialStartTime + timeDelta;
        // Clamp: can't go below 0 and can't make segment shorter than MIN_SEGMENT_DURATION
        newStartTime = Math.max(0, newStartTime);
        newStartTime = Math.min(newStartTime, trimState.initialEndTime - MIN_SEGMENT_DURATION);
      } else {
        // Dragging right edge changes end_time
        newEndTime = trimState.initialEndTime + timeDelta;
        // Clamp: can't exceed duration and can't make segment shorter than MIN_SEGMENT_DURATION
        newEndTime = Math.min(duration, newEndTime);
        newEndTime = Math.max(newEndTime, trimState.initialStartTime + MIN_SEGMENT_DURATION);
      }

      // Calculate actual delta from initial values
      const actualDelta = trimState.edge === "left"
        ? newStartTime - trimState.initialStartTime
        : newEndTime - trimState.initialEndTime;

      setTrimState((prev) =>
        prev ? { ...prev, currentTimeDelta: actualDelta } : null
      );
    },
    [trimState, pixelsPerSecond, duration]
  );

  // Handle mouse up to finish trim
  const handleTrimEnd = useCallback(() => {
    if (!trimState) return;

    let newStartTime = trimState.initialStartTime;
    let newEndTime = trimState.initialEndTime;

    if (trimState.edge === "left") {
      newStartTime = trimState.initialStartTime + trimState.currentTimeDelta;
      // Final clamp
      newStartTime = Math.max(0, newStartTime);
      newStartTime = Math.min(newStartTime, trimState.initialEndTime - MIN_SEGMENT_DURATION);
    } else {
      newEndTime = trimState.initialEndTime + trimState.currentTimeDelta;
      // Final clamp
      newEndTime = Math.min(duration, newEndTime);
      newEndTime = Math.max(newEndTime, trimState.initialStartTime + MIN_SEGMENT_DURATION);
    }

    // Only trigger callback if timing actually changed
    if (Math.abs(trimState.currentTimeDelta) > 0.01) {
      // Use onSegmentTrim if provided, otherwise fall back to onSegmentDrop
      if (onSegmentTrim) {
        onSegmentTrim(trimState.segmentId, newStartTime, newEndTime);
      } else if (onSegmentDrop) {
        onSegmentDrop(trimState.segmentId, newStartTime, newEndTime);
      }
    }

    setTrimState(null);
  }, [trimState, duration, onSegmentTrim, onSegmentDrop]);

  // Attach trim event listeners
  useEffect(() => {
    if (isTrimming) {
      window.addEventListener("mousemove", handleTrimMove);
      window.addEventListener("mouseup", handleTrimEnd);

      return () => {
        window.removeEventListener("mousemove", handleTrimMove);
        window.removeEventListener("mouseup", handleTrimEnd);
      };
    }
  }, [isTrimming, handleTrimMove, handleTrimEnd]);

  // Handle stretch start (Alt+drag on edge)
  const handleStretchStart = useCallback(
    (e: React.MouseEvent, segment: TimelineSegment, edge: "left" | "right") => {
      e.stopPropagation();
      e.preventDefault();

      setStretchState({
        segmentId: segment.id,
        trackId: segment.track_id,
        edge,
        initialStartTime: segment.start_time,
        initialEndTime: segment.end_time,
        initialSpeedFactor: segment.speed_factor,
        stretchStartX: e.clientX,
        currentTimeDelta: 0,
        currentSpeedFactor: segment.speed_factor,
        audioUrl: segment.audio_url,
      });

      // Select the segment being stretched
      onSegmentSelect?.(segment);
    },
    [onSegmentSelect]
  );

  // Handle mouse move during stretch
  const handleStretchMove = useCallback(
    (e: MouseEvent) => {
      if (!stretchState) return;

      const deltaX = e.clientX - stretchState.stretchStartX;
      const timeDelta = deltaX / pixelsPerSecond;

      // Calculate new times based on which edge is being dragged
      let newStartTime = stretchState.initialStartTime;
      let newEndTime = stretchState.initialEndTime;

      if (stretchState.edge === "left") {
        // Dragging left edge: changes start time and duration
        newStartTime = stretchState.initialStartTime + timeDelta;
        // Clamp: can't go below 0 and can't make segment shorter than MIN_SEGMENT_DURATION
        newStartTime = Math.max(0, newStartTime);
        newStartTime = Math.min(newStartTime, stretchState.initialEndTime - MIN_SEGMENT_DURATION);
      } else {
        // Dragging right edge: changes end time and duration
        newEndTime = stretchState.initialEndTime + timeDelta;
        // Clamp: can't exceed duration and can't make segment shorter than MIN_SEGMENT_DURATION
        newEndTime = Math.min(duration, newEndTime);
        newEndTime = Math.max(newEndTime, stretchState.initialStartTime + MIN_SEGMENT_DURATION);
      }

      // Calculate new duration and speed factor
      const originalDuration = stretchState.initialEndTime - stretchState.initialStartTime;
      const newDuration = newEndTime - newStartTime;

      // Calculate new speed factor
      const newSpeedFactor = calculateSpeedFactor(
        originalDuration,
        newDuration,
        stretchState.initialSpeedFactor
      );

      // Enforce speed factor limits by adjusting the duration
      let clampedSpeedFactor = newSpeedFactor;
      let adjustedStartTime = newStartTime;
      let adjustedEndTime = newEndTime;

      if (newSpeedFactor < MIN_SPEED_FACTOR || newSpeedFactor > MAX_SPEED_FACTOR) {
        clampedSpeedFactor = Math.max(MIN_SPEED_FACTOR, Math.min(MAX_SPEED_FACTOR, newSpeedFactor));
        // Recalculate the allowed duration based on clamped speed
        const originalAudioDuration = originalDuration * stretchState.initialSpeedFactor;
        const allowedDuration = originalAudioDuration / clampedSpeedFactor;

        if (stretchState.edge === "left") {
          adjustedStartTime = stretchState.initialEndTime - allowedDuration;
          adjustedStartTime = Math.max(0, adjustedStartTime);
        } else {
          adjustedEndTime = stretchState.initialStartTime + allowedDuration;
          adjustedEndTime = Math.min(duration, adjustedEndTime);
        }
      }

      // Calculate actual delta from initial values
      const actualDelta = stretchState.edge === "left"
        ? adjustedStartTime - stretchState.initialStartTime
        : adjustedEndTime - stretchState.initialEndTime;

      setStretchState((prev) =>
        prev ? { ...prev, currentTimeDelta: actualDelta, currentSpeedFactor: clampedSpeedFactor } : null
      );
    },
    [stretchState, pixelsPerSecond, duration]
  );

  // Handle mouse up to finish stretch
  const handleStretchEnd = useCallback(() => {
    if (!stretchState) return;

    // Stop audio preview
    stopPreview();

    let newStartTime = stretchState.initialStartTime;
    let newEndTime = stretchState.initialEndTime;

    if (stretchState.edge === "left") {
      newStartTime = stretchState.initialStartTime + stretchState.currentTimeDelta;
      // Final clamp
      newStartTime = Math.max(0, newStartTime);
      newStartTime = Math.min(newStartTime, stretchState.initialEndTime - MIN_SEGMENT_DURATION);
    } else {
      newEndTime = stretchState.initialEndTime + stretchState.currentTimeDelta;
      // Final clamp
      newEndTime = Math.min(duration, newEndTime);
      newEndTime = Math.max(newEndTime, stretchState.initialStartTime + MIN_SEGMENT_DURATION);
    }

    // Only trigger callback if timing actually changed
    if (Math.abs(stretchState.currentTimeDelta) > 0.01) {
      if (onSegmentStretch) {
        onSegmentStretch(
          stretchState.segmentId,
          newStartTime,
          newEndTime,
          stretchState.currentSpeedFactor
        );
      }
    }

    setStretchState(null);
  }, [stretchState, duration, onSegmentStretch]);

  // Preview audio during stretch (debounced)
  const previewTimeoutRef = useRef<number | null>(null);
  const lastPreviewSpeedRef = useRef<number | null>(null);

  useEffect(() => {
    // Only preview if we have stretch state with audio
    if (!stretchState || !stretchState.audioUrl) {
      return;
    }

    // Only trigger preview if speed actually changed significantly
    const speedDelta = lastPreviewSpeedRef.current !== null
      ? Math.abs(stretchState.currentSpeedFactor - lastPreviewSpeedRef.current)
      : 1;

    if (speedDelta < 0.05) {
      return; // Skip preview for tiny changes
    }

    // Clear any pending preview
    if (previewTimeoutRef.current) {
      window.clearTimeout(previewTimeoutRef.current);
    }

    // Debounce the preview to avoid too many audio loads
    const audioUrl = stretchState.audioUrl;
    const speedFactor = stretchState.currentSpeedFactor;

    previewTimeoutRef.current = window.setTimeout(() => {
      lastPreviewSpeedRef.current = speedFactor;
      previewAudioAtSpeed(audioUrl, speedFactor, 0, 0.5).catch(console.error);
    }, 150);

    return () => {
      if (previewTimeoutRef.current) {
        window.clearTimeout(previewTimeoutRef.current);
      }
    };
  }, [stretchState]);

  // Attach stretch event listeners
  useEffect(() => {
    if (isStretching) {
      window.addEventListener("mousemove", handleStretchMove);
      window.addEventListener("mouseup", handleStretchEnd);

      return () => {
        window.removeEventListener("mousemove", handleStretchMove);
        window.removeEventListener("mouseup", handleStretchEnd);
        stopPreview();
      };
    }
  }, [isStretching, handleStretchMove, handleStretchEnd]);

  // Handle click on timeline to seek
  const handleTimelineClick = (e: React.MouseEvent) => {
    if (tracksContainerRef.current) {
      const rect = tracksContainerRef.current.getBoundingClientRect();
      const scrollLeft = tracksContainerRef.current.scrollLeft;
      const x = e.clientX - rect.left + scrollLeft;
      const time = x / pixelsPerSecond;
      onSeek(Math.max(0, Math.min(time, duration)));
    }
  };

  // Handle segment click
  const handleSegmentClick = (
    e: React.MouseEvent,
    segment: TimelineSegment
  ) => {
    e.stopPropagation();
    onSegmentSelect?.(segment);
  };

  // Calculate tick marks for time ruler
  const tickInterval = useMemo(() => {
    if (zoom < 0.5) return 10;
    if (zoom < 1) return 5;
    if (zoom < 2) return 2;
    return 1;
  }, [zoom]);

  const ticks = useMemo(() => {
    const result: number[] = [];
    for (let t = 0; t <= duration; t += tickInterval) {
      result.push(t);
    }
    return result;
  }, [duration, tickInterval]);

  // Format time as MM:SS
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div
      ref={timelineRef}
      className="flex h-full flex-col overflow-hidden"
    >
      {/* Single scroll container for ruler + tracks */}
      <div
        ref={tracksContainerRef}
        className="flex-1 overflow-auto bg-muted/20"
        onClick={handleTimelineClick}
      >
        <div
          className="relative"
          style={{
            width: `${timelineWidth}px`,
            minHeight: `${tracks.length * 80 + 32}px`,
          }}
        >
          {/* Time Ruler - sticky at top */}
          <div className="sticky top-0 z-30 h-8 border-b bg-muted/50 cursor-pointer">
            {ticks.map((time) => (
              <div
                key={time}
                className="absolute flex flex-col items-center"
                style={{ left: `${time * pixelsPerSecond}px` }}
              >
                <div className="h-3 w-px bg-border" />
                <span className="text-[10px] text-muted-foreground">
                  {formatTime(time)}
                </span>
              </div>
            ))}
            {/* Playhead on ruler */}
            <div
              className="absolute top-0 h-full w-0.5 bg-red-500"
              style={{ left: `${currentTime * pixelsPerSecond}px` }}
            />
          </div>

          {/* Track lanes */}
          <div
            className="relative"
            style={{ minHeight: `${tracks.length * 80}px` }}
          >
            {/* Playhead */}
            <div
              className="absolute top-0 bottom-0 z-20 w-0.5 bg-red-500 pointer-events-none"
              style={{ left: `${currentTime * pixelsPerSecond}px` }}
            />

          {/* Track lanes */}
          {tracks.map((track, index) => {
            const hasSoloTracks = tracks.some((t) => t.solo);
            const isEffectivelyMuted =
              track.muted || (hasSoloTracks && !track.solo);
            const colors = TRACK_COLORS[track.type] || TRACK_COLORS.dubbed;

            return (
              <div
                key={track.id}
                className={cn(
                  "absolute left-0 right-0 border-b border-border/50",
                  isEffectivelyMuted && "opacity-50"
                )}
                style={{
                  top: `${index * 80}px`,
                  height: "80px",
                }}
              >
                {/* Segments */}
                {track.segments.map((segment) => {
                  const isSelected = selectedSegmentId === segment.id;
                  const isBeingDragged = dragState?.segmentId === segment.id;
                  const isBeingTrimmed = trimState?.segmentId === segment.id;
                  const isBeingStretched = stretchState?.segmentId === segment.id;
                  const trimEdge = isBeingTrimmed ? trimState.edge : null;
                  const stretchEdge = isBeingStretched ? stretchState.edge : null;

                  // Calculate segment dimensions considering trim or stretch state
                  let effectiveStartTime = segment.start_time;
                  let effectiveEndTime = segment.end_time;
                  let displaySpeedFactor = segment.speed_factor;

                  if (isBeingTrimmed) {
                    if (trimState.edge === "left") {
                      effectiveStartTime = Math.max(0, segment.start_time + trimState.currentTimeDelta);
                      effectiveStartTime = Math.min(effectiveStartTime, segment.end_time - MIN_SEGMENT_DURATION);
                    } else {
                      effectiveEndTime = Math.min(duration, segment.end_time + trimState.currentTimeDelta);
                      effectiveEndTime = Math.max(effectiveEndTime, segment.start_time + MIN_SEGMENT_DURATION);
                    }
                  } else if (isBeingStretched) {
                    if (stretchState.edge === "left") {
                      effectiveStartTime = Math.max(0, segment.start_time + stretchState.currentTimeDelta);
                      effectiveStartTime = Math.min(effectiveStartTime, segment.end_time - MIN_SEGMENT_DURATION);
                    } else {
                      effectiveEndTime = Math.min(duration, segment.end_time + stretchState.currentTimeDelta);
                      effectiveEndTime = Math.max(effectiveEndTime, segment.start_time + MIN_SEGMENT_DURATION);
                    }
                    displaySpeedFactor = stretchState.currentSpeedFactor;
                  }

                  const segmentWidth = (effectiveEndTime - effectiveStartTime) * pixelsPerSecond;

                  // Calculate position with drag offset if this segment is being dragged
                  const dragOffset = isBeingDragged
                    ? dragState.currentOffset * pixelsPerSecond
                    : 0;
                  const left = effectiveStartTime * pixelsPerSecond + dragOffset;
                  const displayWidth = Math.max(segmentWidth, 20);

                  // Determine cursor based on current interaction
                  const getCursor = () => {
                    if (isBeingDragged) return "cursor-grabbing";
                    if (isBeingTrimmed || isBeingStretched) return "cursor-ew-resize";
                    return "cursor-grab";
                  };

                  return (
                    <div
                      key={segment.id}
                      className={cn(
                        "absolute top-2 h-[64px] rounded-md border transition-colors group",
                        "flex flex-col overflow-hidden select-none",
                        colors.bg,
                        colors.border,
                        colors.hover,
                        isSelected && "ring-2 ring-primary ring-offset-1",
                        isBeingDragged && "opacity-80 shadow-lg z-30",
                        isBeingTrimmed && "opacity-90 shadow-md z-30",
                        isBeingStretched && "opacity-90 shadow-md z-30 ring-2 ring-orange-500/50",
                        getCursor()
                      )}
                      style={{
                        left: `${left}px`,
                        width: `${displayWidth}px`,
                        transition: (isBeingDragged || isBeingTrimmed || isBeingStretched) ? "none" : "box-shadow 150ms, opacity 150ms",
                      }}
                      onClick={(e) => handleSegmentClick(e, segment)}
                      onMouseDown={(e) => {
                        // Check if click is near edges (within 8px) for trim/stretch
                        const rect = e.currentTarget.getBoundingClientRect();
                        const relativeX = e.clientX - rect.left;
                        const edgeZone = 8; // pixels

                        if (relativeX <= edgeZone) {
                          // Alt+drag on edge = stretch, regular drag on edge = trim
                          if (e.altKey) {
                            handleStretchStart(e, segment, "left");
                          } else {
                            handleTrimStart(e, segment, "left");
                          }
                        } else if (relativeX >= rect.width - edgeZone) {
                          // Alt+drag on edge = stretch, regular drag on edge = trim
                          if (e.altKey) {
                            handleStretchStart(e, segment, "right");
                          } else {
                            handleTrimStart(e, segment, "right");
                          }
                        } else {
                          handleSegmentDragStart(e, segment);
                        }
                      }}
                      title={
                        segment.translated_text ||
                        segment.original_text ||
                        segment.speaker
                      }
                    >
                      {/* Left resize handle */}
                      <div
                        className={cn(
                          "absolute left-0 top-0 bottom-0 w-2 cursor-ew-resize",
                          "opacity-0 group-hover:opacity-100 transition-opacity",
                          "bg-primary/30 hover:bg-primary/50",
                          "flex items-center justify-center",
                          (trimEdge === "left" || stretchEdge === "left") && "opacity-100",
                          trimEdge === "left" && "bg-primary/60",
                          stretchEdge === "left" && "bg-orange-500/60"
                        )}
                        onMouseDown={(e) => {
                          e.stopPropagation();
                          if (e.altKey) {
                            handleStretchStart(e, segment, "left");
                          } else {
                            handleTrimStart(e, segment, "left");
                          }
                        }}
                        title="Drag to trim, Alt+Drag to stretch"
                      >
                        <div className="w-0.5 h-6 bg-primary/80 rounded-full" />
                      </div>

                      {/* Right resize handle */}
                      <div
                        className={cn(
                          "absolute right-0 top-0 bottom-0 w-2 cursor-ew-resize",
                          "opacity-0 group-hover:opacity-100 transition-opacity",
                          "bg-primary/30 hover:bg-primary/50",
                          "flex items-center justify-center",
                          (trimEdge === "right" || stretchEdge === "right") && "opacity-100",
                          trimEdge === "right" && "bg-primary/60",
                          stretchEdge === "right" && "bg-orange-500/60"
                        )}
                        onMouseDown={(e) => {
                          e.stopPropagation();
                          if (e.altKey) {
                            handleStretchStart(e, segment, "right");
                          } else {
                            handleTrimStart(e, segment, "right");
                          }
                        }}
                        title="Drag to trim, Alt+Drag to stretch"
                      >
                        <div className="w-0.5 h-6 bg-primary/80 rounded-full" />
                      </div>

                      {/* Waveform visualization (background layer) */}
                      {segment.audio_url && displayWidth > 30 && (
                        <div className="absolute inset-0 opacity-40 flex items-center justify-center">
                          <SegmentWaveform
                            audioUrl={segment.audio_url}
                            width={displayWidth - 4}
                            height={56}
                            color={
                              track.type === "vocals"
                                ? "#3b82f6"
                                : track.type === "background"
                                ? "#22c55e"
                                : "#a855f7"
                            }
                            zoom={zoom}
                            showLoading={false}
                          />
                        </div>
                      )}

                      {/* Content overlay */}
                      <div className="relative z-10 flex flex-col justify-center h-full px-3">
                        {/* Speaker badge */}
                        {segment.speaker && (
                          <span className="mb-1 truncate text-[10px] font-medium text-muted-foreground">
                            {segment.speaker}
                          </span>
                        )}
                        {/* Text content */}
                        <span className="truncate text-xs">
                          {segment.translated_text ||
                            segment.original_text ||
                            "\u00A0"}
                        </span>
                        {/* Speed factor indicator */}
                        {(displaySpeedFactor !== 1.0 || isBeingStretched) && (
                          <span className={cn(
                            "mt-1 text-[10px]",
                            isBeingStretched ? "text-orange-500 font-medium" : "text-muted-foreground"
                          )}>
                            {displaySpeedFactor.toFixed(2)}x
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })}

            {/* Empty state */}
            {tracks.length === 0 && (
              <div className="flex h-full min-h-[200px] items-center justify-center text-muted-foreground">
                No tracks to display
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Main Timeline component with Twick provider
export function Timeline(props: TimelineProps) {
  const initialData = useMemo(
    () => ({
      tracks: convertToTwickTracks(props.tracks),
      version: 1,
    }),
    [props.tracks]
  );

  return (
    <TimelineProvider
      contextId="dubbing-timeline"
      initialData={initialData}
      analytics={{ enabled: false }}
    >
      <TimelineInner {...props} />
    </TimelineProvider>
  );
}

export default Timeline;
