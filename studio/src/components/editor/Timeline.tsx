"use client";

import { useCallback, useEffect, useRef, useMemo } from "react";
import {
  TimelineProvider,
  useTimelineContext,
  TrackJSON,
} from "@twick/timeline";
import { cn } from "@/lib/utils";
import { SegmentWaveform } from "./SegmentWaveform";

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
  type: "vocals" | "background" | "dubbed";
  muted: boolean;
  solo: boolean;
  volume: number;
  segments: TimelineSegment[];
}

interface TimelineProps {
  tracks: TimelineTrack[];
  duration: number;
  currentTime: number;
  zoom: number;
  onSeek: (time: number) => void;
  onZoomChange: (zoom: number) => void;
  onSegmentSelect?: (segment: TimelineSegment | null) => void;
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
  selectedSegmentId,
}: TimelineProps) {
  const timelineRef = useRef<HTMLDivElement>(null);
  const tracksContainerRef = useRef<HTMLDivElement>(null);
  // Access Twick context - available for future operations (undo/redo, element manipulation)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const { editor } = useTimelineContext();

  const pixelsPerSecond = BASE_PIXELS_PER_SECOND * zoom;
  const timelineWidth = Math.max(duration * pixelsPerSecond, 800);

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
      {/* Time Ruler */}
      <div className="flex h-8 shrink-0 border-b bg-muted/50">
        <div
          className="relative h-full cursor-pointer"
          style={{ width: `${timelineWidth}px` }}
          onClick={handleTimelineClick}
        >
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
      </div>

      {/* Tracks Container */}
      <div
        ref={tracksContainerRef}
        className="flex-1 overflow-auto bg-muted/20"
        onClick={handleTimelineClick}
      >
        <div
          className="relative"
          style={{
            width: `${timelineWidth}px`,
            minHeight: `${tracks.length * 80}px`,
          }}
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
                  const segmentWidth =
                    (segment.end_time - segment.start_time) * pixelsPerSecond;
                  const left = segment.start_time * pixelsPerSecond;
                  const displayWidth = Math.max(segmentWidth, 20);

                  return (
                    <div
                      key={segment.id}
                      className={cn(
                        "absolute top-2 h-[64px] rounded-md border cursor-pointer transition-all",
                        "flex flex-col overflow-hidden",
                        colors.bg,
                        colors.border,
                        colors.hover,
                        isSelected && "ring-2 ring-primary ring-offset-1"
                      )}
                      style={{
                        left: `${left}px`,
                        width: `${displayWidth}px`,
                      }}
                      onClick={(e) => handleSegmentClick(e, segment)}
                      title={
                        segment.translated_text ||
                        segment.original_text ||
                        segment.speaker
                      }
                    >
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
                      <div className="relative z-10 flex flex-col justify-center h-full px-2">
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
                        {segment.speed_factor !== 1.0 && (
                          <span className="mt-1 text-[10px] text-muted-foreground">
                            {segment.speed_factor.toFixed(2)}x
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
