"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { StatusBadge } from "@/components/StatusBadge";
import { cn } from "@/lib/utils";
import {
  ArrowLeft,
  Play,
  Pause,
  ZoomIn,
  ZoomOut,
  Download,
  Volume2,
  VolumeX,
  RefreshCw,
  AlertCircle,
} from "lucide-react";

// Types
interface Track {
  id: string;
  name: string;
  type: "vocals" | "background" | "dubbed";
  muted: boolean;
  solo: boolean;
  volume: number;
  segments: Segment[];
}

interface Segment {
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

interface Project {
  id: string;
  name: string;
  status: string;
  source_language: string;
  target_language: string;
  video_url?: string;
  duration?: number;
  tracks: Track[];
}

interface EditorPageProps {
  params: Promise<{ id: string }>;
}

// Helper functions
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins}:${secs.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`;
}

export default function EditorPage({ params }: EditorPageProps) {
  const [projectId, setProjectId] = useState<string | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Timeline state
  const [zoom, setZoom] = useState(1);
  const [tracks, setTracks] = useState<Track[]>([]);

  // Export state
  const [isExporting, setIsExporting] = useState(false);

  // Resolve params promise
  useEffect(() => {
    params.then((p) => setProjectId(p.id));
  }, [params]);

  // Fetch project data
  const fetchProject = useCallback(async () => {
    if (!projectId) return;

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      const response = await fetch(`${apiUrl}/api/projects/${projectId}`);

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error("Project not found");
        }
        throw new Error(`Failed to fetch project: ${response.statusText}`);
      }

      const data = await response.json();
      setProject(data);
      setTracks(data.tracks || []);
      if (data.duration) {
        setDuration(data.duration);
      }
      setError(null);
    } catch (err) {
      console.error("Failed to fetch project:", err);
      setError(err instanceof Error ? err.message : "Failed to fetch project");
    } finally {
      setIsLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    if (projectId) {
      fetchProject();
    }
  }, [projectId, fetchProject]);

  // Video sync
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleSeek = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  // Zoom controls
  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev * 1.5, 10));
  };

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(prev / 1.5, 0.25));
  };

  // Track controls
  const handleMuteTrack = (trackId: string) => {
    setTracks((prev) =>
      prev.map((track) =>
        track.id === trackId ? { ...track, muted: !track.muted } : track
      )
    );
  };

  const handleSoloTrack = (trackId: string) => {
    setTracks((prev) =>
      prev.map((track) =>
        track.id === trackId ? { ...track, solo: !track.solo } : track
      )
    );
  };

  const handleVolumeChange = (trackId: string, volume: number) => {
    setTracks((prev) =>
      prev.map((track) =>
        track.id === trackId ? { ...track, volume } : track
      )
    );
  };

  // Export handler
  const handleExport = async () => {
    if (!projectId) return;

    setIsExporting(true);
    setError(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      const response = await fetch(`${apiUrl}/api/projects/${projectId}/export`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.detail || `Export failed: ${response.statusText}`
        );
      }

      const data = await response.json();
      if (data.export_url) {
        window.open(data.export_url, "_blank");
      }
    } catch (err) {
      console.error("Export failed:", err);
      setError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setIsExporting(false);
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="flex h-screen flex-col bg-background">
        <div className="flex h-12 items-center justify-center border-b">
          <div className="h-4 w-48 animate-pulse rounded bg-muted" />
        </div>
        <div className="flex flex-1">
          <div className="w-64 animate-pulse bg-muted" />
          <div className="flex flex-1 flex-col">
            <div className="flex-1 animate-pulse bg-muted/50" />
            <div className="h-1/2 animate-pulse border-t bg-muted/30" />
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error && !project) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <AlertCircle className="h-12 w-12 text-destructive" />
          <p className="text-lg font-medium">Error Loading Editor</p>
          <p className="text-sm text-muted-foreground">{error}</p>
          <Link href="/">
            <Button>Back to Projects</Button>
          </Link>
        </div>
      </div>
    );
  }

  if (!project) return null;

  return (
    <TooltipProvider>
      <div className="flex h-screen flex-col bg-background">
        {/* Top Toolbar */}
        <header className="flex h-12 items-center justify-between border-b px-4">
          <div className="flex items-center gap-4">
            <Link href={`/projects/${projectId}`}>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <ArrowLeft className="h-4 w-4" />
              </Button>
            </Link>
            <div className="flex items-center gap-2">
              <span className="font-medium">{project.name}</span>
              <StatusBadge status={project.status as "ready" | "pending" | "processing" | "error" | "exported" | "exporting"} />
            </div>
          </div>

          {/* Playback controls */}
          <div className="flex items-center gap-4">
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={handlePlayPause}
            >
              {isPlaying ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
            </Button>
            <span className="min-w-[80px] font-mono text-sm">
              {formatTime(currentTime)}
            </span>
            <span className="text-muted-foreground">/</span>
            <span className="min-w-[80px] font-mono text-sm text-muted-foreground">
              {formatTime(duration)}
            </span>
          </div>

          {/* Zoom and export controls */}
          <div className="flex items-center gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  className="h-8 w-8"
                  onClick={handleZoomOut}
                >
                  <ZoomOut className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Zoom Out</TooltipContent>
            </Tooltip>

            <span className="min-w-[40px] text-center text-sm text-muted-foreground">
              {Math.round(zoom * 100)}%
            </span>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  className="h-8 w-8"
                  onClick={handleZoomIn}
                >
                  <ZoomIn className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Zoom In</TooltipContent>
            </Tooltip>

            <div className="mx-2 h-6 w-px bg-border" />

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="default"
                  size="sm"
                  onClick={handleExport}
                  disabled={isExporting}
                >
                  {isExporting ? (
                    <>
                      <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                      Exporting...
                    </>
                  ) : (
                    <>
                      <Download className="mr-2 h-4 w-4" />
                      Export
                    </>
                  )}
                </Button>
              </TooltipTrigger>
              <TooltipContent>Export final video</TooltipContent>
            </Tooltip>
          </div>
        </header>

        {/* Main Content Area */}
        <div className="flex flex-1 overflow-hidden">
          {/* Left Sidebar - Track List */}
          <aside className="flex w-64 flex-col border-r bg-muted/30">
            <div className="border-b p-3">
              <h3 className="text-sm font-medium">Tracks</h3>
            </div>
            <div className="flex-1 overflow-y-auto p-2">
              {tracks.length === 0 ? (
                <p className="p-2 text-sm text-muted-foreground">
                  No tracks available
                </p>
              ) : (
                tracks.map((track) => {
                  const hasSoloTracks = tracks.some((t) => t.solo);
                  const isEffectivelyMuted =
                    track.muted || (hasSoloTracks && !track.solo);

                  return (
                    <div
                      key={track.id}
                      className={cn(
                        "mb-2 rounded-lg border bg-background p-3",
                        isEffectivelyMuted && "opacity-50"
                      )}
                    >
                      <div className="mb-2 flex items-center justify-between">
                        <span className="text-sm font-medium truncate">
                          {track.name}
                        </span>
                        <span className="text-xs text-muted-foreground capitalize">
                          {track.type}
                        </span>
                      </div>

                      {/* Mute/Solo buttons */}
                      <div className="mb-2 flex items-center gap-2">
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant={track.muted ? "default" : "outline"}
                              size="icon"
                              className="h-7 w-7"
                              onClick={() => handleMuteTrack(track.id)}
                            >
                              {track.muted ? (
                                <VolumeX className="h-3 w-3" />
                              ) : (
                                <Volume2 className="h-3 w-3" />
                              )}
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>
                            {track.muted ? "Unmute" : "Mute"}
                          </TooltipContent>
                        </Tooltip>

                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant={track.solo ? "default" : "outline"}
                              size="icon"
                              className={cn(
                                "h-7 w-7 font-bold",
                                track.solo && "bg-yellow-500 hover:bg-yellow-600"
                              )}
                              onClick={() => handleSoloTrack(track.id)}
                            >
                              S
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>
                            {track.solo ? "Unsolo" : "Solo"}
                          </TooltipContent>
                        </Tooltip>
                      </div>

                      {/* Volume slider */}
                      <div className="flex items-center gap-2">
                        <Volume2 className="h-3 w-3 text-muted-foreground" />
                        <Slider
                          value={[track.volume * 100]}
                          min={0}
                          max={200}
                          step={1}
                          className="flex-1"
                          onValueChange={([value]) =>
                            handleVolumeChange(track.id, value / 100)
                          }
                        />
                        <span className="min-w-[36px] text-right text-xs text-muted-foreground">
                          {Math.round(track.volume * 100)}%
                        </span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </aside>

          {/* Main Editor Area */}
          <div className="flex flex-1 flex-col">
            {/* Video Player (50% height) */}
            <div className="flex h-1/2 items-center justify-center border-b bg-black">
              {project.video_url ? (
                <video
                  ref={videoRef}
                  className="h-full w-full object-contain"
                  src={project.video_url}
                  onTimeUpdate={handleTimeUpdate}
                  onLoadedMetadata={handleLoadedMetadata}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => setIsPlaying(false)}
                />
              ) : (
                <div className="text-muted-foreground">No video available</div>
              )}
            </div>

            {/* Timeline Editor (50% height) */}
            <div className="flex h-1/2 flex-col overflow-hidden">
              {/* Time ruler */}
              <div className="flex h-8 shrink-0 items-center border-b bg-muted/50 px-4">
                <TimeRuler
                  duration={duration}
                  zoom={zoom}
                  onSeek={handleSeek}
                />
              </div>

              {/* Timeline tracks */}
              <div className="flex-1 overflow-auto bg-muted/20">
                <div
                  className="relative"
                  style={{
                    minWidth: `${Math.max(duration * 50 * zoom, 100)}px`,
                    minHeight: `${tracks.length * 80}px`,
                  }}
                >
                  {/* Playhead */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-10"
                    style={{
                      left: `${currentTime * 50 * zoom}px`,
                    }}
                  />

                  {/* Track lanes */}
                  {tracks.map((track, index) => (
                    <div
                      key={track.id}
                      className="absolute left-0 right-0 border-b border-border/50"
                      style={{
                        top: `${index * 80}px`,
                        height: "80px",
                      }}
                    >
                      {/* Segments */}
                      {track.segments?.map((segment) => (
                        <div
                          key={segment.id}
                          className={cn(
                            "absolute top-2 h-[64px] rounded-md border cursor-pointer",
                            "bg-primary/20 border-primary/40 hover:bg-primary/30",
                            "flex items-center justify-center overflow-hidden"
                          )}
                          style={{
                            left: `${segment.start_time * 50 * zoom}px`,
                            width: `${(segment.end_time - segment.start_time) * 50 * zoom}px`,
                          }}
                          title={segment.translated_text || segment.original_text}
                        >
                          <span className="truncate px-2 text-xs">
                            {segment.translated_text || segment.original_text || segment.speaker}
                          </span>
                        </div>
                      ))}
                    </div>
                  ))}

                  {/* Empty state */}
                  {tracks.length === 0 && (
                    <div className="flex h-full items-center justify-center text-muted-foreground">
                      No tracks to display
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Error toast */}
        {error && project && (
          <div className="absolute bottom-4 right-4 flex items-center gap-2 rounded-lg border border-destructive bg-destructive/10 px-4 py-2">
            <AlertCircle className="h-4 w-4 text-destructive" />
            <span className="text-sm text-destructive">{error}</span>
          </div>
        )}
      </div>
    </TooltipProvider>
  );
}

// Time Ruler Component
interface TimeRulerProps {
  duration: number;
  zoom: number;
  onSeek: (time: number) => void;
}

function TimeRuler({ duration, zoom, onSeek }: TimeRulerProps) {
  const rulerRef = useRef<HTMLDivElement>(null);

  const handleClick = (e: React.MouseEvent) => {
    if (rulerRef.current) {
      const rect = rulerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const time = x / (50 * zoom);
      onSeek(Math.max(0, Math.min(time, duration)));
    }
  };

  // Calculate tick marks
  const tickInterval = zoom < 0.5 ? 10 : zoom < 1 ? 5 : zoom < 2 ? 2 : 1;
  const ticks: number[] = [];
  for (let t = 0; t <= duration; t += tickInterval) {
    ticks.push(t);
  }

  return (
    <div
      ref={rulerRef}
      className="relative h-full w-full cursor-pointer"
      onClick={handleClick}
      style={{ minWidth: `${duration * 50 * zoom}px` }}
    >
      {ticks.map((time) => (
        <div
          key={time}
          className="absolute flex flex-col items-center"
          style={{ left: `${time * 50 * zoom}px` }}
        >
          <div className="h-3 w-px bg-border" />
          <span className="text-[10px] text-muted-foreground">
            {Math.floor(time / 60)}:{(time % 60).toString().padStart(2, "0")}
          </span>
        </div>
      ))}
    </div>
  );
}
