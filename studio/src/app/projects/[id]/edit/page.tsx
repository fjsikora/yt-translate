"use client";

import { useState, useEffect, useCallback, useRef, useMemo } from "react";
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
import { Timeline, TimelineTrack, TimelineSegment } from "@/components/editor/Timeline";
import { VideoPlayer, VideoPlayerRef } from "@/components/editor/VideoPlayer";
import { SegmentDetails } from "@/components/editor/SegmentDetails";
import { ExportDialog } from "@/components/editor/ExportDialog";
import { cn } from "@/lib/utils";
import { supabase } from "@/lib/supabase";
import { useAudioEngine } from "@/lib/use-audio-engine";
import {
  ArrowLeft,
  Play,
  Pause,
  ZoomIn,
  ZoomOut,
  Download,
  Volume2,
  VolumeX,
  AlertCircle,
  Loader2,
  Check,
  HelpCircle,
} from "lucide-react";

// Types
interface Project {
  id: string;
  name: string;
  status: string;
  source_language: string;
  target_language: string;
  video_url?: string;
  duration?: number;
  tracks: TimelineTrack[];
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
  const videoPlayerRef = useRef<VideoPlayerRef>(null);

  // Refs for avoiding stale closures in callbacks
  const currentTimeRef = useRef(0);
  const isPlayingRef = useRef(false);

  // Timeline state
  const [zoom, setZoom] = useState(1);
  const [tracks, setTracks] = useState<TimelineTrack[]>([]);
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [selectedSegment, setSelectedSegment] = useState<TimelineSegment | null>(null);

  // Derive video playback rate from the video track's speed_factor
  const videoSpeedFactor = useMemo(() => {
    const videoTrack = tracks.find((t) => t.type === "video");
    return videoTrack?.segments[0]?.speed_factor ?? 1.0;
  }, [tracks]);

  // Audio engine for multi-track playback
  const audioEngine = useAudioEngine(tracks);

  // Export dialog state
  const [isExportDialogOpen, setIsExportDialogOpen] = useState(false);

  // Autosave state
  type SaveStatus = "idle" | "saving" | "saved" | "error";
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("idle");
  const saveStatusTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pendingSavesRef = useRef<number>(0);

  // Resolve params promise
  useEffect(() => {
    params.then((p) => setProjectId(p.id));
  }, [params]);

  // Fetch project data with tracks and segments from Supabase
  const fetchProject = useCallback(async () => {
    if (!projectId) return;

    try {
      const { data: projectData, error: projectError } = await supabase
        .from("dub_projects")
        .select("*")
        .eq("id", projectId)
        .single();

      if (projectError) {
        if (projectError.code === "PGRST116") {
          throw new Error("Project not found");
        }
        throw new Error(projectError.message);
      }

      // Fetch tracks with their segments
      const { data: tracksData, error: tracksError } = await supabase
        .from("dub_tracks")
        .select("*, segments:dub_segments(*)")
        .eq("project_id", projectId)
        .order("order_index");

      if (tracksError) {
        throw new Error(tracksError.message);
      }

      // Sort segments within each track by start_time
      const tracksWithSortedSegments = (tracksData || []).map((track) => ({
        ...track,
        segments: (track.segments || []).sort(
          (a: TimelineSegment, b: TimelineSegment) =>
            Number(a.start_time) - Number(b.start_time)
        ),
      })) as TimelineTrack[];

      setProject(projectData);
      setTracks(tracksWithSortedSegments);
      if (projectData.duration) {
        setDuration(projectData.duration);
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

  // Video player handlers
  const handleTimeUpdate = useCallback((time: number) => {
    const prev = currentTimeRef.current;
    setCurrentTime(time);
    currentTimeRef.current = time;

    // Detect discontinuous jumps (arrow-key seeks) during playback
    if (isPlayingRef.current && Math.abs(time - prev) > 0.5) {
      audioEngine.seek(time);
    }
  }, [audioEngine]);

  const handleDurationChange = useCallback((newDuration: number) => {
    setDuration(newDuration);
  }, []);

  const handlePlayStateChange = useCallback((playing: boolean) => {
    setIsPlaying(playing);
    isPlayingRef.current = playing;

    if (playing) {
      audioEngine.play(currentTimeRef.current);
    } else {
      audioEngine.pause();
    }
  }, [audioEngine]);

  const handlePlayPause = useCallback(() => {
    videoPlayerRef.current?.togglePlayback();
  }, []);

  const handleSeek = useCallback((time: number) => {
    videoPlayerRef.current?.seek(time);
    setCurrentTime(time);
    currentTimeRef.current = time;
    audioEngine.seek(time);
  }, [audioEngine]);

  // Zoom controls
  const handleZoomIn = () => {
    setZoom((prev) => Math.min(prev * 1.5, 10));
  };

  const handleZoomOut = () => {
    setZoom((prev) => Math.max(prev / 1.5, 0.25));
  };

  const handleZoomChange = useCallback((newZoom: number) => {
    setZoom(newZoom);
  }, []);

  // Helper to manage save status transitions
  const startSave = useCallback(() => {
    pendingSavesRef.current += 1;
    setSaveStatus("saving");
    // Clear any existing timeout that would transition to "idle"
    if (saveStatusTimeoutRef.current) {
      clearTimeout(saveStatusTimeoutRef.current);
      saveStatusTimeoutRef.current = null;
    }
  }, []);

  const completeSave = useCallback((success: boolean) => {
    pendingSavesRef.current = Math.max(0, pendingSavesRef.current - 1);

    // Only update status if no more pending saves
    if (pendingSavesRef.current === 0) {
      if (success) {
        setSaveStatus("saved");
        // Clear existing timeout
        if (saveStatusTimeoutRef.current) {
          clearTimeout(saveStatusTimeoutRef.current);
        }
        // Reset to idle after 2 seconds
        saveStatusTimeoutRef.current = setTimeout(() => {
          setSaveStatus("idle");
        }, 2000);
      } else {
        setSaveStatus("error");
        // Reset to idle after 3 seconds for errors
        if (saveStatusTimeoutRef.current) {
          clearTimeout(saveStatusTimeoutRef.current);
        }
        saveStatusTimeoutRef.current = setTimeout(() => {
          setSaveStatus("idle");
        }, 3000);
      }
    }
  }, []);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (saveStatusTimeoutRef.current) {
        clearTimeout(saveStatusTimeoutRef.current);
      }
    };
  }, []);

  // Save track settings to Supabase
  const saveTrackSettings = useCallback(
    async (trackId: string, settings: { muted?: boolean; solo?: boolean; volume?: number }) => {
      startSave();
      try {
        const { error } = await supabase
          .from("dub_tracks")
          .update(settings)
          .eq("id", trackId);

        if (error) {
          throw new Error(error.message);
        }
        completeSave(true);
      } catch (err) {
        console.error("Failed to save track:", err);
        setError(err instanceof Error ? err.message : "Failed to save track");
        completeSave(false);
      }
    },
    [startSave, completeSave]
  );

  // Track controls
  const handleMuteTrack = useCallback((trackId: string) => {
    setTracks((prev) => {
      const track = prev.find((t) => t.id === trackId);
      if (!track) return prev;

      const newMuted = !track.muted;
      // Save to API asynchronously
      saveTrackSettings(trackId, { muted: newMuted });

      return prev.map((t) =>
        t.id === trackId ? { ...t, muted: newMuted } : t
      );
    });
  }, [saveTrackSettings]);

  const handleSoloTrack = useCallback((trackId: string) => {
    setTracks((prev) => {
      const track = prev.find((t) => t.id === trackId);
      if (!track) return prev;

      const newSolo = !track.solo;
      // Save to API asynchronously
      saveTrackSettings(trackId, { solo: newSolo });

      return prev.map((t) =>
        t.id === trackId ? { ...t, solo: newSolo } : t
      );
    });
  }, [saveTrackSettings]);

  // Volume change with debounced API save
  const volumeTimeoutRef = useRef<Record<string, NodeJS.Timeout>>({});

  const handleVolumeChange = useCallback((trackId: string, volume: number) => {
    // Update local state immediately for responsive UI
    setTracks((prev) =>
      prev.map((track) =>
        track.id === trackId ? { ...track, volume } : track
      )
    );

    // Debounce API save to avoid too many requests during slider drag
    if (volumeTimeoutRef.current[trackId]) {
      clearTimeout(volumeTimeoutRef.current[trackId]);
    }

    volumeTimeoutRef.current[trackId] = setTimeout(() => {
      saveTrackSettings(trackId, { volume });
      delete volumeTimeoutRef.current[trackId];
    }, 300);
  }, [saveTrackSettings]);

  // Sync mute/solo/volume to audio engine when tracks change
  useEffect(() => {
    audioEngine.updateTrackMuteSolo(tracks);
  }, [tracks, audioEngine]);

  // Segment selection
  const handleSegmentSelect = useCallback((segment: TimelineSegment | null) => {
    setSelectedSegmentId(segment?.id ?? null);
    setSelectedSegment(segment);
  }, []);

  // Close segment details panel
  const handleCloseSegmentDetails = useCallback(() => {
    setSelectedSegmentId(null);
    setSelectedSegment(null);
  }, []);

  // Undo/redo history for segment changes
  const historyRef = useRef<{ tracks: TimelineTrack[]; index: number }>({
    tracks: [],
    index: -1,
  });
  const MAX_HISTORY = 50;

  // Push current state to history
  const pushHistory = useCallback(() => {
    const history = historyRef.current;
    const newHistory = history.tracks.slice(0, history.index + 1);
    newHistory.push(JSON.parse(JSON.stringify(tracks)));

    // Trim history if too long
    if (newHistory.length > MAX_HISTORY) {
      newHistory.shift();
    }

    historyRef.current = {
      tracks: newHistory,
      index: newHistory.length - 1,
    };
  }, [tracks]);

  // Initialize history when tracks are first loaded
  useEffect(() => {
    if (tracks.length > 0 && historyRef.current.tracks.length === 0) {
      historyRef.current = {
        tracks: [JSON.parse(JSON.stringify(tracks))],
        index: 0,
      };
    }
  }, [tracks]);

  // Keep selected segment in sync when tracks are updated
  useEffect(() => {
    if (selectedSegmentId && tracks.length > 0) {
      // Find the segment by ID in the updated tracks
      for (const track of tracks) {
        const segment = track.segments.find((s) => s.id === selectedSegmentId);
        if (segment) {
          setSelectedSegment(segment);
          return;
        }
      }
      // Segment not found (possibly deleted), clear selection
      setSelectedSegment(null);
      setSelectedSegmentId(null);
    }
  }, [tracks, selectedSegmentId]);

  // Save segment timing to Supabase (shared by drag-drop and trim operations)
  const saveSegmentTiming = useCallback(
    async (segmentId: string, startTime: number, endTime: number, speedFactor?: number) => {
      startSave();
      try {
        const updates: Record<string, number> = {
          start_time: startTime,
          end_time: endTime,
        };
        if (speedFactor !== undefined) {
          updates.speed_factor = speedFactor;
        }

        const { error } = await supabase
          .from("dub_segments")
          .update(updates)
          .eq("id", segmentId);

        if (error) {
          throw new Error(error.message);
        }
        completeSave(true);
      } catch (err) {
        console.error("Failed to save segment:", err);
        setError(err instanceof Error ? err.message : "Failed to save segment");
        completeSave(false);
      }
    },
    [startSave, completeSave]
  );

  // Handle segment drop (after drag-and-drop repositioning)
  const handleSegmentDrop = useCallback(
    async (segmentId: string, startTime: number, endTime: number) => {
      // Push current state to history before making changes
      pushHistory();

      // Update local state immediately for responsive UI
      setTracks((prev) =>
        prev.map((track) => ({
          ...track,
          segments: track.segments.map((segment) =>
            segment.id === segmentId
              ? { ...segment, start_time: startTime, end_time: endTime }
              : segment
          ),
        }))
      );

      // Save to API
      await saveSegmentTiming(segmentId, startTime, endTime);
    },
    [pushHistory, saveSegmentTiming]
  );

  // Handle segment trim (after edge resizing)
  const handleSegmentTrim = useCallback(
    async (segmentId: string, startTime: number, endTime: number) => {
      // Push current state to history before making changes
      pushHistory();

      // Update local state immediately for responsive UI
      setTracks((prev) =>
        prev.map((track) => ({
          ...track,
          segments: track.segments.map((segment) =>
            segment.id === segmentId
              ? { ...segment, start_time: startTime, end_time: endTime }
              : segment
          ),
        }))
      );

      // Save to API
      await saveSegmentTiming(segmentId, startTime, endTime);
    },
    [pushHistory, saveSegmentTiming]
  );

  // Handle segment stretch (Alt+drag edge resizing with speed_factor change)
  const handleSegmentStretch = useCallback(
    async (segmentId: string, startTime: number, endTime: number, speedFactor: number) => {
      // Push current state to history before making changes
      pushHistory();

      // Update local state immediately for responsive UI
      setTracks((prev) =>
        prev.map((track) => ({
          ...track,
          segments: track.segments.map((segment) =>
            segment.id === segmentId
              ? { ...segment, start_time: startTime, end_time: endTime, speed_factor: speedFactor }
              : segment
          ),
        }))
      );

      // Save to API with speed_factor
      await saveSegmentTiming(segmentId, startTime, endTime, speedFactor);
    },
    [pushHistory, saveSegmentTiming]
  );

  // Undo handler
  const handleUndo = useCallback(() => {
    const history = historyRef.current;
    if (history.index > 0) {
      const newIndex = history.index - 1;
      const previousTracks = JSON.parse(JSON.stringify(history.tracks[newIndex]));
      historyRef.current = { ...history, index: newIndex };
      setTracks(previousTracks);
    }
  }, []);

  // Redo handler
  const handleRedo = useCallback(() => {
    const history = historyRef.current;
    if (history.index < history.tracks.length - 1) {
      const newIndex = history.index + 1;
      const nextTracks = JSON.parse(JSON.stringify(history.tracks[newIndex]));
      historyRef.current = { ...history, index: newIndex };
      setTracks(nextTracks);
    }
  }, []);

  // Delete selected segment
  const handleDeleteSegment = useCallback(async () => {
    if (!selectedSegmentId) return;

    // Push current state to history before deleting
    pushHistory();

    // Remove the segment from tracks
    setTracks((prev) =>
      prev.map((track) => ({
        ...track,
        segments: track.segments.filter((s) => s.id !== selectedSegmentId),
      }))
    );

    // Clear selection
    setSelectedSegmentId(null);
    setSelectedSegment(null);

    // Delete from Supabase
    try {
      const { error } = await supabase
        .from("dub_segments")
        .delete()
        .eq("id", selectedSegmentId);

      if (error) {
        throw new Error(error.message);
      }
    } catch (err) {
      console.error("Failed to delete segment:", err);
      setError(err instanceof Error ? err.message : "Failed to delete segment");
    }
  }, [selectedSegmentId, pushHistory]);

  // Keyboard shortcuts for undo/redo/delete
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check if we're focused on an input element
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // Ctrl/Cmd + Z for undo
      if ((e.ctrlKey || e.metaKey) && e.key === "z" && !e.shiftKey) {
        e.preventDefault();
        handleUndo();
      }

      // Ctrl/Cmd + Shift + Z for redo
      if ((e.ctrlKey || e.metaKey) && e.key === "z" && e.shiftKey) {
        e.preventDefault();
        handleRedo();
      }

      // Ctrl/Cmd + Y for redo (alternative)
      if ((e.ctrlKey || e.metaKey) && e.key === "y") {
        e.preventDefault();
        handleRedo();
      }

      // Delete or Backspace for deleting selected segment
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        handleDeleteSegment();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleUndo, handleRedo, handleDeleteSegment]);

  // Open export dialog
  const handleOpenExportDialog = () => {
    setIsExportDialogOpen(true);
  };

  // Count total segments across all tracks
  const totalSegmentCount = tracks.reduce(
    (sum, track) => sum + track.segments.length,
    0
  );

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
              {/* Autosave status indicator */}
              {saveStatus === "saving" && (
                <div className="flex items-center gap-1 text-sm text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  <span>Saving...</span>
                </div>
              )}
              {saveStatus === "saved" && (
                <div className="flex items-center gap-1 text-sm text-green-600">
                  <Check className="h-3 w-3" />
                  <span>Saved</span>
                </div>
              )}
              {saveStatus === "error" && (
                <div className="flex items-center gap-1 text-sm text-destructive">
                  <AlertCircle className="h-3 w-3" />
                  <span>Save failed</span>
                </div>
              )}
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
              <TooltipContent>Zoom Out (Ctrl+Scroll)</TooltipContent>
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
              <TooltipContent>Zoom In (Ctrl+Scroll)</TooltipContent>
            </Tooltip>

            <div className="mx-2 h-6 w-px bg-border" />

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="default"
                  size="sm"
                  onClick={handleOpenExportDialog}
                >
                  <Download className="mr-2 h-4 w-4" />
                  Export
                </Button>
              </TooltipTrigger>
              <TooltipContent>Export final video</TooltipContent>
            </Tooltip>

            <div className="mx-2 h-6 w-px bg-border" />

            {/* Keyboard shortcuts help */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8">
                  <HelpCircle className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs" side="bottom" align="end">
                <div className="space-y-2 text-xs">
                  <p className="font-semibold">Keyboard Shortcuts</p>
                  <div className="space-y-1">
                    <div className="flex justify-between gap-4">
                      <span className="text-muted-foreground">Play/Pause</span>
                      <kbd className="rounded bg-muted px-1.5 py-0.5 font-mono">Space</kbd>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-muted-foreground">Seek -1 second</span>
                      <kbd className="rounded bg-muted px-1.5 py-0.5 font-mono">←</kbd>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-muted-foreground">Seek +1 second</span>
                      <kbd className="rounded bg-muted px-1.5 py-0.5 font-mono">→</kbd>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-muted-foreground">Undo</span>
                      <kbd className="rounded bg-muted px-1.5 py-0.5 font-mono">Ctrl+Z</kbd>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-muted-foreground">Redo</span>
                      <kbd className="rounded bg-muted px-1.5 py-0.5 font-mono">Ctrl+Shift+Z</kbd>
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-muted-foreground">Delete segment</span>
                      <kbd className="rounded bg-muted px-1.5 py-0.5 font-mono">Delete</kbd>
                    </div>
                  </div>
                </div>
              </TooltipContent>
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
          <div className="flex min-w-0 flex-1 flex-col">
            {/* Video Player (50% height) */}
            <div className="flex h-1/2 items-center justify-center border-b bg-black">
              <VideoPlayer
                ref={videoPlayerRef}
                src={project.video_url}
                isPlaying={isPlaying}
                currentTime={currentTime}
                onTimeUpdate={handleTimeUpdate}
                onDurationChange={handleDurationChange}
                onPlayStateChange={handlePlayStateChange}
                playbackRate={videoSpeedFactor}
                muted
                className="h-full w-full"
              />
            </div>

            {/* Timeline Editor (50% height) - Using Twick Timeline */}
            <div className="flex h-1/2 flex-col overflow-hidden">
              {/* Timeline */}
              <div className={cn(
                "overflow-hidden transition-all duration-200",
                selectedSegment ? "flex-1" : "h-full"
              )}>
                <Timeline
                  tracks={tracks}
                  duration={duration}
                  currentTime={currentTime}
                  zoom={zoom}
                  onSeek={handleSeek}
                  onZoomChange={handleZoomChange}
                  onSegmentSelect={handleSegmentSelect}
                  onSegmentDrop={handleSegmentDrop}
                  onSegmentTrim={handleSegmentTrim}
                  onSegmentStretch={handleSegmentStretch}
                  selectedSegmentId={selectedSegmentId}
                />
              </div>

              {/* Segment Details Panel */}
              {selectedSegment && (
                <SegmentDetails
                  segment={selectedSegment}
                  onClose={handleCloseSegmentDetails}
                  className="max-h-[200px] overflow-y-auto"
                />
              )}
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

        {/* Export Dialog */}
        <ExportDialog
          open={isExportDialogOpen}
          onOpenChange={setIsExportDialogOpen}
          projectId={projectId || ""}
          projectName={project.name}
          duration={duration}
          trackCount={tracks.length}
          segmentCount={totalSegmentCount}
        />
      </div>
    </TooltipProvider>
  );
}
