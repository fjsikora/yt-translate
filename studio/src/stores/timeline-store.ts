import { create } from "zustand";
import type { TimelineTrack } from "@/components/editor/Timeline";

interface TimelineState {
  // Playback state
  isPlaying: boolean;
  currentTime: number;
  duration: number;

  // Timeline view state
  zoom: number;
  scrollLeft: number;

  // Selection state
  selectedSegmentId: string | null;
  selectedTrackId: string | null;

  // Track data
  tracks: TimelineTrack[];

  // Undo/redo history
  history: TimelineTrack[][];
  historyIndex: number;

  // Actions
  setIsPlaying: (isPlaying: boolean) => void;
  setCurrentTime: (time: number) => void;
  setDuration: (duration: number) => void;
  setZoom: (zoom: number) => void;
  setScrollLeft: (scrollLeft: number) => void;
  selectSegment: (segmentId: string | null) => void;
  selectTrack: (trackId: string | null) => void;
  setTracks: (tracks: TimelineTrack[]) => void;

  // Track operations
  muteTrack: (trackId: string) => void;
  soloTrack: (trackId: string) => void;
  setTrackVolume: (trackId: string, volume: number) => void;

  // Segment operations
  updateSegmentTiming: (
    segmentId: string,
    startTime: number,
    endTime: number
  ) => void;
  updateSegmentSpeed: (segmentId: string, speedFactor: number) => void;

  // History operations
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;
  pushHistory: () => void;
}

const MAX_HISTORY_SIZE = 50;

export const useTimelineStore = create<TimelineState>((set, get) => ({
  // Initial playback state
  isPlaying: false,
  currentTime: 0,
  duration: 0,

  // Initial view state
  zoom: 1,
  scrollLeft: 0,

  // Initial selection state
  selectedSegmentId: null,
  selectedTrackId: null,

  // Initial track data
  tracks: [],

  // Initial history
  history: [],
  historyIndex: -1,

  // Playback actions
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  setCurrentTime: (currentTime) => set({ currentTime }),
  setDuration: (duration) => set({ duration }),

  // View actions
  setZoom: (zoom) => set({ zoom: Math.max(0.25, Math.min(10, zoom)) }),
  setScrollLeft: (scrollLeft) => set({ scrollLeft }),

  // Selection actions
  selectSegment: (segmentId) => set({ selectedSegmentId: segmentId }),
  selectTrack: (trackId) => set({ selectedTrackId: trackId }),

  // Track data actions
  setTracks: (tracks) => {
    const state = get();
    if (state.history.length === 0) {
      // Initialize history with first state
      set({ tracks, history: [tracks], historyIndex: 0 });
    } else {
      set({ tracks });
    }
  },

  // Track operations
  muteTrack: (trackId) => {
    get().pushHistory();
    set((state) => ({
      tracks: state.tracks.map((track) =>
        track.id === trackId ? { ...track, muted: !track.muted } : track
      ),
    }));
  },

  soloTrack: (trackId) => {
    get().pushHistory();
    set((state) => ({
      tracks: state.tracks.map((track) =>
        track.id === trackId ? { ...track, solo: !track.solo } : track
      ),
    }));
  },

  setTrackVolume: (trackId, volume) => {
    set((state) => ({
      tracks: state.tracks.map((track) =>
        track.id === trackId
          ? { ...track, volume: Math.max(0, Math.min(2, volume)) }
          : track
      ),
    }));
  },

  // Segment operations
  updateSegmentTiming: (segmentId, startTime, endTime) => {
    get().pushHistory();
    set((state) => ({
      tracks: state.tracks.map((track) => ({
        ...track,
        segments: track.segments.map((segment) =>
          segment.id === segmentId
            ? { ...segment, start_time: startTime, end_time: endTime }
            : segment
        ),
      })),
    }));
  },

  updateSegmentSpeed: (segmentId, speedFactor) => {
    get().pushHistory();
    set((state) => ({
      tracks: state.tracks.map((track) => ({
        ...track,
        segments: track.segments.map((segment) =>
          segment.id === segmentId
            ? { ...segment, speed_factor: Math.max(0.5, Math.min(2, speedFactor)) }
            : segment
        ),
      })),
    }));
  },

  // History operations
  pushHistory: () => {
    set((state) => {
      const newHistory = state.history.slice(0, state.historyIndex + 1);
      newHistory.push(JSON.parse(JSON.stringify(state.tracks)));

      // Trim history if too long
      if (newHistory.length > MAX_HISTORY_SIZE) {
        newHistory.shift();
      }

      return {
        history: newHistory,
        historyIndex: newHistory.length - 1,
      };
    });
  },

  undo: () => {
    set((state) => {
      if (state.historyIndex > 0) {
        const newIndex = state.historyIndex - 1;
        return {
          tracks: JSON.parse(JSON.stringify(state.history[newIndex])),
          historyIndex: newIndex,
        };
      }
      return state;
    });
  },

  redo: () => {
    set((state) => {
      if (state.historyIndex < state.history.length - 1) {
        const newIndex = state.historyIndex + 1;
        return {
          tracks: JSON.parse(JSON.stringify(state.history[newIndex])),
          historyIndex: newIndex,
        };
      }
      return state;
    });
  },

  canUndo: () => {
    const state = get();
    return state.historyIndex > 0;
  },

  canRedo: () => {
    const state = get();
    return state.historyIndex < state.history.length - 1;
  },
}));

export type { TimelineState };
