"use client";

import { X, User, Languages, Clock, Gauge } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TimelineSegment } from "./Timeline";
import { cn } from "@/lib/utils";

interface SegmentDetailsProps {
  segment: TimelineSegment | null;
  onClose: () => void;
  className?: string;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins}:${secs.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`;
}

export function SegmentDetails({ segment, onClose, className }: SegmentDetailsProps) {
  if (!segment) return null;

  const duration = segment.end_time - segment.start_time;

  return (
    <div
      className={cn(
        "border-t bg-muted/30 animate-in slide-in-from-bottom-2 duration-200",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b px-4 py-2">
        <h3 className="text-sm font-medium">Segment Details</h3>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={onClose}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Content */}
      <div className="grid grid-cols-2 gap-4 p-4">
        {/* Left column - Text content */}
        <div className="space-y-4">
          {/* Original text */}
          {segment.original_text && (
            <div>
              <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                <Languages className="h-3 w-3" />
                Original Text
              </div>
              <div className="rounded-md border bg-background p-3 text-sm">
                {segment.original_text}
              </div>
            </div>
          )}

          {/* Translated text */}
          {segment.translated_text && (
            <div>
              <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                <Languages className="h-3 w-3" />
                Translated Text
              </div>
              <div className="rounded-md border bg-background p-3 text-sm">
                {segment.translated_text}
              </div>
            </div>
          )}

          {/* No text message */}
          {!segment.original_text && !segment.translated_text && (
            <div className="flex h-20 items-center justify-center rounded-md border bg-muted/50 text-sm text-muted-foreground">
              No text content available for this segment
            </div>
          )}
        </div>

        {/* Right column - Metadata */}
        <div className="space-y-3">
          {/* Speaker */}
          {segment.speaker && (
            <div className="flex items-center gap-3 rounded-md border bg-background p-3">
              <User className="h-4 w-4 text-muted-foreground" />
              <div>
                <div className="text-xs text-muted-foreground">Speaker</div>
                <div className="text-sm font-medium">{segment.speaker}</div>
              </div>
            </div>
          )}

          {/* Timing */}
          <div className="flex items-center gap-3 rounded-md border bg-background p-3">
            <Clock className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">Timing</div>
              <div className="text-sm font-medium">
                {formatTime(segment.start_time)} - {formatTime(segment.end_time)}
                <span className="ml-2 text-muted-foreground">
                  ({duration.toFixed(2)}s)
                </span>
              </div>
            </div>
          </div>

          {/* Speed factor */}
          <div className="flex items-center gap-3 rounded-md border bg-background p-3">
            <Gauge className="h-4 w-4 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">Speed Factor</div>
              <div className="text-sm font-medium">
                {segment.speed_factor.toFixed(2)}x
                {segment.speed_factor !== 1.0 && (
                  <span className="ml-2 text-muted-foreground">
                    (Original: {(duration * segment.speed_factor).toFixed(2)}s)
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SegmentDetails;
