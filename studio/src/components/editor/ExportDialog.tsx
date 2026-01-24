"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import {
  Download,
  CheckCircle,
  AlertCircle,
  Loader2,
  FileVideo,
  Music,
  Clock,
} from "lucide-react";

type ExportStatus = "idle" | "exporting" | "complete" | "error";

interface ExportDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: string;
  projectName: string;
  duration: number;
  trackCount: number;
  segmentCount: number;
}

// Helper to format duration
function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function ExportDialog({
  open,
  onOpenChange,
  projectId,
  projectName,
  duration,
  trackCount,
  segmentCount,
}: ExportDialogProps) {
  const [status, setStatus] = useState<ExportStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [exportUrl, setExportUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleExport = async () => {
    setStatus("exporting");
    setProgress(0);
    setError(null);
    setExportUrl(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;

      // Simulate progress for UX (export is a single request)
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          // Slow down as we approach 90% (waiting for server response)
          if (prev >= 90) return prev;
          const increment = Math.max(1, Math.floor((90 - prev) / 10));
          return Math.min(prev + increment, 90);
        });
      }, 500);

      const response = await fetch(`${apiUrl}/api/projects/${projectId}/export`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.detail || `Export failed: ${response.statusText}`
        );
      }

      const data = await response.json();

      setProgress(100);
      setExportUrl(data.export_url);
      setStatus("complete");
    } catch (err) {
      console.error("Export failed:", err);
      setError(err instanceof Error ? err.message : "Export failed");
      setStatus("error");
    }
  };

  const handleDownload = () => {
    if (exportUrl) {
      window.open(exportUrl, "_blank");
    }
  };

  const handleClose = () => {
    if (status !== "exporting") {
      // Reset state when closing
      setStatus("idle");
      setProgress(0);
      setExportUrl(null);
      setError(null);
      onOpenChange(false);
    }
  };

  const handleRetry = () => {
    setStatus("idle");
    setProgress(0);
    setError(null);
    setExportUrl(null);
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Export Video</DialogTitle>
          <DialogDescription>
            {status === "idle" &&
              "Review export settings and start the export process."}
            {status === "exporting" && "Exporting your dubbed video..."}
            {status === "complete" && "Your video is ready for download!"}
            {status === "error" && "There was an error during export."}
          </DialogDescription>
        </DialogHeader>

        <div className="py-4">
          {/* Project Summary - shown in idle state */}
          {status === "idle" && (
            <div className="space-y-4">
              <div className="rounded-lg border bg-muted/30 p-4">
                <h4 className="mb-3 text-sm font-medium">Export Summary</h4>
                <div className="grid gap-3">
                  <div className="flex items-center gap-3">
                    <FileVideo className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">
                      Project:
                    </span>
                    <span className="text-sm font-medium">{projectName}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">
                      Duration:
                    </span>
                    <span className="text-sm font-medium">
                      {formatDuration(duration)}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Music className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">
                      Tracks:
                    </span>
                    <span className="text-sm font-medium">
                      {trackCount} tracks, {segmentCount} segments
                    </span>
                  </div>
                </div>
              </div>

              <p className="text-xs text-muted-foreground">
                The export will apply all your edits (timing changes, speed
                adjustments, track mute/solo/volume settings) and generate a
                final video file.
              </p>
            </div>
          )}

          {/* Progress - shown during export */}
          {status === "exporting" && (
            <div className="space-y-4">
              <div className="flex flex-col items-center gap-4 py-6">
                <Loader2 className="h-12 w-12 animate-spin text-primary" />
                <div className="w-full space-y-2">
                  <Progress value={progress} className="h-2" />
                  <p className="text-center text-sm text-muted-foreground">
                    {progress < 90
                      ? "Processing audio tracks..."
                      : "Finalizing video..."}
                  </p>
                </div>
              </div>
              <p className="text-center text-xs text-muted-foreground">
                Please wait. This may take a few minutes depending on the video
                length.
              </p>
            </div>
          )}

          {/* Success - shown when complete */}
          {status === "complete" && (
            <div className="space-y-4">
              <div className="flex flex-col items-center gap-4 py-6">
                <div className="rounded-full bg-green-100 p-3 dark:bg-green-900/30">
                  <CheckCircle className="h-12 w-12 text-green-600 dark:text-green-400" />
                </div>
                <div className="text-center">
                  <p className="font-medium">Export Complete!</p>
                  <p className="text-sm text-muted-foreground">
                    Your dubbed video is ready to download.
                  </p>
                </div>
              </div>

              {exportUrl && (
                <Button
                  className="w-full"
                  size="lg"
                  onClick={handleDownload}
                >
                  <Download className="mr-2 h-4 w-4" />
                  Download Video
                </Button>
              )}
            </div>
          )}

          {/* Error - shown on failure */}
          {status === "error" && (
            <div className="space-y-4">
              <div className="flex flex-col items-center gap-4 py-6">
                <div className="rounded-full bg-red-100 p-3 dark:bg-red-900/30">
                  <AlertCircle className="h-12 w-12 text-red-600 dark:text-red-400" />
                </div>
                <div className="text-center">
                  <p className="font-medium">Export Failed</p>
                  <p className="text-sm text-muted-foreground">{error}</p>
                </div>
              </div>

              <Button
                variant="outline"
                className="w-full"
                onClick={handleRetry}
              >
                Try Again
              </Button>
            </div>
          )}
        </div>

        <DialogFooter>
          {status === "idle" && (
            <>
              <Button variant="outline" onClick={handleClose}>
                Cancel
              </Button>
              <Button onClick={handleExport}>
                <Download className="mr-2 h-4 w-4" />
                Start Export
              </Button>
            </>
          )}

          {status === "exporting" && (
            <p className="text-xs text-muted-foreground">
              Export in progress...
            </p>
          )}

          {(status === "complete" || status === "error") && (
            <Button variant="outline" onClick={handleClose}>
              Close
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
