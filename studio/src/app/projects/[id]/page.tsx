"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/StatusBadge";
import { supabase } from "@/lib/supabase";
import { Project, ProjectStatus } from "@/types/project";
import {
  ArrowLeft,
  Play,
  Edit,
  RefreshCw,
  AlertCircle,
  Clock,
  Languages,
} from "lucide-react";

interface ProjectPageProps {
  params: Promise<{ id: string }>;
}

interface ProjectDetail extends Project {
  error_message?: string;
  duration?: number;
}

// Language code to name mapping
const LANGUAGE_NAMES: Record<string, string> = {
  en: "English",
  es: "Spanish",
  fr: "French",
  de: "German",
  it: "Italian",
  pt: "Portuguese",
  ja: "Japanese",
  ko: "Korean",
  zh: "Chinese",
  ru: "Russian",
  ar: "Arabic",
  hi: "Hindi",
};

function getLanguageName(code: string): string {
  return LANGUAGE_NAMES[code] || code.toUpperCase();
}

function formatDuration(seconds?: number): string {
  if (!seconds) return "--:--";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function ProjectPage({ params }: ProjectPageProps) {
  const [projectId, setProjectId] = useState<string | null>(null);
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isStartingDub, setIsStartingDub] = useState(false);

  // Resolve params promise
  useEffect(() => {
    params.then((p) => setProjectId(p.id));
  }, [params]);

  const fetchProject = useCallback(async () => {
    if (!projectId) return;

    try {
      const { data, error: queryError } = await supabase
        .from("dub_projects")
        .select("*")
        .eq("id", projectId)
        .single();

      if (queryError) {
        if (queryError.code === "PGRST116") {
          throw new Error("Project not found");
        }
        throw new Error(queryError.message);
      }

      setProject(data);
      setError(null);
    } catch (err) {
      console.error("Failed to fetch project:", err);
      setError(err instanceof Error ? err.message : "Failed to fetch project");
    } finally {
      setIsLoading(false);
    }
  }, [projectId]);

  // Initial fetch
  useEffect(() => {
    if (projectId) {
      fetchProject();
    }
  }, [projectId, fetchProject]);

  // Poll status every 5 seconds while processing
  useEffect(() => {
    if (!project || project.status !== "processing") return;

    const pollInterval = setInterval(() => {
      fetchProject();
    }, 5000);

    return () => clearInterval(pollInterval);
  }, [project, fetchProject]);

  const handleStartDubbing = async () => {
    if (!project || !project.video_url) return;

    setIsStartingDub(true);
    setError(null);

    // Set processing state immediately so the UI shows the processing
    // indicator and polling starts. The backend also sets this in the DB.
    setProject((prev) =>
      prev ? { ...prev, status: "processing" as ProjectStatus } : null
    );

    const apiUrl = process.env.NEXT_PUBLIC_API_URL;

    // Fire-and-forget: the pipeline takes several minutes and the Self-hosted GPU
    // proxy will timeout (HTTP 524) before it completes. The backend
    // updates project status in the DB; the 5-second polling loop picks
    // up "ready" or "error" status automatically.
    fetch(`${apiUrl}/dub`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        video_url: project.video_url,
        source_lang: project.source_language,
        target_lang: project.target_language,
        project_id: project.id,
      }),
    }).catch(() => {
      // Ignore network/timeout errors — the backend is processing.
      // Status polling will detect any errors set by the backend.
    });

    setIsStartingDub(false);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <header className="border-b">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center gap-4">
              <Link href="/">
                <Button variant="ghost" size="icon">
                  <ArrowLeft className="h-4 w-4" />
                </Button>
              </Link>
              <div className="h-6 w-48 animate-pulse rounded bg-muted" />
            </div>
          </div>
        </header>
        <main className="container mx-auto px-4 py-8">
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <div className="aspect-video animate-pulse rounded-xl bg-muted" />
            </div>
            <div className="h-64 animate-pulse rounded-xl bg-muted" />
          </div>
        </main>
      </div>
    );
  }

  if (error && !project) {
    return (
      <div className="min-h-screen bg-background">
        <header className="border-b">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center gap-4">
              <Link href="/">
                <Button variant="ghost" size="icon">
                  <ArrowLeft className="h-4 w-4" />
                </Button>
              </Link>
              <h1 className="text-xl font-bold">Project Details</h1>
            </div>
          </div>
        </header>
        <main className="container mx-auto px-4 py-8">
          <Card className="border-destructive">
            <CardContent className="flex flex-col items-center justify-center py-12">
              <AlertCircle className="mb-4 h-12 w-12 text-destructive" />
              <h2 className="mb-2 text-lg font-medium">Error Loading Project</h2>
              <p className="mb-4 text-sm text-muted-foreground">{error}</p>
              <Link href="/">
                <Button>Back to Projects</Button>
              </Link>
            </CardContent>
          </Card>
        </main>
      </div>
    );
  }

  if (!project) return null;

  const canStartDubbing =
    project.status === "pending" && project.video_url && !isStartingDub;
  const canOpenEditor =
    project.status === "ready" ||
    project.status === "exported" ||
    project.status === "exporting";
  const isProcessing = project.status === "processing";

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/">
                <Button variant="ghost" size="icon">
                  <ArrowLeft className="h-4 w-4" />
                </Button>
              </Link>
              <div>
                <h1 className="text-xl font-bold">{project.name}</h1>
                <StatusBadge status={project.status} />
              </div>
            </div>
            <div className="flex items-center gap-2">
              {canStartDubbing && (
                <Button onClick={handleStartDubbing} disabled={isStartingDub}>
                  {isStartingDub ? (
                    <>
                      <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                      Starting...
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Start Dubbing
                    </>
                  )}
                </Button>
              )}
              {canOpenEditor && (
                <Link href={`/projects/${project.id}/edit`}>
                  <Button>
                    <Edit className="mr-2 h-4 w-4" />
                    Open Editor
                  </Button>
                </Link>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Video Preview */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Video Preview</CardTitle>
                <CardDescription>Source video for dubbing</CardDescription>
              </CardHeader>
              <CardContent>
                {project.video_url ? (
                  <video
                    className="w-full rounded-lg bg-black"
                    controls
                    preload="metadata"
                    src={project.video_url}
                  >
                    Your browser does not support the video tag.
                  </video>
                ) : (
                  <div className="flex aspect-video items-center justify-center rounded-lg bg-muted">
                    <p className="text-sm text-muted-foreground">
                      No video uploaded
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Project Info */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Project Details</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Languages */}
                <div className="flex items-start gap-3">
                  <Languages className="mt-0.5 h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">Languages</p>
                    <p className="text-sm text-muted-foreground">
                      {getLanguageName(project.source_language)} →{" "}
                      {getLanguageName(project.target_language)}
                    </p>
                  </div>
                </div>

                {/* Duration */}
                <div className="flex items-start gap-3">
                  <Clock className="mt-0.5 h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="text-sm font-medium">Duration</p>
                    <p className="text-sm text-muted-foreground">
                      {formatDuration(project.duration)}
                    </p>
                  </div>
                </div>

                {/* Created Date */}
                <div className="pt-4 border-t">
                  <p className="text-xs text-muted-foreground">
                    Created: {formatDate(project.created_at)}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Updated: {formatDate(project.updated_at)}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Processing Status */}
            {isProcessing && (
              <Card className="border-blue-200 dark:border-blue-800">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-3">
                    <RefreshCw className="h-5 w-5 animate-spin text-blue-600" />
                    <div>
                      <p className="font-medium text-blue-600">
                        Processing Video
                      </p>
                      <p className="text-sm text-muted-foreground">
                        Running AI dubbing pipeline...
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Status updates every 5 seconds
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Error Status */}
            {project.status === "error" && (
              <Card className="border-destructive">
                <CardContent className="pt-6">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="h-5 w-5 text-destructive mt-0.5" />
                    <div>
                      <p className="font-medium text-destructive">
                        Pipeline Error
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {project.error_message ||
                          "An error occurred during processing."}
                      </p>
                      <Button
                        variant="outline"
                        size="sm"
                        className="mt-2"
                        onClick={handleStartDubbing}
                        disabled={isStartingDub}
                      >
                        <RefreshCw className="mr-2 h-3 w-3" />
                        Retry
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Error from actions */}
            {error && project && (
              <Card className="border-destructive">
                <CardContent className="pt-6">
                  <div className="flex items-start gap-3">
                    <AlertCircle className="h-5 w-5 text-destructive mt-0.5" />
                    <div>
                      <p className="font-medium text-destructive">Error</p>
                      <p className="text-sm text-muted-foreground">{error}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
