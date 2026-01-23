"use client";

import { useState, useEffect } from "react";
import { ProjectCard } from "@/components/ProjectCard";
import { NewProjectDialog } from "@/components/NewProjectDialog";
import { Project } from "@/types/project";
import { Film } from "lucide-react";

// Mock data for development - will be replaced with API call once US-013 is complete
const MOCK_PROJECTS: Project[] = [
  {
    id: "1",
    name: "Product Demo Video",
    status: "ready",
    source_language: "en",
    target_language: "es",
    created_at: "2026-01-20T10:30:00Z",
    updated_at: "2026-01-21T14:45:00Z",
  },
  {
    id: "2",
    name: "Marketing Webinar",
    status: "processing",
    source_language: "en",
    target_language: "fr",
    created_at: "2026-01-22T09:00:00Z",
    updated_at: "2026-01-22T09:15:00Z",
  },
  {
    id: "3",
    name: "Tutorial Series Episode 1",
    status: "pending",
    source_language: "en",
    target_language: "de",
    created_at: "2026-01-23T08:00:00Z",
    updated_at: "2026-01-23T08:00:00Z",
  },
];

export default function HomePage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchProjects = async () => {
    setIsLoading(true);
    try {
      // TODO: Replace with actual API call once US-013 is complete
      // const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/projects`);
      // const data = await response.json();
      // setProjects(data);

      // For now, use mock data
      await new Promise((resolve) => setTimeout(resolve, 300));
      setProjects(MOCK_PROJECTS);
    } catch (error) {
      console.error("Failed to fetch projects:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchProjects();
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Film className="h-6 w-6" />
              <h1 className="text-xl font-bold">Dubbing Studio</h1>
            </div>
            <NewProjectDialog onProjectCreated={fetchProjects} />
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <h2 className="mb-6 text-2xl font-semibold">Your Projects</h2>

        {isLoading ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-[140px] animate-pulse rounded-xl border bg-muted"
              />
            ))}
          </div>
        ) : projects.length === 0 ? (
          <div className="flex flex-col items-center justify-center rounded-xl border border-dashed p-12 text-center">
            <Film className="mb-4 h-12 w-12 text-muted-foreground" />
            <h3 className="mb-2 text-lg font-medium">No projects yet</h3>
            <p className="mb-4 text-sm text-muted-foreground">
              Get started by creating your first dubbing project.
            </p>
            <NewProjectDialog onProjectCreated={fetchProjects} />
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {projects.map((project) => (
              <ProjectCard key={project.id} project={project} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
