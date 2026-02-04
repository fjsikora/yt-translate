export type ProjectStatus =
  | "pending"
  | "processing"
  | "ready"
  | "exporting"
  | "exported"
  | "error";

export interface Project {
  id: string;
  name: string;
  status: ProjectStatus;
  source_language: string;
  target_language: string;
  created_at: string;
  updated_at: string;
  video_url?: string;
  thumbnail_url?: string;
  error_message?: string;
  duration?: number;
}

export const STATUS_LABELS: Record<ProjectStatus, string> = {
  pending: "Pending",
  processing: "Processing",
  ready: "Ready to Edit",
  exporting: "Exporting",
  exported: "Exported",
  error: "Error",
};

export const STATUS_COLORS: Record<ProjectStatus, string> = {
  pending: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
  processing: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  ready: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
  exporting: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
  exported: "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200",
  error: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200",
};
