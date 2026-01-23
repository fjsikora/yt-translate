import { cn } from "@/lib/utils";
import { ProjectStatus, STATUS_LABELS, STATUS_COLORS } from "@/types/project";

interface StatusBadgeProps {
  status: ProjectStatus;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        STATUS_COLORS[status],
        className
      )}
    >
      {status === "processing" && (
        <span className="mr-1.5 h-2 w-2 animate-pulse rounded-full bg-current" />
      )}
      {STATUS_LABELS[status]}
    </span>
  );
}
