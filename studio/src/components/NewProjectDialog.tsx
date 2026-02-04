"use client";

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Plus, Upload, X, FileVideo } from "lucide-react";
import { supabase, uploadVideo } from "@/lib/supabase";

const LANGUAGES = [
  { code: "en", name: "English" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "it", name: "Italian" },
  { code: "pt", name: "Portuguese" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "zh", name: "Chinese" },
  { code: "ru", name: "Russian" },
  { code: "ar", name: "Arabic" },
  { code: "hi", name: "Hindi" },
];

const ACCEPTED_VIDEO_TYPES = [
  "video/mp4",
  "video/quicktime", // .mov
  "video/webm",
];

const ACCEPTED_EXTENSIONS = ".mp4,.mov,.webm";

interface NewProjectDialogProps {
  onProjectCreated?: () => void;
}

export function NewProjectDialog({ onProjectCreated }: NewProjectDialogProps) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [sourceLanguage, setSourceLanguage] = useState("en");
  const [targetLanguage, setTargetLanguage] = useState("es");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!ACCEPTED_VIDEO_TYPES.includes(file.type)) {
      setError("Please select a valid video file (MP4, MOV, or WebM)");
      return;
    }

    setSelectedFile(file);
    setError(null);
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setUploadProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleCreate = async () => {
    if (!name.trim()) {
      setError("Please enter a project name");
      return;
    }
    if (!selectedFile) {
      setError("Please select a video file");
      return;
    }

    setIsCreating(true);
    setError(null);
    setUploadProgress(0);

    try {
      // Generate a temporary ID for the upload path (will be replaced by actual ID)
      const tempId = crypto.randomUUID();

      // Upload video to Supabase storage
      const videoUrl = await uploadVideo(selectedFile, tempId, (progress) => {
        setUploadProgress(progress);
      });

      // Create project in Supabase
      const { error: insertError } = await supabase
        .from("dub_projects")
        .insert({
          name: name.trim(),
          source_language: sourceLanguage,
          target_language: targetLanguage,
          video_url: videoUrl,
          status: "pending",
        });

      if (insertError) {
        throw new Error(insertError.message);
      }

      // Success - close dialog and reset form
      setOpen(false);
      resetForm();
      onProjectCreated?.();
    } catch (err) {
      console.error("Failed to create project:", err);
      setError(err instanceof Error ? err.message : "Failed to create project");
    } finally {
      setIsCreating(false);
    }
  };

  const resetForm = () => {
    setName("");
    setSourceLanguage("en");
    setTargetLanguage("es");
    setSelectedFile(null);
    setUploadProgress(0);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleOpenChange = (newOpen: boolean) => {
    setOpen(newOpen);
    if (!newOpen) {
      resetForm();
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          New Project
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Create New Project</DialogTitle>
          <DialogDescription>
            Upload a video and select languages to start a new dubbing project.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          {/* Project Name */}
          <div className="grid gap-2">
            <Label htmlFor="name">Project Name</Label>
            <Input
              id="name"
              placeholder="My Dubbing Project"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={isCreating}
            />
          </div>

          {/* Source Language */}
          <div className="grid gap-2">
            <Label htmlFor="source-language">Source Language</Label>
            <Select
              value={sourceLanguage}
              onValueChange={setSourceLanguage}
              disabled={isCreating}
            >
              <SelectTrigger id="source-language">
                <SelectValue placeholder="Select source language" />
              </SelectTrigger>
              <SelectContent>
                {LANGUAGES.map((lang) => (
                  <SelectItem key={lang.code} value={lang.code}>
                    {lang.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Target Language */}
          <div className="grid gap-2">
            <Label htmlFor="target-language">Target Language</Label>
            <Select
              value={targetLanguage}
              onValueChange={setTargetLanguage}
              disabled={isCreating}
            >
              <SelectTrigger id="target-language">
                <SelectValue placeholder="Select target language" />
              </SelectTrigger>
              <SelectContent>
                {LANGUAGES.map((lang) => (
                  <SelectItem key={lang.code} value={lang.code}>
                    {lang.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Video Upload */}
          <div className="grid gap-2">
            <Label>Video File</Label>
            {!selectedFile ? (
              <div
                className="flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-muted-foreground/25 p-6 transition-colors hover:border-muted-foreground/50 cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className="h-8 w-8 text-muted-foreground" />
                <div className="text-center">
                  <p className="text-sm font-medium">Click to upload</p>
                  <p className="text-xs text-muted-foreground">
                    MP4, MOV, or WebM (max 500MB)
                  </p>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={ACCEPTED_EXTENSIONS}
                  onChange={handleFileSelect}
                  className="hidden"
                  disabled={isCreating}
                />
              </div>
            ) : (
              <div className="rounded-lg border p-3">
                <div className="flex items-center gap-3">
                  <FileVideo className="h-8 w-8 text-muted-foreground flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">
                      {selectedFile.name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(selectedFile.size)}
                    </p>
                  </div>
                  {!isCreating && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 flex-shrink-0"
                      onClick={handleRemoveFile}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  )}
                </div>

                {/* Progress Bar */}
                {isCreating && uploadProgress > 0 && (
                  <div className="mt-3 space-y-1">
                    <Progress value={uploadProgress} />
                    <p className="text-xs text-muted-foreground text-right">
                      {uploadProgress < 100
                        ? `Uploading... ${uploadProgress}%`
                        : "Creating project..."}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <p className="text-sm text-destructive">{error}</p>
          )}
        </div>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => handleOpenChange(false)}
            disabled={isCreating}
          >
            Cancel
          </Button>
          <Button
            onClick={handleCreate}
            disabled={!name.trim() || !selectedFile || isCreating}
          >
            {isCreating
              ? uploadProgress < 100
                ? "Uploading..."
                : "Creating..."
              : "Create Project"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
