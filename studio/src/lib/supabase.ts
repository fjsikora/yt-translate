import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

/**
 * Upload a video file to Supabase storage.
 * Returns the public URL of the uploaded file.
 */
export async function uploadVideo(
  file: File,
  projectId: string,
  onProgress?: (progress: number) => void
): Promise<string> {
  const fileExt = file.name.split(".").pop();
  const fileName = `${projectId}/source.${fileExt}`;

  // Signal upload started
  onProgress?.(0);

  const { data, error } = await supabase.storage
    .from("dub-videos")
    .upload(fileName, file, {
      upsert: true,
      contentType: file.type || "video/mp4",
    });

  if (error) {
    throw new Error(`Upload failed: ${error.message}`);
  }

  // Signal upload complete
  onProgress?.(100);

  // Get public URL
  const { data: urlData } = supabase.storage
    .from("dub-videos")
    .getPublicUrl(fileName);

  return urlData.publicUrl;
}
