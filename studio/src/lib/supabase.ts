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

  // Supabase JS v2 upload doesn't support native progress tracking
  // We simulate progress by using XMLHttpRequest for the upload
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

    xhr.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable && onProgress) {
        const progress = Math.round((event.loaded / event.total) * 100);
        onProgress(progress);
      }
    });

    xhr.addEventListener("load", async () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        // Get public URL after successful upload
        const { data } = supabase.storage
          .from("dub-videos")
          .getPublicUrl(fileName);
        resolve(data.publicUrl);
      } else {
        reject(new Error(`Upload failed with status ${xhr.status}`));
      }
    });

    xhr.addEventListener("error", () => {
      reject(new Error("Upload failed due to network error"));
    });

    // Build the Supabase storage upload URL
    const uploadUrl = `${supabaseUrl}/storage/v1/object/dub-videos/${fileName}`;

    xhr.open("POST", uploadUrl);
    xhr.setRequestHeader("Authorization", `Bearer ${supabaseAnonKey}`);
    xhr.setRequestHeader("x-upsert", "true");
    xhr.send(file);
  });
}
