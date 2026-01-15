-- Initial database schema for video translation monetization
-- This migration creates the core tables: profiles, preview_jobs, and translation_jobs

-- =============================================================================
-- PROFILES TABLE
-- Stores user profile information linked to Supabase Auth
-- =============================================================================
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    tos_accepted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Enable RLS on profiles
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Users can only read their own profile
CREATE POLICY "Users can read own profile"
    ON profiles FOR SELECT
    USING (auth.uid() = id);

-- Users can update their own profile
CREATE POLICY "Users can update own profile"
    ON profiles FOR UPDATE
    USING (auth.uid() = id);

-- Allow insert for new users (triggered by auth)
CREATE POLICY "Users can insert own profile"
    ON profiles FOR INSERT
    WITH CHECK (auth.uid() = id);


-- =============================================================================
-- PREVIEW_JOBS TABLE
-- Stores free 60-second preview translation jobs
-- =============================================================================
CREATE TABLE IF NOT EXISTS preview_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    video_url TEXT NOT NULL,
    video_title TEXT,
    video_duration_seconds INTEGER,
    target_language TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    stage TEXT,
    preview_file_path TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_preview_jobs_session_id ON preview_jobs(session_id);
CREATE INDEX IF NOT EXISTS idx_preview_jobs_user_id ON preview_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_preview_jobs_status ON preview_jobs(status);

-- Enable RLS on preview_jobs
ALTER TABLE preview_jobs ENABLE ROW LEVEL SECURITY;

-- Guests can read their own previews by session_id (anon access)
CREATE POLICY "Anyone can read preview by session_id"
    ON preview_jobs FOR SELECT
    USING (true);  -- Session ID check done at application level for anon users

-- Authenticated users can read their own preview jobs
CREATE POLICY "Users can read own preview jobs"
    ON preview_jobs FOR SELECT
    USING (auth.uid() = user_id);

-- Allow insert from service role (backend creates these)
CREATE POLICY "Service role can insert preview jobs"
    ON preview_jobs FOR INSERT
    WITH CHECK (true);  -- Service role bypasses RLS anyway

-- Allow update from service role (for status updates)
CREATE POLICY "Service role can update preview jobs"
    ON preview_jobs FOR UPDATE
    USING (true);  -- Service role bypasses RLS anyway


-- =============================================================================
-- TRANSLATION_JOBS TABLE
-- Stores paid full translation jobs
-- =============================================================================
CREATE TABLE IF NOT EXISTS translation_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    preview_job_id UUID REFERENCES preview_jobs(id) ON DELETE SET NULL,
    video_url TEXT NOT NULL,
    video_title TEXT,
    target_language TEXT NOT NULL,
    processing_cost INTEGER NOT NULL CHECK (processing_cost > 0),
    payment-provider_checkout_session_id TEXT,
    payment-provider_payment_intent_id TEXT,
    payment_status TEXT NOT NULL DEFAULT 'pending' CHECK (payment_status IN ('pending', 'paid', 'failed', 'refunded')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    stage TEXT,
    output_file_path TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Create indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_translation_jobs_user_id ON translation_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_translation_jobs_payment-provider_session ON translation_jobs(payment-provider_checkout_session_id);
CREATE INDEX IF NOT EXISTS idx_translation_jobs_status ON translation_jobs(status);
CREATE INDEX IF NOT EXISTS idx_translation_jobs_payment_status ON translation_jobs(payment_status);

-- Enable RLS on translation_jobs
ALTER TABLE translation_jobs ENABLE ROW LEVEL SECURITY;

-- Users can only read their own translation jobs
CREATE POLICY "Users can read own translation jobs"
    ON translation_jobs FOR SELECT
    USING (auth.uid() = user_id);

-- Allow insert from service role (backend creates these after checkout)
CREATE POLICY "Service role can insert translation jobs"
    ON translation_jobs FOR INSERT
    WITH CHECK (true);  -- Service role bypasses RLS anyway

-- Allow update from service role (for status updates)
CREATE POLICY "Service role can update translation jobs"
    ON translation_jobs FOR UPDATE
    USING (true);  -- Service role bypasses RLS anyway


-- =============================================================================
-- UPDATED_AT TRIGGER FUNCTION
-- Automatically updates the updated_at column on row changes
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to preview_jobs
CREATE TRIGGER update_preview_jobs_updated_at
    BEFORE UPDATE ON preview_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to translation_jobs
CREATE TRIGGER update_translation_jobs_updated_at
    BEFORE UPDATE ON translation_jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- =============================================================================
-- STORAGE BUCKETS (for Supabase Storage)
-- =============================================================================
-- Note: These need to be created via Supabase Dashboard or API
-- Bucket names:
--   - previews: For 60-second preview videos
--   - translations: For full translated videos
