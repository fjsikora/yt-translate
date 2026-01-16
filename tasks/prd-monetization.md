# PRD: Video Translation Monetization

## 1. Overview

### Problem Statement
The Universal Video Translator currently has no way to monetize. All translation services charge monthly subscriptions, creating a barrier for occasional users who just want to translate one video.

### Solution
Implement a pay-per-video model with a freemium conversion funnel:
- Free 60-second preview (no account required)
- Per-minute pricing for full translation
- One-time payments via Stripe (no subscriptions)

## 2. Goals

- [ ] Users can preview first 60 seconds of any video translation for free
- [ ] Users can pay per-minute to get full translated video
- [ ] User accounts persist job history and enable repeat purchases
- [ ] Platform generates revenue from video translations

## 3. Non-Goals (Out of Scope)

- Subscription plans (future consideration)
- Bulk/enterprise pricing
- Referral/affiliate program
- Multiple simultaneous jobs per user
- Video editing features
- Mobile app

## 4. User Stories

### US-001: Database Schema Setup

**Description:** As a developer, I need database tables to store users, preview jobs, and translation jobs so that data persists across sessions.

**Acceptance Criteria:**
- [ ] `profiles` table created with id, email, tos_accepted_at, created_at
- [ ] `preview_jobs` table created with session_id, user_id, video_url, target_language, status, preview_file_path
- [ ] `translation_jobs` table created with user_id, preview_job_id, price_cents, stripe fields, status, output_file_path
- [ ] Row Level Security (RLS) enabled on all tables
- [ ] Users can only read their own records
- [ ] Typecheck passes

**Priority:** 1

---

### US-002: Supabase Client Integration

**Description:** As a developer, I need Supabase Python client configured so that the backend can interact with the database.

**Acceptance Criteria:**
- [ ] `supabase` package added to requirements-cloud.txt
- [ ] Supabase client initialized with env vars (SUPABASE_URL, SUPABASE_SERVICE_KEY)
- [ ] Helper functions for CRUD operations on each table
- [ ] Connection tested successfully
- [ ] Typecheck passes

**Priority:** 2

---

### US-003: Video Info Extraction Endpoint

**Description:** As a user, I want to paste a video URL and see its title and duration before committing to a preview.

**Acceptance Criteria:**
- [ ] `POST /video-info` endpoint accepts video_url
- [ ] Returns video_title, duration_seconds, thumbnail_url (if available)
- [ ] Returns price_quote for full translation (calculated from duration)
- [ ] Handles invalid URLs with appropriate error message
- [ ] Typecheck passes

**Priority:** 3

---

### US-004: Preview Job Creation

**Description:** As a guest user, I want to start a free preview translation so I can evaluate the quality before paying.

**Acceptance Criteria:**
- [ ] `POST /preview` endpoint accepts video_url, target_language, session_id
- [ ] Creates preview_job record in Supabase
- [ ] Starts background task to process first 60 seconds only
- [ ] Returns preview_id and initial status
- [ ] Typecheck passes

**Priority:** 4

---

### US-005: Partial Video Download (60 seconds)

**Description:** As a developer, I need to download only the first 60 seconds of a video for preview processing.

**Acceptance Criteria:**
- [ ] Modify download_video function to accept optional duration_limit parameter
- [ ] Uses ffmpeg to extract only first 60 seconds
- [ ] Audio extraction limited to 60 seconds
- [ ] Works with all yt-dlp supported sites
- [ ] Typecheck passes

**Priority:** 5

---

### US-006: Preview Processing Pipeline

**Description:** As a developer, I need the translation pipeline to work with partial videos for preview generation.

**Acceptance Criteria:**
- [ ] process_preview function handles 60-second clips
- [ ] Transcription, translation, TTS all work on partial audio
- [ ] Preview video saved to Supabase Storage
- [ ] preview_job status updated to 'completed' with file path
- [ ] Typecheck passes

**Priority:** 6

---

### US-007: Preview Status Endpoint

**Description:** As a user, I want to check the status of my preview so I know when it's ready.

**Acceptance Criteria:**
- [ ] `GET /preview/{preview_id}` returns status, progress, stage
- [ ] When completed, returns preview_url and price_quote
- [ ] Guests can access by session_id match
- [ ] Logged-in users can access by user_id match
- [ ] Typecheck passes

**Priority:** 7

---

### US-008: Preview Video Serving

**Description:** As a user, I want to watch my preview video so I can evaluate the translation quality.

**Acceptance Criteria:**
- [ ] `GET /preview/{preview_id}/video` streams the preview file
- [ ] Generates signed URL from Supabase Storage
- [ ] URL expires after 1 hour
- [ ] Only accessible to preview owner (session_id or user_id)
- [ ] Typecheck passes

**Priority:** 8

---

### US-009: User Authentication - Signup

**Description:** As a guest, I want to create an account so I can purchase full translations.

**Acceptance Criteria:**
- [ ] `POST /auth/signup` accepts email, password
- [ ] Creates user in Supabase Auth
- [ ] Creates profile record with tos_accepted_at timestamp
- [ ] Returns session token
- [ ] Validates email format and password strength
- [ ] Typecheck passes

**Priority:** 9

---

### US-010: User Authentication - Login

**Description:** As a returning user, I want to log in so I can access my account.

**Acceptance Criteria:**
- [ ] `POST /auth/login` accepts email, password
- [ ] Returns session token on success
- [ ] Returns 401 on invalid credentials
- [ ] Typecheck passes

**Priority:** 10

---

### US-011: User Authentication - Google OAuth

**Description:** As a user, I want to sign in with Google for faster account creation.

**Acceptance Criteria:**
- [ ] `GET /auth/google` redirects to Google OAuth
- [ ] `GET /auth/callback` handles OAuth callback
- [ ] Creates profile if new user
- [ ] Returns session token
- [ ] Typecheck passes

**Priority:** 11

---

### US-012: Authentication Middleware

**Description:** As a developer, I need middleware to protect routes that require authentication.

**Acceptance Criteria:**
- [ ] JWT verification middleware created
- [ ] Extracts user_id from valid token
- [ ] Returns 401 for missing/invalid token
- [ ] `get_current_user` dependency for FastAPI routes
- [ ] Typecheck passes

**Priority:** 12

---

### US-013: Stripe Checkout Session Creation

**Description:** As a logged-in user, I want to pay for a full translation so I can get the complete video.

**Acceptance Criteria:**
- [ ] `POST /checkout/create` accepts preview_id
- [ ] Requires authentication
- [ ] Creates translation_job record with price_cents
- [ ] Creates Stripe Checkout Session with correct amount
- [ ] Returns checkout_url for redirect
- [ ] Typecheck passes

**Priority:** 13

---

### US-014: Stripe Webhook Handler

**Description:** As a developer, I need to handle Stripe webhooks to confirm payment and start processing.

**Acceptance Criteria:**
- [ ] `POST /webhook/stripe` receives Stripe events
- [ ] Verifies webhook signature
- [ ] On checkout.session.completed: updates translation_job payment_status to 'paid'
- [ ] On payment success: triggers full translation processing
- [ ] Typecheck passes

**Priority:** 14

---

### US-015: Full Translation Processing

**Description:** As a developer, I need to process full video translations after payment confirmation.

**Acceptance Criteria:**
- [ ] process_full_translation function created
- [ ] Reuses preview transcription for first 60 seconds if available
- [ ] Processes entire video duration
- [ ] Uploads output to Supabase Storage
- [ ] Updates translation_job status and output path
- [ ] Typecheck passes

**Priority:** 15

---

### US-016: Translation Job Status Endpoint

**Description:** As a user, I want to check the status of my paid translation.

**Acceptance Criteria:**
- [ ] `GET /jobs/{job_id}` returns status, progress, stage
- [ ] Requires authentication
- [ ] Only returns jobs owned by current user
- [ ] When completed, returns download_url
- [ ] Typecheck passes

**Priority:** 16

---

### US-017: Translation Download Endpoint

**Description:** As a user, I want to download my completed translation.

**Acceptance Criteria:**
- [ ] `GET /jobs/{job_id}/download` returns signed URL
- [ ] Requires authentication
- [ ] Verifies job belongs to current user
- [ ] Verifies job status is 'completed'
- [ ] URL expires after 24 hours
- [ ] Typecheck passes

**Priority:** 17

---

### US-018: User Jobs List Endpoint

**Description:** As a user, I want to see all my translation jobs in one place.

**Acceptance Criteria:**
- [ ] `GET /jobs` returns list of user's translation jobs
- [ ] Requires authentication
- [ ] Includes status, video_title, created_at, download availability
- [ ] Ordered by created_at descending
- [ ] Typecheck passes

**Priority:** 18

---

### US-019: IP-Based Rate Limiting

**Description:** As a developer, I need to prevent preview abuse by limiting requests per IP.

**Acceptance Criteria:**
- [ ] Rate limiting middleware added
- [ ] Limits preview requests to 5 per hour per IP
- [ ] Returns 429 Too Many Requests when exceeded
- [ ] Does not affect authenticated users' paid jobs
- [ ] Typecheck passes

**Priority:** 19

---

### US-020: Frontend - Video Info Display

**Description:** As a user, I want to see video details after pasting a URL.

**Acceptance Criteria:**
- [ ] Input field for video URL
- [ ] On paste/submit, calls /video-info endpoint
- [ ] Displays: thumbnail, title, duration, price quote
- [ ] Shows "Preview FREE" button
- [ ] Typecheck passes
- [ ] Verify in browser using MCP Toolkit browser tools

**Priority:** 20

---

### US-021: Frontend - Language Selection

**Description:** As a user, I want to select a target language for translation.

**Acceptance Criteria:**
- [ ] Language dropdown/grid shown after video info loaded
- [ ] Shows all supported languages
- [ ] Selection required before preview can start
- [ ] Typecheck passes
- [ ] Verify in browser using MCP Toolkit browser tools

**Priority:** 21

---

### US-022: Frontend - Preview Progress

**Description:** As a user, I want to see preview processing progress.

**Acceptance Criteria:**
- [ ] Progress bar shows current stage and percentage
- [ ] Polls /preview/{id} endpoint every 2 seconds
- [ ] Updates UI in real-time
- [ ] Shows error message if preview fails
- [ ] Typecheck passes
- [ ] Verify in browser using MCP Toolkit browser tools

**Priority:** 22

---

### US-023: Frontend - Preview Player

**Description:** As a user, I want to watch my preview video and see the price for full translation.

**Acceptance Criteria:**
- [ ] Video player shows 60-second preview
- [ ] Price displayed prominently: "Full video: $X.XX"
- [ ] "Get Full Video" CTA button
- [ ] Button triggers auth flow if not logged in
- [ ] Typecheck passes
- [ ] Verify in browser using MCP Toolkit browser tools

**Priority:** 23

---

### US-024: Frontend - Auth Modal

**Description:** As a user, I want to create an account or log in to purchase.

**Acceptance Criteria:**
- [ ] Modal appears when "Get Full Video" clicked (if not logged in)
- [ ] Shows Google OAuth button (primary)
- [ ] Shows email/password form (secondary)
- [ ] Toggle between signup and login
- [ ] ToS checkbox required for signup
- [ ] Typecheck passes
- [ ] Verify in browser using MCP Toolkit browser tools

**Priority:** 24

---

### US-025: Frontend - Checkout Redirect

**Description:** As a logged-in user, I want to be redirected to Stripe to complete payment.

**Acceptance Criteria:**
- [ ] "Get Full Video" calls /checkout/create when logged in
- [ ] Redirects to Stripe Checkout URL
- [ ] Shows loading state during redirect
- [ ] Handles errors gracefully
- [ ] Typecheck passes
- [ ] Verify in browser using MCP Toolkit browser tools

**Priority:** 25

---

### US-026: Frontend - Payment Success Page

**Description:** As a user, I want to see confirmation after successful payment.

**Acceptance Criteria:**
- [ ] /success page shows "Payment successful!"
- [ ] Shows translation job status
- [ ] Auto-redirects to job progress view
- [ ] Handles Stripe redirect parameters
- [ ] Typecheck passes
- [ ] Verify in browser using MCP Toolkit browser tools

**Priority:** 26

---

### US-027: Frontend - Job Progress View

**Description:** As a user, I want to see my full translation progress after payment.

**Acceptance Criteria:**
- [ ] Shows processing stages and progress
- [ ] Polls /jobs/{id} endpoint
- [ ] When complete, shows download button
- [ ] Typecheck passes
- [ ] Verify in browser using MCP Toolkit browser tools

**Priority:** 27

---

### US-028: Frontend - User Dashboard

**Description:** As a logged-in user, I want to see all my past translations.

**Acceptance Criteria:**
- [ ] /dashboard page lists all user's jobs
- [ ] Shows: video title, language, status, date
- [ ] Download button for completed jobs
- [ ] Empty state for new users
- [ ] Typecheck passes
- [ ] Verify in browser using MCP Toolkit browser tools

**Priority:** 28

---

## 5. Functional Requirements

- FR-1: Preview videos are limited to 60 seconds regardless of original length
- FR-2: Price calculated at $0.50 per minute (configurable via env var)
- FR-3: First 60 seconds are free; billing starts at second 61
- FR-4: Minimum charge of $0.50 for any billable content
- FR-5: Download links expire after 7 days
- FR-6: Preview files deleted after 24 hours if not converted to paid job
- FR-7: Users can only have one processing job at a time

## 6. Technical Considerations

### Database Changes
- Supabase PostgreSQL with 3 tables: profiles, preview_jobs, translation_jobs
- Row Level Security for data isolation
- Indexes on user_id and status columns

### API Changes
- New endpoints: /video-info, /preview, /auth/*, /checkout/*, /jobs/*
- Existing /translate endpoint deprecated (replaced by preview flow)

### File Storage
- Supabase Storage for preview and translated videos
- Signed URLs for secure access
- Automatic cleanup of expired files

### Dependencies
- supabase-py (Supabase client)
- stripe (payment processing)
- python-jose (JWT handling)

## 7. Design Considerations

- Maintain existing retro-futuristic aesthetic
- Modal for auth (not separate page) to keep users in flow
- Progress indicators consistent with existing VU meter style
- Mobile-responsive for all new components

## 8. Success Metrics

- Preview completion rate > 80%
- Preview-to-purchase conversion rate > 10%
- Payment success rate > 95%
- Average time from preview to purchase < 5 minutes

## 9. Open Questions

- [x] File storage solution → Supabase Storage
- [x] Authentication methods → Google OAuth + Email/password
- [x] Rate limiting approach → IP-based, 5 previews/hour
- [ ] What happens if Stripe webhook fails? (retry logic needed)
- [ ] Should we offer refunds if translation fails? (manual process initially)
