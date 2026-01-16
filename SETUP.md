# Universal Video Translator - Setup Guide

## Implementation Status: COMPLETE

All 28 user stories have been implemented:
- Database schema & Supabase client
- Preview flow (60-second free previews)
- User authentication (Google OAuth + email/password)
- Stripe checkout & webhooks
- Full translation processing
- Frontend UI (video info, preview player, auth modal, dashboard)

---

## Environment Setup

### 1. Supabase Setup

**Create project at [supabase.com](https://supabase.com):**

1. Create a new project
2. Go to **Settings > API** and copy:
   - Project URL → `SUPABASE_URL`
   - Service Role Key → `SUPABASE_SERVICE_KEY`

**Run the database migration:**

1. Go to **SQL Editor** in Supabase Dashboard
2. Copy contents of `supabase_migrations/migrations/001_initial_schema.sql`
3. Paste and run

**Create storage buckets:**

1. Go to **Storage** in Supabase Dashboard
2. Create bucket: `previews` (for 60-second preview videos)
3. Create bucket: `translations` (for full translated videos)

**Enable Google OAuth:**

1. Go to **Authentication > Providers**
2. Enable Google provider
3. Add your Google OAuth credentials from [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
4. Set redirect URL: `https://your-domain.com/auth/callback`

---

### 2. Stripe Setup

**Create account at [stripe.com](https://stripe.com):**

1. Go to **Developers > API keys**
   - Copy Secret key → `STRIPE_SECRET_KEY`
   - Use `sk_test_...` for testing, `sk_live_...` for production

2. Go to **Developers > Webhooks**
   - Add endpoint: `https://your-domain.com/webhook/stripe`
   - Select event: `checkout.session.completed`
   - Copy Signing secret → `STRIPE_WEBHOOK_SECRET`

---

### 3. All Environment Variables

```bash
# ===================
# REQUIRED - AI APIs
# ===================
OPENAI_API_KEY=sk-...
REPLICATE_API_TOKEN=r8_...

# ===================
# REQUIRED - Supabase
# ===================
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...

# ===================
# REQUIRED - Stripe
# ===================
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...

# URLs for Stripe redirects
STRIPE_SUCCESS_URL=https://your-domain.com/success
STRIPE_CANCEL_URL=https://your-domain.com/

# ===================
# REQUIRED - OAuth
# ===================
OAUTH_REDIRECT_URL=https://your-domain.com/auth/callback

# ===================
# OPTIONAL - Defaults shown
# ===================
PRICE_PER_MINUTE_CENTS=50        # $0.50 per minute
PREVIEW_DURATION_SECONDS=60      # Free preview length
PREVIEW_RATE_LIMIT=5             # Max previews per window
PREVIEW_RATE_WINDOW=3600         # Rate limit window (1 hour)
MAX_VIDEO_DURATION=600           # Max video length (10 min)
```

---

### 4. Local Testing

```bash
# Install dependencies
pip install -r requirements-cloud.txt

# Create .env file with your credentials
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
REPLICATE_API_TOKEN=r8_...
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_SUCCESS_URL=http://localhost:8000/success
STRIPE_CANCEL_URL=http://localhost:8000/
OAUTH_REDIRECT_URL=http://localhost:8000/auth/callback
EOF

# Load environment variables
export $(cat .env | xargs)

# Run server
python cloud_translate.py
# or
uvicorn cloud_translate:app --reload --host 0.0.0.0 --port 8000

# Open http://localhost:8000
```

**Testing Stripe webhooks locally:**

```bash
# Install Stripe CLI
# https://stripe.com/docs/stripe-cli

# Forward webhooks to localhost
stripe listen --forward-to localhost:8000/webhook/stripe

# Copy the webhook signing secret it displays
# Update STRIPE_WEBHOOK_SECRET in your .env
```

---

### 5. Railway Deployment

1. Push code to GitHub:
   ```bash
   git add -A
   git commit -m "Complete monetization implementation"
   git push origin main
   ```

2. In Railway Dashboard:
   - Connect your GitHub repo
   - Add all environment variables from section 3
   - Update URLs to use your Railway domain:
     - `STRIPE_SUCCESS_URL=https://your-app.up.railway.app/success`
     - `STRIPE_CANCEL_URL=https://your-app.up.railway.app/`
     - `OAUTH_REDIRECT_URL=https://your-app.up.railway.app/auth/callback`

3. Update Stripe webhook endpoint to Railway URL

4. Update Google OAuth redirect URI in Google Cloud Console

---

## Setup Checklist

| Service | What You Need | Done |
|---------|--------------|------|
| **OpenAI** | API key for Whisper | ☐ |
| **Replicate** | API token for Chatterbox TTS | ☐ |
| **Supabase** | Project URL + Service Key | ☐ |
| **Supabase** | Database migration run | ☐ |
| **Supabase** | Storage buckets created (`previews`, `translations`) | ☐ |
| **Supabase** | Google OAuth configured | ☐ |
| **Stripe** | Secret key | ☐ |
| **Stripe** | Webhook endpoint configured | ☐ |
| **Stripe** | Webhook signing secret | ☐ |
| **Google** | OAuth credentials | ☐ |

---

## Pricing Model

- **Free preview**: First 60 seconds
- **Per-minute pricing**: $0.50/minute (configurable)
- **Minimum charge**: $0.50

Examples:
| Video Length | Price |
|-------------|-------|
| 60 seconds | FREE (preview only) |
| 90 seconds | $0.50 (minimum) |
| 5 minutes | $2.00 |
| 10 minutes | $4.50 |

---

## Architecture

```
User Flow:
┌─────────────────────────────────────────────────────────────┐
│  1. Paste video URL → See title, duration, price quote      │
│  2. Click "Preview FREE" → Watch 60-second translated clip  │
│  3. Click "Get Full Video" → Create account (if needed)     │
│  4. Pay via Stripe Checkout → Webhook triggers processing   │
│  5. Download full translated video from dashboard           │
└─────────────────────────────────────────────────────────────┘

Tech Stack:
- Backend: FastAPI (Python)
- Database: Supabase (PostgreSQL)
- Auth: Supabase Auth (Google OAuth + email/password)
- Storage: Supabase Storage
- Payments: Stripe Checkout
- AI: OpenAI Whisper (transcription) + Replicate Chatterbox (TTS)
- Video: yt-dlp (download) + ffmpeg (processing)
```

---

## Troubleshooting

**"Module not found" errors:**
```bash
pip install -r requirements-cloud.txt
```

**Stripe webhook not receiving events:**
- Check webhook URL is correct in Stripe Dashboard
- Verify `STRIPE_WEBHOOK_SECRET` matches the endpoint's signing secret
- For local testing, use `stripe listen --forward-to localhost:8000/webhook/stripe`

**Google OAuth not working:**
- Verify redirect URI matches exactly in Google Cloud Console
- Check `OAUTH_REDIRECT_URL` environment variable
- Ensure Google provider is enabled in Supabase

**Storage upload fails:**
- Verify buckets `previews` and `translations` exist in Supabase Storage
- Check `SUPABASE_SERVICE_KEY` has correct permissions

---

## License Information

Both AI models used are MIT licensed and allow commercial use:
- **OpenAI Whisper**: MIT License
- **ResembleAI Chatterbox**: MIT License (includes built-in watermarking)
