# LENS v3 — Backend Setup Guide

## What This Is
Python FastAPI backend powering LENS face recognition.
- **Face Engine:** InsightFace buffalo_l (99.86% accuracy)
- **Embeddings:** 512-dimension vectors, cosine similarity matching
- **Streams:** RTMP → FFmpeg frame extraction → recognition
- **Database:** Supabase (Postgres)
- **Real-time:** WebSocket broadcasts to dashboard

---

## Prerequisites
- Python 3.11+
- FFmpeg installed on server
- Docker + Docker Compose (recommended)
- Supabase account (free tier works)

---

## Setup (Docker — Recommended)

### Step 1 — Clone and configure
```bash
git clone YOUR_REPO
cd lens_backend
cp .env.example .env
# Edit .env with your Supabase credentials
```

### Step 2 — Set up Supabase database
1. Go to supabase.com → your project → SQL Editor
2. Paste and run the entire contents of `schema.sql`
3. Copy your project URL and service_role key into `.env`

### Step 3 — Start everything
```bash
docker-compose up -d
```

This starts:
- **MediaMTX** on ports 1935 (RTMP), 8888 (HLS), 8889 (WebRTC)
- **LENS Backend** on port 8000 (REST + WebSocket)

### Step 4 — Verify it's running
```bash
curl http://localhost:8000/api/health
```
Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "buffalo_l",
  "identity_count": 0,
  "active_streams": 0
}
```

**Note:** First startup downloads the InsightFace buffalo_l model (~300MB).
This only happens once — it's cached in a Docker volume.

---

## Setup (Without Docker)

```bash
# Install FFmpeg
sudo apt-get install ffmpeg   # Ubuntu/Debian
brew install ffmpeg           # macOS

# Install Python dependencies
pip install -r requirements.txt

# Start MediaMTX separately
docker run --network=host bluenviron/mediamtx:latest

# Start backend
cp .env.example .env
# Edit .env with your values
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Connecting Cameras

Any camera that supports RTMP output can push to LENS.

### Camera push URL format:
```
rtmp://YOUR_SERVER_IP:1935/CAMERA_NAME
```

### Examples:
| Camera | RTMP URL |
|--------|----------|
| Front Gate | rtmp://192.168.1.100:1935/front_gate |
| Lobby | rtmp://192.168.1.100:1935/lobby |
| Parking | rtmp://192.168.1.100:1935/parking |

### Browser view URL (HLS):
```
http://YOUR_SERVER_IP:8888/CAMERA_NAME/index.m3u8
```

### How to push from a phone (free):
1. Download **Larix Broadcaster** (iOS or Android)
2. Settings → Connections → Add
3. URL: `rtmp://YOUR_SERVER_IP:1935/my_phone`
4. Press the record button — it goes live instantly

### How to push from OBS Studio:
1. Settings → Stream → Custom
2. Server: `rtmp://YOUR_SERVER_IP:1935`
3. Stream Key: `front_gate` (or any name)
4. Start Streaming

### IP Cameras (Hikvision, Dahua, Reolink):
Most have an RTMP push setting in their web admin panel.
Point it to: `rtmp://YOUR_SERVER_IP:1935/camera_name`

---

## API Reference

### Health Check
```
GET /api/health
```

### Enroll a Face
```
POST /api/enroll
Content-Type: multipart/form-data

Fields:
  full_name (required)
  nin (required)
  id_type (NIN/PASSPORT/STAFF_ID/CUSTOM)
  date_of_birth
  gender
  nationality
  group_tag (staff/vip/watchlist/public)
  notes
  image (file, required)
```

### Add More Angles to Existing Identity
```
POST /api/enroll/{identity_id}/angles
Content-Type: multipart/form-data

Fields:
  image (file, required)
```

### Bulk Enroll from CSV
```
POST /api/enroll/bulk
Content-Type: multipart/form-data

Fields:
  csv_file (file, required)

CSV columns:
  full_name, nin, id_type, date_of_birth, gender,
  nationality, group_tag, notes, photo_url
```

### Identify Faces in Image
```
POST /api/identify
Content-Type: multipart/form-data

Fields:
  image (file, required)
```

### List Identities
```
GET /api/identities?search=name_or_nin&group=staff&limit=100&offset=0
```

### Set Search Targets (Find Mode)
```
POST /api/search/targets
Body: { "identity_ids": ["uuid1", "uuid2"] }
```

### Add Camera
```
POST /api/cameras
Body: {
  "name": "Front Gate",
  "location": "Building A Entrance",
  "rtmp_key": "front_gate"
}
```

### WebSocket Connection
```
ws://YOUR_SERVER:8000/ws
```

Messages you'll receive:
```json
{ "type": "detection", "camera_id": "...", "detections": [...] }
{ "type": "target_found", "camera_name": "...", "identity": {...} }
{ "type": "camera_status", "camera_id": "...", "status": "live|offline" }
{ "type": "targets_updated", "identity_ids": [...] }
```

---

## Tuning Recognition Accuracy

The similarity threshold controls how strict matching is.

| Threshold | Effect |
|-----------|--------|
| 0.35 | Very lenient — catches more but may mis-identify |
| 0.42 | Balanced — recommended default |
| 0.45 | Strict — fewer false positives |
| 0.55 | Very strict — only very confident matches |

To change at runtime, edit `face_engine.py`:
```python
self.similarity_threshold = 0.45
```

For best accuracy:
1. Enroll each person from multiple angles (use `/angles` endpoint)
2. Enroll in similar lighting to where they'll be recognized
3. Use high quality photos (min 200x200px face crop)

---

## Production Deployment (Railway / Render / VPS)

### Railway:
1. Push code to GitHub
2. New project → Deploy from GitHub
3. Add environment variables from `.env`
4. Deploy

### VPS (Ubuntu):
```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Clone and run
git clone YOUR_REPO && cd lens_backend
cp .env.example .env && nano .env
docker-compose up -d

# Open firewall ports
ufw allow 8000   # Backend API
ufw allow 1935   # RTMP
ufw allow 8888   # HLS
```

---

## File Structure
```
lens_backend/
├── main.py              ← FastAPI app, all API routes
├── face_engine.py       ← InsightFace wrapper, matching logic
├── database.py          ← Supabase client, all DB operations
├── stream_processor.py  ← RTMP frame extraction + recognition loop
├── websocket_manager.py ← WebSocket broadcast manager
├── requirements.txt     ← Python dependencies
├── Dockerfile           ← Container definition
├── docker-compose.yml   ← Full stack (MediaMTX + Backend)
├── mediamtx.yml         ← Media server configuration
├── schema.sql           ← Run this in Supabase SQL Editor
└── .env.example         ← Copy to .env and fill in values
```
