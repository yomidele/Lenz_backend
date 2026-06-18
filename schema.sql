-- ─────────────────────────────────────────────
-- LENS v3 — Supabase Database Schema
-- Run this entire file in Supabase SQL Editor
-- ─────────────────────────────────────────────


-- IDENTITIES TABLE
create table if not exists identities (
  id uuid primary key default gen_random_uuid(),
  full_name text not null,
  nin text,
  id_type text default 'NIN',
  date_of_birth date,
  gender text,
  nationality text,
  photo_url text,
  embedding jsonb,            -- single 512-number array
  embeddings_multi jsonb,     -- array of 512-number arrays (multi-angle)
  age_estimate integer,
  group_tag text default 'public',  -- staff | vip | watchlist | public
  notes text,
  enrolled_at timestamptz default now(),
  is_active boolean default true
);

-- Index for fast name/NIN search
create index if not exists idx_identities_full_name on identities (full_name);
create index if not exists idx_identities_nin on identities (nin);
create index if not exists idx_identities_group on identities (group_tag);
create index if not exists idx_identities_active on identities (is_active);


-- CAMERAS TABLE
create table if not exists cameras (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  location text,
  rtmp_key text not null unique,    -- e.g. "front_gate"
  stream_url text,
  is_active boolean default true,
  added_at timestamptz default now()
);


-- DETECTION LOGS TABLE
create table if not exists detection_logs (
  id uuid primary key default gen_random_uuid(),
  identity_id uuid references identities(id) on delete set null,
  full_name text,
  nin text,
  confidence float,
  camera_id uuid references cameras(id) on delete set null,
  camera_name text,
  snapshot_url text,
  age_estimate integer,
  gender text,
  detected_at timestamptz default now()
);

-- Index for fast log queries
create index if not exists idx_logs_identity on detection_logs (identity_id);
create index if not exists idx_logs_camera on detection_logs (camera_id);
create index if not exists idx_logs_detected_at on detection_logs (detected_at desc);


-- ─────────────────────────────────────────────
-- ROW LEVEL SECURITY (Recommended for production)
-- ─────────────────────────────────────────────

-- Enable RLS on all tables
alter table identities enable row level security;
alter table cameras enable row level security;
alter table detection_logs enable row level security;

-- Allow all operations for authenticated users
-- (In production, refine these by role)
create policy "Authenticated users can read identities"
  on identities for select using (auth.role() = 'authenticated');

create policy "Authenticated users can insert identities"
  on identities for insert with check (auth.role() = 'authenticated');

create policy "Authenticated users can update identities"
  on identities for update using (auth.role() = 'authenticated');

create policy "Authenticated users can delete identities"
  on identities for delete using (auth.role() = 'authenticated');

create policy "Authenticated users can manage cameras"
  on cameras for all using (auth.role() = 'authenticated');

create policy "Authenticated users can manage logs"
  on detection_logs for all using (auth.role() = 'authenticated');


-- ─────────────────────────────────────────────
-- OPTIONAL: Use service role key in backend
-- to bypass RLS (easier for backend-only access)
-- Set SUPABASE_KEY to service_role key in .env
-- ─────────────────────────────────────────────
