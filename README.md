# Class Absentee Checker

A single-purpose tool: take a classroom photo, instantly see who's absent. No
sensitivity slider, no history log — just enroll students once and scan.

## How accuracy is handled (instead of a slider)

Matching uses a fixed, tuned threshold internally. Every detected face lands in
one of three buckets:

- **Confident match** → marked present automatically.
- **Borderline match** → shown in a "🟠 needs review" queue with one-tap
  Confirm / Not present buttons — this only appears when a face is genuinely
  ambiguous, so it doesn't slow down the normal case.
- **No reasonable match** → left unrecognized.

Back-row / low-resolution accuracy comes from stacking several things:

- **Multiple classroom photos per scan** — take one from the front and one from
  the back of the room; results merge automatically. This is the single biggest
  lever for far-away faces.
- **Tiled multi-scale detection** with cross-tile deduplication.
- **Face-crop super-resolution refinement** — any detected face under ~100px gets
  re-cropped from the original photo, upscaled, and re-processed on its own for a
  cleaner embedding.
- **Multiple reference photos per student** at enrollment — matching checks a
  detected face against all of a student's photos and takes the best match.

**Physical limit, honestly:** if the source photo itself is low resolution, no
processing invents detail that isn't there. The app warns if an uploaded photo is
under 1000px wide — a phone camera has plenty of resolution, it's just about
getting a decent shot of the back rows too.

## What's in the app

- **🔍 Check Absentees** — the only tab you need day-to-day. Pick a class, upload
  or capture photo(s), scan, get an absentee list + downloadable CSV.
- **➕ Add Student (Admin)** — enroll a student with one or more reference photos.
- **🏫 Manage Classes & Rosters (Admin)** — create/delete classes, view/remove
  students, add extra reference photos to an existing student, and **backup/restore
  the whole roster database** (see below — important once deployed).

Admin tabs are behind a PIN (sidebar) so a scan-only link can be shared safely
without letting anyone edit the roster.

## Deploying it as a website (Streamlit Community Cloud — free, simplest)

1. Push `app.py`, `database.py`, `face_utils.py`, and `requirements.txt` to a
   GitHub repo (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub,
   click "New app", point it at the repo and `app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   admin_pin = "your-own-pin"
   ```
   (Without this, the app falls back to `1234` and shows a warning in the sidebar
   — fine for testing, not for real use.)
4. Deploy. You'll get a permanent `https://your-app.streamlit.app` link — bookmark
   it, that's your "just open the website" access.

First load will be slow (~1–2 min) while it downloads the face model; after that
it's cached and fast.

### The one real gotcha: storage is not permanent

Streamlit Community Cloud (and most free hosts) reset the app's local filesystem
whenever it restarts — goes to sleep from inactivity, gets redeployed, etc. Since
your roster (`attendance.db`) is a local file, **it can get wiped**, which would
mean re-enrolling every student.

The fix is built into the app: **Manage Classes & Rosters → 💾 Backup / Restore**.
After enrolling students, download a backup (`.db` file, a few KB–MB). If you ever
open the app and the roster is empty, upload that backup file and you're back in
seconds — no re-enrolling.

Practical habit: back up once after you finish enrolling a class, and again any
time you add/remove students. That single file is your whole roster.

### If it feels slow or memory-constrained on the free tier

`buffalo_l` (the current face model) is the more accurate option but heavier. If
the free tier struggles (crashes, very slow scans), two levers in the code:

- Lower `det_size=(1280, 1280)` in `app.py`'s `load_face_model()` to e.g.
  `(960, 960)` — faster, slightly less reach on tiny faces.
- In `face_utils.py`, the `>= 2000` finer-tiling-grid cutoff and `REFINE_MAX_SIDE`
  are the next things to tighten if needed.

If it's consistently too tight, a paid tier on Streamlit Cloud, or a small VPS /
Railway / Render instance with more RAM, removes the ceiling entirely — but for a
single class's worth of photos, the free tier is usually fine.

## File layout

- `app.py` — Streamlit UI
- `face_utils.py` — image enhancement, tiled detection, dedup, face refinement,
  matching logic
- `database.py` — SQLite layer (classes, students, multiple embeddings per student)
- `requirements.txt`

## Local setup

```bash
pip install -r requirements.txt
export ATTENDANCE_ADMIN_PIN="your-own-pin"   # optional locally, required before real use
streamlit run app.py
```

## Note on your existing `attendance.db`

If you're upgrading from an older single-photo-per-student version, this version
auto-migrates it on first run — nothing needs to be re-enrolled.
