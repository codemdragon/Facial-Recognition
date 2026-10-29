# CPLC Face Search

**Citizens — Police Liaison Committee · Sindh Governor's Secretariat**

A high-performance facial recognition and similarity search platform featuring a modern dark-mode PyQt5 desktop client, Flask REST API server, and optional Docker deployment. Powered by **DeepFace** embeddings and **FAISS** vector indexing with PostgreSQL database logging for audit trails.

---

## Releases & Interface Versions

### Release 1 — Basic Interface
The foundational release for the CPLC Face Search system:
- **PyQt5 Dark Desktop Application**: Complete forensic workstation client with folder indexing, query drag-and-drop, real-time threshold filtering, and CSV report export.
- **Flask REST API & Web UI**: Headless server architecture with full endpoints (`/api/search`, `/api/build`, `/api/gpu`, etc.) and web interface.
- **DeepFace & FAISS Engine**: Support for multiple face recognition backends (ArcFace, GhostFaceNet, Facenet) and deep detectors (RetinaFace, MTCNN, OpenCV).
- **Audit Logging**: PostgreSQL connection for persistent search history, index metadata, and multi-tenant operator records.

### Interface V1 — Claude UI Prototype (`Claude UI Test/`)
A specialized investigative web dashboard prototype located in `Claude UI Test/files`:
- **Interactive Multi-Panel Workspace**: Side navigation rail, collapsible search configuration drawer, and dataset multi-selector.
- **Comprehensive Match Review Suite**: Side-by-side identity comparison, metadata occurrence cards, case notes, and review status flagging.
- **Watchlists & Sources**: Watchlist builder and source management wizard.
- **Pure Client-Side Architecture**: Built with lightweight zero-dependency HTML, CSS, and vanilla JS (`app.js`, `data.js`, `styles.css`) for rapid UI iteration and design verification.

---

## Key Features

- **Multi-Folder Image Indexing**: Configure and watch multiple reference folders (e.g., historical records, missing persons, found persons across departments). Folder paths are automatically persisted in `face_app/search_config.json`.
- **Fast Similarity Search**: Powered by FAISS vector clustering and indexing over DeepFace embeddings.
- **Multiple Models & Detectors**: Support for face recognition models (`Facenet`, `GhostFaceNet`, `VGG-Face`, `Facenet512`, `ArcFace`, `SFace`, `OpenFace`, `DeepFace`) and face detectors (`opencv`, `mtcnn`, `retinaface`, `dlib`).
- **Flexible Hardware Acceleration**: Auto-detects and supports GPU acceleration (NVIDIA CUDA) as well as CPU fallback.
- **PyQt5 Dark Desktop GUI**:
  - Interactive image drop zone (drag-and-drop or file picker)
  - Configurable results limit (1–100) and similarity threshold filter
  - Global vs. per-section search modes
  - Visual similarity bars, thumbnail previews, and detailed result cards
  - One-click CSV export of search results
- **PostgreSQL Audit & History Logging**: Automatically stores index build metadata, image section assignments, and query logs (falls back gracefully if PostgreSQL is offline).
- **Web UI & REST API**: Includes Flask web endpoints and web browser interface (`face_app/templates/index.html`) for headless or remote deployments.
- **Docker Support**: Containerized PostgreSQL 16 + Flask API service via `docker-compose.yml`.

---

## System Requirements

- **Operating System**: Windows 10/11 / Windows Server or Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.10 or 3.11 (Python 3.11 recommended)
- **NVIDIA GPU** *(Optional)*: CUDA-compatible GPU with current NVIDIA drivers
- **PostgreSQL** *(Optional)*: v14+ or Docker container (desktop app runs fully without it if DB is offline)

---

## Quick Start (Windows Desktop Application)

### 1. Environment Setup

Open PowerShell or Command Prompt in the repository root:

```batch
# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r face_app\requirements.txt

# If deepface is cloned locally in the repository root:
pip install -e deepface
```

*(Alternatively, run `python install_deps.py` to install and verify dependencies.)*

### 2. Launch the Desktop App

Double-click `launch_desktop.bat` or run:

```batch
launch_desktop.bat
```

Or run directly via Python:

```batch
cd face_app
..\.venv\Scripts\python.exe main_app.py
```

### 3. Login Screen
Upon startup, the **Login** window appears. Authentication is currently in placeholder mode:
- Click **LOGIN** or **Continue without login** to enter the search interface.

---

## Usage Guide (Desktop Application)

### Adding Image Folders & Building the Index

1. In the **Image Folders** panel (left side):
   - Click **＋ Add folder** and select your directory containing reference images (e.g. `Found 1`, `Found 2`).
   - You can add multiple directories.
   - Folders are automatically saved to `face_app/search_config.json`.
2. Under **Index Management**:
   - Choose your embedding **Model** (default: `Facenet`) and face **Detector** (default: `opencv`).
   - Configure **Clusters** (default: `8`) and worker threads (**Workers**, default: `4`).
   - Set **Device** mode (`auto`, `gpu`, or `cpu`).
   - Check **Force full rebuild** if you need to regenerate all embeddings from scratch.
3. Click **Build Index**:
   - A progress bar will track extraction and embedding.
   - Upon completion, status pills at the top indicate total indexed faces and index readiness.

### Performing a Search

1. **Upload Query Photo**:
   - Drag and drop an image onto the **Drop Zone** or click inside to browse.
2. **Configure Settings**:
   - **Results Slider**: Set the maximum number of matches to display (1–100).
   - **Min Score %**: Filter out any matches below a given similarity threshold (e.g., 30%).
   - **Search Mode**: Check **Global (across all sections)** for broad match across all images, or uncheck for section-based matching.
3. **Execute Search**:
   - Click **SEARCH**.
   - Results are rendered in the right-hand table showing:
     - **Thumbnail**
     - **Rank**
     - **Score % (color-coded similarity bar)**
     - **Filename**
     - **Relative Path**
     - **Section / Folder Group**
4. **Inspect & Export**:
   - Click any result row to view the full details and enlarged image in the **Detail Card** below.
   - Click **Export CSV** in the top-right to save the results list.
   - Click **Clear results** or **Clear image** to reset.

---

## Server & Web UI Modes

### Running the Flask Server Locally

To run the Flask backend for browser access or network clients:

- **Standard Launch**:
  ```batch
  start_app.bat
  ```
  *(Starts server at `http://127.0.0.1:5000`)*

- **GPU Network Launch**:
  ```batch
  start_app_gpu.bat
  ```
  *(Sets `FACE_DEVICE=gpu` and binds to `0.0.0.0:5000`)*

- **Manual CLI**:
  ```batch
  cd face_app
  python app.py --host 0.0.0.0 --port 5000
  ```

Open your browser and navigate to `http://localhost:5000` to access the web interface.

---

## Docker Deployment (Headless Server + PostgreSQL)

Docker allows containerised deployment of both the Flask search server and PostgreSQL database.

```bash
# 1. Open docker-compose.yml and mount your image directories under the `app` service volumes:
#    volumes:
#      - /path/to/images/Found_1:/data/Found_1:ro
#      - /path/to/images/Found_2:/data/Found_2:ro

# 2. Launch services
docker compose up -d

# 3. Check container logs
docker compose logs -f
```

- **Web UI & REST API**: `http://<server-ip>:5000`
- **PostgreSQL**: `localhost:5432` (`POSTGRES_DB=cplc_faces`, `POSTGRES_USER=cplc`, `POSTGRES_PASSWORD=cplc`)
- **Adding new image folders**: Simply edit the `volumes:` section of `docker-compose.yml` and re-run `docker compose up -d`.

---

## REST API Reference

The Flask server (`face_app/app.py`) provides the following endpoints:

| Endpoint | Method | Parameters / Body | Description |
|----------|--------|-------------------|-------------|
| `/` | `GET` | None | Serves the web search UI (`index.html`) |
| `/api/gpu` | `GET` | None | Returns GPU detection and runtime memory status |
| `/api/device` | `POST` | `{"device": "auto" \| "gpu" \| "cpu"}` | Sets runtime hardware device preference |
| `/api/listdir` | `GET` | `?path=<dir>` | Directory browser for selecting server-side folders |
| `/api/build` | `POST` | JSON: `db_root`, `model`, `detector`, `clusters`, `workers`, `force` | Triggers background index build |
| `/api/build_status` | `GET` | None | Status and progress of current or last index build job |
| `/api/sections` | `GET` | None | Retrieves all cluster sections and image distributions |
| `/api/search` | `POST` | Multipart: `image` (file), `top_n`, `min_score`, `mode` (`all` or `sections`) | Runs facial recognition match on the query photo |
| `/api/image` | `GET` | `?kind=db\|query&rel=<path>&full=0\|1` | Serves thumbnails or full-size images |
| `/api/db/status` | `GET` | None | Checks PostgreSQL connectivity status |
| `/api/history` | `GET` | `?limit=50` | Retrieves recent search logs from PostgreSQL |

---

## PostgreSQL Schema Details

When PostgreSQL is active (connection configured via `CPLC_PG_DSN`), the app automatically logs data across four tables:

1. **`face_index_meta`**: Stores build configurations (`db_root`, `model`, `detector`, `num_clusters`, `total_faces`, `built_at`).
2. **`face_records`**: Maps every indexed image to its parent index and cluster section.
3. **`search_history`**: Audit log recording every query file name, thresholds, total results, JSON matches, and timestamps.
4. **`users`**: Schema table ready for authentication and operator access control.

> [!NOTE]
> Database operations are non-blocking and fail gracefully. If PostgreSQL is offline, all desktop search and indexing operations remain fully functional.

---

## Configuration & Project Structure

### Configuration Files

| File | Purpose |
|------|---------|
| `face_app/search_config.json` | Persists watched folder paths, result limits, and search settings. Auto-created and updated. |
| `face_app/settings.json` | Stores GPU/CPU hardware preference (`device: "auto" \| "gpu" \| "cpu"`) and cached build parameters. |
| `docker-compose.yml` | Container definitions, environment variables (`CPLC_PG_DSN`, `FACE_DEVICE`), and volume mounts. |

### Directory Layout

```
facial Recog/
├── face_app/
│   ├── main_app.py              # Desktop application entry point (PyQt5)
│   ├── app.py                   # Flask REST API server & web UI handler
│   ├── db.py                    # PostgreSQL connection and audit logging
│   ├── vector_db.py             # DeepFace feature extraction + FAISS indexer
│   ├── gpu.py                   # GPU / CPU hardware detection & configuration
│   ├── requirements.txt         # Python package dependencies
│   ├── search_config.json       # Persisted desktop UI configuration
│   ├── settings.json            # Hardware & last build configuration
│   ├── scripts/
│   │   ├── setup_gpu_windows.bat # GPU environment setup script (Windows)
│   │   └── setup_gpu_linux.sh   # GPU environment setup script (Linux)
│   ├── static/
│   │   └── cplc_logo.png        # CPLC emblem and UI branding
│   ├── templates/
│   │   └── index.html           # Flask web search interface
│   └── ui/
│       ├── __init__.py
│       ├── login_window.py      # Login window UI
│       └── main_window.py       # Main search window UI & worker threads
├── deepface/                    # (Optional) Cloned DeepFace source repository
├── Dockerfile                   # Docker image definition for headless server
├── docker-compose.yml           # Docker Compose stack (App + PostgreSQL)
├── install_deps.py              # Python dependency installation helper
├── launch_desktop.bat           # One-click launcher for Windows desktop app
├── start_app.bat                # One-click launcher for local Flask web server
└── start_app_gpu.bat            # One-click launcher for GPU Flask server
```

---

## Troubleshooting

- **PyQt5 and TensorFlow DLL conflict on Windows**:
  - `main_app.py` and `ui/main_window.py` pre-import `tensorflow` prior to initializing PyQt5 to avoid Windows DLL loader lock conflicts. Ensure you launch via `launch_desktop.bat` or use the repository's `.venv`.
- **OpenCV Errors**:
  - On desktop Windows, `opencv-python==4.10.0.84` is required. In headless Docker environments, `opencv-python-headless` is used.
- **GPU Not Detected**:
  - Check `face_app/settings.json` to verify `device` is set to `"auto"` or `"gpu"`.
  - Ensure CUDA 12 and cuDNN DLLs are accessible on your Windows system `PATH`.
  - Check GPU detection anytime via the status bar in the desktop app or `/api/gpu`.
- **PostgreSQL Offline**:
  - If the database is unreachable, the desktop client will show `PostgreSQL offline` in the header bar. Search and indexing continue working locally without interruption.
