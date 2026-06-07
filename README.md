---
title: AI Image Classifier
emoji: 🖼️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# 🖼️ Advanced Image Classification Pipeline

A **production-ready PyTorch image classifier** with a modern FastAPI backend, a
beautiful drag-and-drop web UI (Tailwind CSS), a polished CLI, and a fully
optimized multi-stage Docker build ready for free / cheap cloud hosting.

> Uses **MobileNet_V3_Large** (default) or **EfficientNet_B0** — both more modern,
> faster and more accurate than the classic ResNet-18, pretrained on ImageNet (1000 classes).

---

## ✨ Features

- ⚡ **FastAPI** backend with async upload handling
- 🧠 Modern CNN with **singleton model caching** (loads weights only once)
- 📊 `/predict` returns **top-k classes, confidence scores & inference time (ms)**
- 🎨 **Drag-and-drop web UI** with live results, confidence bars & loading spinner
- 🖥️ **CLI tool** (`predict.py`) with a clean ASCII results table
- 📝 Structured **request + timing logging**
- 🐳 **Multi-stage Dockerfile** (CPU-only torch wheels, slim final image, non-root user)
- 🩺 `/health` endpoint for cloud readiness probes

---

## 📁 Directory Structure

```
Image-Classification-Pipeline/
├── app/
│   ├── __init__.py
│   └── main.py            # FastAPI backend + Tailwind single-page UI
├── predict.py             # Advanced argparse CLI classifier
├── requirements.txt       # Pinned dependencies
├── Dockerfile             # Optimized multi-stage build (python:3.10-slim)
├── .dockerignore
├── .gitignore
└── README.md              # You are here
```

---

## 🚀 Local Setup (Windows)

> Requires **Python 3.10+**. Commands shown for **PowerShell**.

### 1. Clone / open the project

```powershell
cd D:\Image-Classification-Pipeline
```

### 2. Create & activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If activation is blocked by execution policy, run once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 💡 For a **smaller, CPU-only** PyTorch install (recommended for laptops/servers
> without a GPU):
> ```powershell
> pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
> ```

### 4. Run the web server

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Now open **http://localhost:8000** in your browser and drag in an image. 🎉

---

## 🖥️ CLI Usage

Classify any image directly from the terminal:

```powershell
python predict.py path\to\cat.jpg
python predict.py path\to\cat.jpg --top_k 10
python predict.py path\to\cat.jpg --model efficientnet_b0
```

### Sample output

```
  Model : mobilenet_v3_large
  Image : path\to\cat.jpg

+------+------------------------------+------------+------------------------+
| #    | Class                        | Conf.      | Confidence             |
+------+------------------------------+------------+------------------------+
| 1    | tabby                        |  74.18%    | ################...... |
| 2    | tiger cat                    |  18.55%    | ####.................. |
| 3    | Egyptian cat                 |   4.92%    | #..................... |
| 4    | lynx                         |   0.81%    | ...................... |
| 5    | tiger                        |   0.22%    | ...................... |
+------+------------------------------+------------+------------------------+

  Top prediction : tabby (74.18%)
  Inference time : 41.3 ms
```

---

## 🔌 API Usage

### `POST /predict`

```powershell
curl -X POST "http://localhost:8000/predict?top_k=5" -F "file=@cat.jpg"
```

```json
{
  "filename": "cat.jpg",
  "model": "mobilenet_v3_large",
  "device": "cpu",
  "inference_time_ms": 41.3,
  "total_time_ms": 58.7,
  "predictions": [
    { "rank": 1, "label": "tabby", "confidence": 0.7418, "confidence_pct": 74.18 },
    { "rank": 2, "label": "tiger cat", "confidence": 0.1855, "confidence_pct": 18.55 }
  ]
}
```

### `GET /health`

```json
{ "status": "ok", "model": "mobilenet_v3_large", "device": "cpu" }
```

### Interactive docs

FastAPI auto-generates Swagger UI at **http://localhost:8000/docs**.

---

## 🐳 Docker

Build and run locally:

```powershell
docker build -t image-classifier .
docker run -p 8000:8000 image-classifier
```

Then visit **http://localhost:8000**.

> The build uses a **multi-stage** approach: dependencies are compiled in a
> builder stage, and only the slim virtualenv + app are copied into the final
> `python:3.10-slim` runtime. CPU-only torch wheels keep the image small and
> free-tier friendly.

---

## ☁️ Production Deployment Guides

### a) Hugging Face Spaces (Docker Space)

1. Create a new Space → **SDK: Docker** → **Blank** template.
2. Push this repo to the Space (it ships with a `Dockerfile`).
   ```powershell
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
3. **Important:** HF Spaces routes traffic to **port 7860** by default. Either:
   - Add this line to your `Dockerfile` and update the `CMD` port:
     ```dockerfile
     ENV PORT=7860
     CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
     ```
   - **or** add `app_port: 8000` to the Space's `README.md` YAML front-matter:
     ```yaml
     ---
     title: AI Image Classifier
     sdk: docker
     app_port: 8000
     ---
     ```
4. The Space builds automatically and serves your UI publicly. ✅

---

### b) Render.com (Web Service via GitHub)

1. Push your repo to GitHub (see Git section below).
2. On [render.com](https://render.com): **New → Web Service → Connect your repo**.
3. Settings:
   - **Environment:** `Docker`
   - **Region:** closest to you
   - **Instance type:** `Free`
   - Render auto-detects the `Dockerfile` — no build/start command needed.
4. Render injects a `$PORT` env var. Make uvicorn respect it by changing the
   `CMD` in your `Dockerfile`:
   ```dockerfile
   CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
   ```
5. Click **Create Web Service**. Render builds & deploys on every `git push`. ✅

---

### c) Koyeb / Railway (modern alternatives)

#### Koyeb
1. [app.koyeb.com](https://app.koyeb.com) → **Create App → GitHub** (or Docker image).
2. Builder: **Dockerfile**. Koyeb provides `$PORT` (default `8000`).
3. Set the exposed port to **8000** in the *Ports* section, health check path `/health`.
4. Deploy — Koyeb gives you a public `*.koyeb.app` URL on the free tier. ✅

#### Railway
1. [railway.app](https://railway.app) → **New Project → Deploy from GitHub repo**.
2. Railway detects the `Dockerfile` automatically.
3. Add a public domain under **Settings → Networking → Generate Domain**.
4. Railway sets `$PORT` — use the same `${PORT:-8000}` `CMD` shown for Render. ✅

> 🔑 **General cloud tip:** Most PaaS providers inject a `$PORT` environment
> variable. The portable production command is:
> ```dockerfile
> CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
> ```

---

## 🔗 Git: Initialize & Push to GitHub

```powershell
# 1. Initialize the repository
git init
git add .
git commit -m "Initial commit: advanced PyTorch image classification pipeline"

# 2. Rename default branch to main
git branch -M main

# 3. Create a new EMPTY repo on github.com, then link it:
git remote add origin https://github.com/<your-username>/image-classification-pipeline.git

# 4. Push
git push -u origin main
```

### Subsequent updates

```powershell
git add .
git commit -m "Describe your change"
git push
```

> 💡 Prefer the GitHub CLI? After `gh auth login`:
> ```powershell
> gh repo create image-classification-pipeline --public --source=. --remote=origin --push
> ```

---

## ⚙️ Configuration (Environment Variables)

| Variable          | Default               | Description                                   |
|-------------------|-----------------------|-----------------------------------------------|
| `MODEL_NAME`      | `mobilenet_v3_large`  | `mobilenet_v3_large` or `efficientnet_b0`     |
| `MAX_UPLOAD_BYTES`| `10485760` (10 MB)    | Max accepted upload size                       |
| `PORT`            | `8000`                | Port to bind (when using the portable `CMD`)  |

---

## 🧪 Quick Smoke Test

```powershell
# Terminal 1 — start the server
uvicorn app.main:app --reload

# Terminal 2 — hit the health endpoint
curl http://localhost:8000/health
```

---

## 📜 License

MIT — free to use, modify and deploy. Happy classifying! 🚀
