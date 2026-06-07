"""
Advanced FastAPI image-classification backend.

Features
--------
* Loads a modern, accurate CNN (MobileNet_V3_Large by default, EfficientNet_B0
  optional) exactly once via an lru_cache-backed singleton.
* Structured logging of every request, the predicted label and the wall-clock
  inference time.
* /predict  -> JSON: top-k classes, confidence scores, inference time (ms).
* /         -> Modern Tailwind single-page UI with drag-and-drop upload,
                live results and a loading spinner.
* /health   -> Lightweight readiness probe for cloud platforms.
"""

from __future__ import annotations

import io
import logging
import os
import time
from functools import lru_cache
from typing import List, Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from torchvision import models, transforms

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME = os.getenv("MODEL_NAME", "mobilenet_v3_large").lower()  # or efficientnet_b0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB

# Keep the CPU + memory footprint small on free-tier hosts (e.g. Render's 512 MB).
# Single-threaded inference uses noticeably less RAM and avoids CPU overcommit.
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("image-classifier")


# --------------------------------------------------------------------------- #
# Model + weights (cached so they load only once)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_model_bundle() -> Tuple[torch.nn.Module, transforms.Compose, List[str]]:
    """Build, cache and return (model, preprocess, categories).

    The lru_cache guarantees the heavy weights are downloaded / loaded a single
    time for the lifetime of the process, regardless of how many requests hit
    the endpoint concurrently.
    """
    logger.info("Loading model '%s' on device '%s' ...", MODEL_NAME, DEVICE)
    t0 = time.perf_counter()

    if MODEL_NAME == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        # Default: MobileNet_V3_Large — modern, fast and accurate.
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = models.mobilenet_v3_large(weights=weights)

    model.eval().to(DEVICE)

    preprocess = weights.transforms()      # the canonical preprocessing for the weights
    categories = list(weights.meta["categories"])

    logger.info(
        "Model ready in %.0f ms (%d classes).",
        (time.perf_counter() - t0) * 1000,
        len(categories),
    )
    return model, preprocess, categories


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def classify_bytes(raw: bytes, top_k: int = 5) -> Tuple[List[dict], float]:
    """Run inference on raw image bytes; return (results, inference_ms)."""
    model, preprocess, categories = get_model_bundle()

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    inference_ms = (time.perf_counter() - t0) * 1000

    top_k = max(1, min(top_k, probs.numel()))
    confidences, indices = torch.topk(probs, top_k)

    results = [
        {
            "rank": rank,
            "label": categories[idx],
            "confidence": round(float(conf), 6),
            "confidence_pct": round(float(conf) * 100, 2),
        }
        for rank, (conf, idx) in enumerate(zip(confidences.tolist(), indices.tolist()), start=1)
    ]
    return results, inference_ms


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Advanced Image Classification API",
    description="Production-ready PyTorch image classifier with a modern web UI.",
    version="1.0.0",
)


@app.on_event("startup")
def _warm_up() -> None:
    """Pre-load the model at boot so the first request isn't slow.

    On very small free-tier hosts you can set WARMUP=0 so the service becomes
    healthy immediately and the model loads lazily on the first request.
    """
    if os.getenv("WARMUP", "1") == "1":
        get_model_bundle()


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "model": MODEL_NAME, "device": str(DEVICE)})


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5) -> JSONResponse:
    """Classify an uploaded image and return the top-k predictions."""
    request_start = time.perf_counter()

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (> {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )

    results, inference_ms = classify_bytes(raw, top_k=top_k)
    total_ms = (time.perf_counter() - request_start) * 1000

    logger.info(
        "predict | file=%s size=%dB top1=%s (%.2f%%) inference=%.1fms total=%.1fms",
        file.filename,
        len(raw),
        results[0]["label"],
        results[0]["confidence_pct"],
        inference_ms,
        total_ms,
    )

    return JSONResponse(
        {
            "filename": file.filename,
            "model": MODEL_NAME,
            "device": str(DEVICE),
            "inference_time_ms": round(inference_ms, 2),
            "total_time_ms": round(total_ms, 2),
            "predictions": results,
        }
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the single-page drag-and-drop UI."""
    return HTMLResponse(INDEX_HTML)


# --------------------------------------------------------------------------- #
# Front-end (Tailwind via CDN, vanilla JS — zero build step)
# --------------------------------------------------------------------------- #
INDEX_HTML = """<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>VisionAI — Image Classifier</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'] },
          colors: { brand: { 50:'#eef2ff',100:'#e0e7ff',500:'#6366f1',600:'#4f46e5',700:'#4338ca' } },
        },
      },
    };
  </script>
  <script>
    // Apply saved theme BEFORE first paint to avoid a flash of the wrong theme.
    (function () {
      const saved = localStorage.getItem('theme');
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (saved === 'dark' || (!saved && prefersDark)) {
        document.documentElement.classList.add('dark');
      }
    })();
  </script>
  <style>
    body { font-family: 'Inter', sans-serif; }
    .spinner { border-top-color: transparent; }
    .fade-in { animation: fadeIn .4s ease both; }
    @keyframes fadeIn { from { opacity:0; transform: translateY(8px); } to { opacity:1; transform:none; } }
    .bar-fill { transition: width .8s cubic-bezier(.22,1,.36,1); }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius: 8px; }
  </style>
</head>
<body class="h-full bg-slate-50 text-slate-800 antialiased transition-colors duration-300
             dark:bg-slate-950 dark:text-slate-100">
  <!-- Top bar -->
  <header class="sticky top-0 z-10 bg-white/80 backdrop-blur border-b border-slate-200
                 dark:bg-slate-900/80 dark:border-slate-800">
    <div class="mx-auto max-w-5xl px-5 h-16 flex items-center justify-between">
      <div class="flex items-center gap-2.5">
        <div class="grid place-items-center h-9 w-9 rounded-xl bg-brand-600 text-white shadow-sm shadow-brand-600/30">
          <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
            <path stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 19.5h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5z" />
          </svg>
        </div>
        <div>
          <p class="font-bold leading-tight tracking-tight">VisionAI</p>
          <p class="text-[11px] text-slate-400 leading-tight">Image Classification</p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <span id="statusBadge" class="inline-flex items-center gap-1.5 rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-500 dark:bg-slate-800 dark:text-slate-400">
          <span class="h-2 w-2 rounded-full bg-slate-300"></span> connecting…
        </span>
        <button id="themeToggle" type="button" title="Toggle dark mode" aria-label="Toggle dark mode"
                class="grid place-items-center h-9 w-9 rounded-xl border border-slate-200 bg-white text-slate-600
                       transition hover:bg-slate-100 active:scale-95
                       dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700">
          <!-- sun (shown in dark mode) -->
          <svg id="iconSun" class="hidden h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.75">
            <path stroke-linecap="round" stroke-linejoin="round" d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z" />
          </svg>
          <!-- moon (shown in light mode) -->
          <svg id="iconMoon" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.75">
            <path stroke-linecap="round" stroke-linejoin="round" d="M21.752 15.002A9.718 9.718 0 0118 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 003 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 009.002-5.998z" />
          </svg>
        </button>
      </div>
    </div>
  </header>

  <main class="mx-auto max-w-5xl px-5 py-10">
    <!-- Hero -->
    <div class="text-center max-w-2xl mx-auto mb-10">
      <h1 class="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white">
        Classify any image in milliseconds
      </h1>
      <p class="mt-3 text-slate-500 dark:text-slate-400">
        Upload a photo and our deep-learning model will identify what it sees, ranked by confidence.
      </p>
    </div>

    <div class="grid lg:grid-cols-5 gap-6 items-start">
      <!-- Left: uploader -->
      <section class="lg:col-span-2 space-y-4">
        <div id="dropzone"
             class="group relative cursor-pointer rounded-2xl border-2 border-dashed border-slate-300 bg-white
                    p-8 text-center shadow-sm transition-all duration-200 hover:border-brand-500 hover:shadow-md
                    dark:border-slate-700 dark:bg-slate-900 dark:hover:border-brand-500">
          <input id="fileInput" type="file" accept="image/*" class="hidden" />
          <div class="grid place-items-center h-14 w-14 mx-auto rounded-2xl bg-brand-50 text-brand-600 transition group-hover:scale-105 dark:bg-brand-500/10">
            <svg class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.75">
              <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5V18a2.25 2.25 0 002.25 2.25h13.5A2.25 2.25 0 0021 18v-1.5M16.5 12L12 7.5 7.5 12M12 7.5V18" />
            </svg>
          </div>
          <p class="mt-4 font-semibold text-slate-700 dark:text-slate-200">Drop your image here</p>
          <p class="text-sm text-slate-400">or <span class="text-brand-600 dark:text-brand-500 font-medium">browse files</span></p>
          <p class="mt-3 text-[11px] text-slate-400">PNG · JPG · WEBP — up to 10&nbsp;MB</p>
        </div>

        <!-- Preview -->
        <div id="previewCard" class="hidden fade-in rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
          <div class="flex items-center gap-4">
            <div class="relative h-20 w-20 shrink-0">
              <img id="preview" alt="preview" class="h-20 w-20 rounded-xl object-cover ring-1 ring-slate-200" />
              <div id="spinner" class="hidden absolute inset-0 grid place-items-center rounded-xl bg-white/70">
                <div class="spinner h-7 w-7 animate-spin rounded-full border-[3px] border-brand-600"></div>
              </div>
            </div>
            <div class="min-w-0 flex-1">
              <p id="fileName" class="truncate font-medium text-slate-700 dark:text-slate-200"></p>
              <p id="fileMeta" class="text-sm text-slate-400"></p>
            </div>
            <button id="resetBtn" title="Remove"
                    class="shrink-0 rounded-lg p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-600 transition dark:hover:bg-slate-800 dark:hover:text-slate-300">
              <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.75">
                <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div id="error" class="hidden fade-in rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-300"></div>
      </section>

      <!-- Right: results -->
      <section class="lg:col-span-3">
        <div id="resultsCard" class="rounded-2xl border border-slate-200 bg-white shadow-sm overflow-hidden dark:border-slate-800 dark:bg-slate-900">
          <div class="flex items-center justify-between border-b border-slate-100 px-5 py-4 dark:border-slate-800">
            <h2 class="font-semibold text-slate-800 dark:text-slate-100">Predictions</h2>
            <span id="timing" class="hidden font-mono text-xs text-slate-400"></span>
          </div>

          <!-- Empty state -->
          <div id="emptyState" class="px-5 py-16 text-center">
            <div class="grid place-items-center h-12 w-12 mx-auto rounded-full bg-slate-100 text-slate-300 dark:bg-slate-800 dark:text-slate-600">
              <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <p class="mt-3 text-sm text-slate-400">Your results will appear here once you upload an image.</p>
          </div>

          <!-- Results list -->
          <ul id="predList" class="hidden divide-y divide-slate-100 dark:divide-slate-800"></ul>

          <div id="resultFooter" class="hidden border-t border-slate-100 px-5 py-4 dark:border-slate-800">
            <button id="againBtn"
                    class="w-full rounded-xl bg-brand-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm
                           shadow-brand-600/30 transition hover:bg-brand-700 active:scale-[.99]">
              Classify another image
            </button>
          </div>
        </div>
      </section>
    </div>

    <p class="mt-10 text-center text-xs text-slate-400">
      Powered by <span class="font-medium text-slate-500">PyTorch</span> &amp;
      <span class="font-medium text-slate-500">FastAPI</span> · runs locally on your machine
    </p>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const dropzone = $('dropzone'), fileInput = $('fileInput');
    const previewCard = $('previewCard'), preview = $('preview');
    const fileName = $('fileName'), fileMeta = $('fileMeta'), spinner = $('spinner');
    const predList = $('predList'), timing = $('timing');
    const emptyState = $('emptyState'), resultFooter = $('resultFooter');
    const errorBox = $('error'), resetBtn = $('resetBtn'), againBtn = $('againBtn');
    const statusBadge = $('statusBadge');

    // --- dark / light mode toggle ---------------------------------------- //
    const themeToggle = $('themeToggle'), iconSun = $('iconSun'), iconMoon = $('iconMoon');

    function syncThemeIcons() {
      const dark = document.documentElement.classList.contains('dark');
      iconSun.classList.toggle('hidden', !dark);   // sun visible only in dark mode
      iconMoon.classList.toggle('hidden', dark);   // moon visible only in light mode
    }
    syncThemeIcons();

    themeToggle.addEventListener('click', () => {
      const nowDark = document.documentElement.classList.toggle('dark');
      localStorage.setItem('theme', nowDark ? 'dark' : 'light');
      syncThemeIcons();
    });

    // --- model status badge ---------------------------------------------- //
    fetch('/health').then(r => r.json()).then(d => {
      statusBadge.innerHTML =
        `<span class="h-2 w-2 rounded-full bg-emerald-500"></span> ${d.model} · ${d.device}`;
      statusBadge.className =
        'inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400';
    }).catch(() => {
      statusBadge.innerHTML = `<span class="h-2 w-2 rounded-full bg-red-500"></span> offline`;
    });

    // --- events ---------------------------------------------------------- //
    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });
    resetBtn.addEventListener('click', resetAll);
    againBtn.addEventListener('click', () => { resetAll(); fileInput.click(); });

    ['dragenter', 'dragover'].forEach(evt =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add('border-brand-500', 'bg-brand-50', 'scale-[1.01]');
      }));
    ['dragleave', 'drop'].forEach(evt =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove('border-brand-500', 'bg-brand-50', 'scale-[1.01]');
      }));
    dropzone.addEventListener('drop', (e) => { if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });

    // --- helpers --------------------------------------------------------- //
    function humanSize(b) {
      if (b < 1024) return b + ' B';
      if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
      return (b / 1048576).toFixed(2) + ' MB';
    }

    function resetAll() {
      previewCard.classList.add('hidden');
      errorBox.classList.add('hidden');
      predList.classList.add('hidden');
      resultFooter.classList.add('hidden');
      timing.classList.add('hidden');
      emptyState.classList.remove('hidden');
      predList.innerHTML = '';
      fileInput.value = '';
    }

    async function handleFile(file) {
      errorBox.classList.add('hidden');
      predList.classList.add('hidden');
      resultFooter.classList.add('hidden');
      emptyState.classList.remove('hidden');

      if (!file.type.startsWith('image/')) { showError('Please choose a valid image file.'); return; }

      preview.src = URL.createObjectURL(file);
      fileName.textContent = file.name;
      fileMeta.textContent = humanSize(file.size) + ' · ' + (file.type || 'image');
      previewCard.classList.remove('hidden');
      spinner.classList.remove('hidden');

      const form = new FormData();
      form.append('file', file);

      try {
        const res = await fetch('/predict?top_k=5', { method: 'POST', body: form });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail || 'Request failed');
        }
        renderResults(await res.json());
      } catch (err) {
        showError(err.message);
      } finally {
        spinner.classList.add('hidden');
      }
    }

    function renderResults(data) {
      emptyState.classList.add('hidden');
      predList.innerHTML = '';
      timing.textContent = `${data.inference_time_ms} ms`;
      timing.classList.remove('hidden');

      data.predictions.forEach((p, i) => {
        const top = i === 0;
        const li = document.createElement('li');
        li.className = 'px-5 py-4 fade-in' + (top ? ' bg-brand-50/40 dark:bg-brand-500/10' : '');
        li.style.animationDelay = (i * 60) + 'ms';
        li.innerHTML = `
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center gap-2.5 min-w-0">
              <span class="grid place-items-center h-6 w-6 shrink-0 rounded-md text-xs font-bold
                           ${top ? 'bg-brand-600 text-white' : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400'}">${i + 1}</span>
              <span class="truncate font-medium capitalize ${top ? 'text-slate-900 dark:text-white' : 'text-slate-600 dark:text-slate-300'}">${p.label}</span>
              ${top ? '<span class="shrink-0 rounded-full bg-brand-100 px-2 py-0.5 text-[10px] font-semibold text-brand-700 dark:bg-brand-500/20 dark:text-brand-100">TOP MATCH</span>' : ''}
            </div>
            <span class="shrink-0 font-mono text-sm ${top ? 'text-brand-700 dark:text-brand-100 font-semibold' : 'text-slate-500 dark:text-slate-400'}">${p.confidence_pct.toFixed(2)}%</span>
          </div>
          <div class="h-2 w-full rounded-full bg-slate-100 overflow-hidden dark:bg-slate-800">
            <div class="bar-fill h-full rounded-full ${top ? 'bg-brand-600' : 'bg-slate-300 dark:bg-slate-600'}" style="width:0%"></div>
          </div>`;
        predList.appendChild(li);
        // animate bar after paint
        requestAnimationFrame(() => {
          li.querySelector('.bar-fill').style.width = Math.max(p.confidence_pct, 1.5) + '%';
        });
      });

      predList.classList.remove('hidden');
      resultFooter.classList.remove('hidden');
    }

    function showError(msg) {
      errorBox.textContent = msg;
      errorBox.classList.remove('hidden');
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
