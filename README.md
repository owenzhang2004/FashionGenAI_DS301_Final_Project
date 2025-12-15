# FashionGenAI — Project Codebase & Reproducible Colab Setup

End-to-end “text → outfit items → matching catalog images → Google Lens product links” pipeline, packaged as a single Google Colab–runnable notebook (plus a small `services.py` module used by the Gradio/Flask demo).

---

## What this project does

Given a natural-language outfit request (e.g., “oversized camel coat fall streetwear”), the pipeline:

1. **Generates a list of clothing items** from the user’s free-form text using **Gemini** (returns a JSON array of clothing strings).
2. **Retrieves the closest matching catalog image** for each clothing string using **CLIP (SentenceTransformer `clip-ViT-B-32`)** over the Kaggle Fashion Product Images dataset.
3. **Uploads the selected catalog image** to **ImgBB** to get a public URL.
4. **Runs Google Lens product search** through **SerpAPI** to return real product links + prices + thumbnails.
5. Displays results in-table and optionally via a **Gradio demo UI**.

---

## Repository / notebook components

- `FashionGenAI_DS301_Final.ipynb`
  - Main runnable notebook (installs deps, downloads dataset, builds embeddings, runs pipeline, launches demo).
- `services.py` (written by the notebook via `%%writefile`)
  - “Glue” module that exposes `run_full_pipeline()` for the Gradio/Flask frontends.

---

## Experimental setup (reproducibility)

This project is primarily a **retrieval + API orchestration** system (no model training). Reproducibility means:

- **Dataset is fixed**: `paramaggarwal/fashion-product-images-small` is downloaded the same way each run.
- **Embedding model is fixed**: `clip-ViT-B-32` via `sentence-transformers`.
- **Deterministic retrieval**: the same text query against the same embedding index yields consistent nearest-neighbor results (minor GPU/FP differences are possible but typically negligible).
- **External APIs are not deterministic**: Google Lens results can change over time, by region, and by SerpAPI response variations. For best reproducibility:
  - Keep the same `country`, `hl`, and `type` parameters (already set in code).
  - Log timestamps and store returned JSON if you need strict reproducibility for reporting.

---

## Quickstart (Google Colab — recommended)

### 1) Open the notebook
Upload and open `FashionGenAI_DS301_Final.ipynb` in Google Colab.

### 2) Add API keys in Colab Secrets
In Colab (left sidebar) → **Secrets** → add the following key/value pairs:

- `SERPAPI_API_KEY`
- `IMGBB_API_KEY`
- `FASHIONGENAI_KEY`

The notebook reads them using:
- `from google.colab import userdata`
- `userdata.get("SERPAPI_API_KEY")`, etc.

### 3) Install dependencies (run the install cells)
The notebook includes installs for:
- `google-search-results` (SerpAPI client)
- `flask`, `flask-cors`, `pyngrok` (optional demo infra)
- `kaggle` (dataset download)
- `sentence-transformers`, `torch`, `Pillow`, etc.
- `gradio` (demo UI)

### 4) Download dataset (Kaggle)
You’ll be prompted to upload your `kaggle.json`.

Steps the notebook performs:
- Mounts Google Drive
- Uploads `kaggle.json`
- Moves it to a Drive folder and copies it to `~/.kaggle/kaggle.json`
- Runs:
  - `kaggle datasets download -d paramaggarwal/fashion-product-images-small -p /content/data`
  - unzips to `/content/data/fashion_small`

After this, images are expected at:
- `/content/data/fashion_small/images/*.jpg`
and metadata at:
- `/content/data/fashion_small/styles.csv`

### 5) Build embeddings (one-time per runtime)
The notebook:
- Loads all catalog images
- Encodes them with CLIP into a tensor embedding matrix

This step is the main compute cost. Once built, retrieval is fast.

### 6) Run the pipeline
There are two main ways:

**A) Notebook function call**
Use the end-to-end helper:
- `text_to_links(...)` (simple path)
or, in the demo module:
- `run_full_pipeline(user_text, top_k=2, max_results=3)`

**B) Gradio demo**
Run the “GRADIO APP FOR DEMO” section to launch an interactive UI that calls `run_full_pipeline()`.

---

## Environment variables / secrets (required)

| Name | Used for | Where read |
|---|---|---|
| `SERPAPI_API_KEY` | Google Lens product search | `userdata.get("SERPAPI_API_KEY")` |
| `IMGBB_API_KEY` | Upload selected catalog image to public URL | `userdata.get("IMGBB_API_KEY")` |
| `FASHIONGENAI_KEY` | Gemini access (text → clothing list) | `userdata.get("FASHIONGENAI_KEY")` |

---

## Robust error handling (what’s covered)

Implemented checks and safeguards include:
- **Missing SerpAPI key** → raises `RuntimeError` before calling Lens.
- **ImgBB upload failure** → `requests.raise_for_status()` surfaces the HTTP error.
- **Gemini output parsing** → `parse_clothing_output()` attempts `json.loads`, and falls back to extracting the first `[...]` block if extra text appears.
- **Price field variability** → `extract_price()` handles dict/string/extension-based price formats from SerpAPI Lens results.

Recommended best practice if you extend this:
- Wrap API calls in retries with exponential backoff.
- Cache Lens responses for repeat experiments.

---

## Performance notes / optimization knobs

- `top_k` (retrieval): controls how many candidate catalog images to consider.
- `max_results` (Lens results): controls how many product links returned per clothing item.
- Biggest runtime cost is **encoding all images**. If you want faster iteration:
  - Precompute and save embeddings to Drive (e.g., `torch.save(all_img_embs, ...)`)
  - Reload embeddings without re-encoding.

---

## Troubleshooting

**Kaggle download fails**
- Ensure `kaggle.json` is valid and has permissions enabled.
- Confirm it is copied to `/root/.kaggle/kaggle.json` with chmod `600`.

**`SERPAPI_API_KEY` error**
- Confirm the key exists in Colab **Secrets** and is spelled exactly.
- Re-run the cell that loads `userdata.get(...)` after adding secrets.

**ImgBB upload issues**
- Verify `IMGBB_API_KEY` is correct.
- Some images can exceed limits depending on ImgBB account settings; try another image or reduce size.

**Gemini returns non-JSON**
- The parser already attempts to recover, but if it still fails:
  - Strengthen the prompt to “Return ONLY valid JSON array, no prose.”

---

## Usage example (copy/paste prompt)

Try:
- “Oversized camel coat fall outfit with neutral tones, minimal streetwear vibe, comfortable for NYC weather.”

You should see:
1) Gemini-generated clothing list  
2) Selected catalog image per clothing item + similarity scores  
3) Public ImgBB URL for Lens input  
4) Lens product links (title/source/price/thumbnail)

---

## Grading alignment checklist

- **Clean structure**: notebook + `services.py` module separation
- **Documentation**: this README + inline docstrings in key functions
- **Error handling**: key checks for API keys + JSON parsing + HTTP errors
- **Experiment reproducibility**: fixed dataset + fixed embedding model; log/caching guidance for API variability
- **Tutorial**: Colab-first install + secrets + Kaggle dataset workflow + demo run steps

---
