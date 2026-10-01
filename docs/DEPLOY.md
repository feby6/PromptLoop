# Deploying PromptLoop

The app ships as one Docker image: FastAPI serves the API under `/api` and the built
React frontend at `/`. Users paste their own API key in the UI, so **the server needs no
provider keys** — don't set `GROQ_API_KEY` etc. on the host.

Constraints that apply everywhere:

- **Single worker.** Runs live in process memory; more than one worker would split
  them across processes. The Dockerfile already starts one.
- **Runs are lost on restart** (by design: no user data at rest). Free hosts that sleep
  when idle will drop in-flight runs.
- `PROMPTLOOP_MAX_ACTIVE_RUNS` caps concurrent runs (default 4) to protect a small box.

## Option A: Hugging Face Spaces (free, Docker)

1. Create a Space → SDK **Docker** → blank template.
2. Push this repo to the Space (or connect it to GitHub).
3. The Space's `README.md` must start with this header (Spaces reads it for config):
   ```yaml
   ---
   title: PromptLoop
   emoji: 🔁
   sdk: docker
   app_port: 7860
   ---
   ```
   Keep the GitHub README clean by adding the header only on the Space's copy.
4. The Space builds the Dockerfile and serves on port 7860.

## Option B: Render (free web service)

1. Push the repo to GitHub.
2. Render dashboard → **New → Blueprint** → select the repo. `render.yaml` configures a
   free Docker web service with a health check on `/api/health`.
3. Render injects `$PORT`; the container listens on it.

## Option C: frontend on Vercel, backend elsewhere

1. Deploy the backend with A or B.
2. On the backend set `PROMPTLOOP_CORS_ORIGINS=https://<your-app>.vercel.app`.
3. Import `frontend/` into Vercel (framework: Vite) with the environment variable
   `VITE_API_BASE=https://<your-backend-host>`.

## Local production check

```bash
docker build -t promptloop .
docker run --rm -p 7860:7860 promptloop
# open http://localhost:7860
```
