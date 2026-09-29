# Deploying to Vercel

The app talks to your vLLM endpoint directly from API routes, so nothing
Python ships with it and no model weights are downloaded. `inference/` stays
in the repo for running everything locally on the Mac.

## 1. Project settings

Push the repo, then in Vercel: **Add New → Project → import it**, and set

* **Root Directory: `webapp`** — the repo root is the Python project.

Framework preset (Next.js), build command and output are detected.

## 2. Environment variables

Settings → Environment Variables, for Production and Preview:

| name | value |
|---|---|
| `LEGALAI_ENDPOINT` | `https://<id>.endpoints.huggingface.cloud/v1` |
| `HF_TOKEN` | a token allowed to CALL the endpoint — see below |
| `APP_PASSWORD` | the shared password for the demo |

**The token must be able to invoke an Inference Endpoint.** A classic token
with the Read role can. A fine-grained token cannot unless "Make calls to
Inference Endpoints" is ticked; without it every request comes back

    403 ... missing permissions: inference.endpoints.infer.write

and the review fails. Beyond that permission the app needs nothing: it never
writes to the Hub, so do not give it write access.

Environment variable changes only take effect on a new deployment.

## 3. Storage

Storage → **Create Database → Upstash Redis (KV)** → connect it to the
project. It injects `KV_REST_API_URL` and `KV_REST_API_TOKEN`, which
`lib/server/store.ts` picks up. With no Redis configured, the store writes
JSON files under `.data/`, which is what `npm run dev` uses locally — on
Vercel that would silently lose data between requests, so connect the
database before the demo.

## 4. Deploy and check

* `/login` accepts the password and sets a cookie.
* `/api/health` should report `{"model":"ready","served":"contract"}`.
  `served` must be the LoRA module name. If it says the base repo id, the
  endpoint is running without `--enable-lora` and every answer is untuned.
* Upload `test_nda.pdf` and confirm 3 Needs attention, 3 Not addressed,
  11 Meets standard.

## Limits to know

* **60 s per request.** A review runs inside one invocation of
  `/api/reviews/<id>/run` (`maxDuration = 60`, the Hobby ceiling; Pro allows
  300). The test NDA takes ~13 s. A long contract split into several windows
  will exceed it, and the review is then marked interrupted with a *Run
  again* button. For the demo, use contracts that fit one window (roughly
  15k tokens, about 50k characters).
* **Closing the tab kills the run.** The browser holds the request open, so
  the review stops if the tab closes. It is marked interrupted after three
  minutes and can be re-run.
* **Scale-to-zero cold starts.** A sleeping endpoint takes minutes to wake an
  L4, and the first call returns 503; the UI says "the endpoint is waking
  up". Turn scale-to-zero off before a live demo.
* **Contracts leave the machine.** Uploads go to Vercel's function and on to
  your HF endpoint. Fine for test documents; a disclosure decision for a real
  client contract.
* **Pause the endpoint afterwards** — it bills by the hour whether or not
  anyone uses it.

## Local development

    cd webapp
    cp ../.env .env.local       # HF_TOKEN and LEGALAI_ENDPOINT
    npm run dev                 # http://localhost:3000

Leave `APP_PASSWORD` unset locally and the login is skipped. Reviews are
written to `webapp/.data/`, which git ignores.
