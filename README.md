# Intro-To-AI-Rework
# LeafLens

Potato leaf disease classifier — **early blight**, **late blight**, or
**healthy** — built on a custom PyTorch CNN (`FINALCNN`), served through a
FastAPI backend, with a Lovable-built UI as the frontend.

**Live app:** https://cropguardleaflens.lovable.appS
**Live API:** https://intro-to-ai-rework.onrender.com

## How it's put together

```
leaflens/
├── Leaf_Lens.ipynb          Training notebook (data prep, CNN training, KNN baseline, comparison)
├── backend/
│   ├── app.py                FastAPI app — /health and /predict endpoints
│   ├── model.py               FINALCNN architecture + inference logic
│   ├── labels.py               Class name parsing helpers
│   ├── knowledge.py             Symptoms/causes/treatment/prevention text per disease
│   ├── requirements.txt
│   ├── Dockerfile               Used by Render to build and run the API
│   └── models/
│       ├── leaf_cnn.pt            Trained weights (tracked via Git LFS)
│       └── classes.json            Class name list matching the checkpoint
├── .gitattributes             Git LFS tracking rule for *.pt files
└── README.md
```

The frontend lives in Lovable, not in this repo — it's a hosted, separately
published app that calls the API below.

## Running the notebook (train or retrain the model)

1. Open `Leaf_Lens.ipynb` in Google Colab.
2. Run the cells top to bottom. Section 1 clones the potato leaf dataset,
   Sections 2–3 write and sanity-check the backend files, Section 4 trains
   the CNN and saves `models/leaf_cnn.pt` + `models/classes.json`, Sections
   5–6 run a KNN baseline for comparison.
3. Download both files from Colab's file browser (they'll be under
   `models/` inside the cloned repo folder) and place them into this
   repo's `backend/models/`, replacing any placeholder there.

No ngrok, no manual tunnel setup needed — training and the local API
sanity-check both run self-contained inside Colab.

## Running the backend locally

Requires Python 3.11+ and the trained checkpoint files already sitting in
`backend/models/`.

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Check it's working:

```bash
curl http://localhost:8000/health
# {"status":"ok","architecture":"FINALCNN (PyTorch)","classes":3,"trained":true}
```

If `trained` comes back `false`, the checkpoint files aren't in
`backend/models/` yet — see the notebook section above.

## Cloning this repo (Git LFS required)

The trained model weights (`leaf_cnn.pt`) are stored via **Git LFS**, not
as a normal file. If you clone without LFS set up, that file will come
down as a small text pointer instead of the real weights, and the API
will fail to load the model.

```bash
git lfs install        # once per machine
git clone https://github.com/Pinkraaaa/Intro-to-AI-Group-9.git
cd Intro-to-AI-Group-9
git checkout leaflens-rework
git lfs pull            # only needed if the weights didn't come down automatically
```

Don't have Git LFS installed? Get it from https://git-lfs.com before
cloning, or run `git lfs install` first if you already cloned without it.

## Deploying the backend

The backend is currently deployed on **Render** (free tier) as a Docker
web service, pointed at the `backend/` folder with `backend/Dockerfile`.
Render assigns the port via the `$PORT` environment variable, which the
Dockerfile already reads from — no extra config needed beyond setting
Root Directory to `backend` and Dockerfile Path to `backend/Dockerfile`
when creating the service.

Note: Render's free tier sleeps after inactivity, so the first request
after a quiet period can take 30–60 seconds to respond while it wakes up.

## API reference

**`GET /health`**
```json
{"status": "ok", "architecture": "FINALCNN (PyTorch)", "classes": 3, "trained": true}
```

**`POST /predict`**

Request:
```json
{ "imageDataUrl": "data:image/jpeg;base64,...." }
```

Response:
```json
{
  "disease": "Early blight",
  "confidence": "high",
  "healthy": false,
  "plant": "Potato",
  "symptoms": ["..."],
  "causes": ["..."],
  "treatment": ["..."],
  "prevention": ["..."],
  "summary": "The CNN classifies this as early blight on potato with 92% confidence.",
  "model": "FINALCNN (PyTorch)",
  "trained": true,
  "topK": [{"label": "Potato___Early_blight", "probability": 0.92}]
}
```

CORS is open to all origins by default (`ALLOWED_ORIGINS=*` in `app.py`);
set that environment variable on the deployment host to restrict it to a
specific frontend domain once needed.s