# VisionScribe — Image Captioning (CNN + LSTM)

An image captioning system that generates natural language descriptions for
photos. A frozen ResNet-50 CNN extracts image features, which are fed into
an LSTM decoder that generates captions word by word (trained with teacher
forcing). Served via FastAPI and packaged in Docker.

Built for NextBridge Summer Internship 2026 — AI/ML Track, Task 4.

---

## 1. Project Structure

```
VisionScribe/
├── app/
│   ├── main.py            # FastAPI app, /caption endpoint
│   └── inference.py        # Image preprocessing + caption generation logic
├── src/
│   ├── preprocessing.py    # Vocabulary: tokenization, numericalization, padding
│   ├── dataset.py          # Dataset split + PyTorch CaptionDataset
│   ├── encoder.py           # CNNEncoder (ResNet-50 feature extraction)
│   ├── decoder.py           # LSTMDecoder (training forward pass + greedy generate)
│   ├── train.py             # Training loop (Step 5)
│   └── evaluate.py          # BLEU-1–4 evaluation on the test set (Step 6)
├── data/
│   ├── raw/                 # Flickr8k images + captions.txt (gitignored)
│   └── processed/           # train/val/test caption CSV splits (gitignored)
├── features/                 # Cached ResNet-50 feature vectors (gitignored)
├── models/                   # Saved decoder_best.pt, vocabulary.json, config.json (gitignored)
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── pyproject.toml
```

---

## 2. Setup

### Environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or with `uv`:
```bash
uv venv
uv pip install -r requirements.txt
```

### Dataset

Download [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place it at:
```
data/raw/Images/          # 8,091 .jpg files
data/raw/captions.txt     # image,caption columns
```

---

## 3. Running the Pipeline

### Step 1 — Dataset split
```bash
cd src
python dataset.py
```
Splits images 80/10/10 (train/val/test) by **image**, not by caption, and saves
`data/processed/{train,val,test}_captions.csv`.

### Step 2 — Feature extraction (cache CNN features)
Run the ResNet-50 feature extraction (see `encoder.py`) to populate `features/`
with one `.pt` file per image. This only needs to run once — training reads
these cached features instead of re-running the CNN every epoch.

### Step 3 — Training
```bash
python train.py
```
Trains the LSTM decoder with teacher forcing, tracks train/val loss per
epoch, and saves the best checkpoint (lowest val loss) to `models/`:
- `decoder_best.pt` — decoder weights
- `vocabulary.json` — word↔index mappings
- `config.json` — hyperparameters needed to rebuild the model

### Step 4 — Evaluation
```bash
python evaluate.py
```
Runs greedy decoding on the held-out test set and reports BLEU-1 through
BLEU-4 against the reference captions, plus sample generated captions.

---

## 4. Running the API Locally

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

**Example request:**
```bash
curl -X POST "http://127.0.0.1:8000/caption" \
  -F "file=@test_images/sample.jpg"
```


---

## 5. Docker

### Build
```bash
docker build -t image-captioning-api .
```

### Run
```bash
docker run -p 8000:8000 image-captioning-api
```

### Test
```bash
curl -X POST "http://127.0.0.1:8000/caption" \
  -F "file=@test_images/sample.jpg"
```

### Pull from registry
```bash
docker pull <your-username>/image-captioning-api
```

---

## 6. Model Details

| Component | Choice |
|---|---|
| CNN encoder | ResNet-50 (ImageNet pretrained, frozen — used as a fixed feature extractor, not fine-tuned) |
| Feature dimension | 2048 |
| Decoder | Single-layer LSTM |
| Embedding dimension | 256 |
| Hidden dimension | 512 |
| Vocabulary | Built from training captions only, min frequency = 2 |
| Loss | Cross-entropy, padding tokens masked out |
| Decoding (inference) | Greedy (argmax at each step) |

---

## 7. Results

| Metric | Score |
|---|---|
| Best validation loss | ≈ 2.7 |
| Test loss | ≈ 2.9 |
| BLEU-1 | ≈ 0.67 |
| BLEU-2 | ≈ 0.48 |
| BLEU-3 | ≈ 0.34 |
| BLEU-4 | ≈ 0.23 |

The decoder learns meaningful image-to-language associations. The higher
BLEU-1 relative to BLEU-4 is expected: single-word overlap with references
is common, but matching longer 3–4 word sequences exactly is harder and
drops off more sharply — a normal pattern for a single-layer LSTM decoder
without attention.



---

## 8. Notes

- Trained/inference-time image preprocessing must match exactly — the API
  imports the same transform used during feature extraction to avoid the
  classic "train/serve skew" bug.
- The model is loaded once at API startup, not per-request.
- `data/`, `features/`, and `models/*.pt` are gitignored — raw images and
  trained weights are not committed to the repository.