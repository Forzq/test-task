# Nut and Bolt Detection API

Python service that detects and counts `nut` and `bolt` objects in JPG or PNG images. It provides a FastAPI endpoint, Docker packaging, a YOLOv8 fine-tuning script, and an evaluation script for the held-out test split.

## Scope and architecture

```text
JPG/PNG upload -> POST /predict -> image validation -> YOLO inference
    -> confidence/class filtering -> JSON objects + counts
```

The API only exposes the two task classes: `nut` and `bolt`. Detections below the configured confidence threshold and labels outside these classes are not counted. This keeps ambiguous or unsupported model outputs out of the business result; they should be logged and analysed before adding a new class in production.

## Project layout

```text
app/                 FastAPI application and inference layers
scripts/train.py     Fine-tunes pretrained YOLOv8n weights
scripts/evaluate.py  Evaluates best.pt on the held-out test split
data/                Dataset configuration example; downloaded data is ignored by Git
models/              Trained best.pt is placed here before API inference
tests/               HTTP-level tests using a fake detector
```

## Dataset

The original [Bolts and Nuts](https://universe.roboflow.com/kim-sxidz/bolts-and-nuts-vhkyw/dataset/8) dataset is stored in `data/`; the additional [Nuts and Bolts Detector](https://universe.roboflow.com/nutsandbolts/nuts-and-bolts-detector/dataset/2) dataset is stored in `data/dataset2/`. Both use the same class order: `0: bolt`, `1: nut`. The repository's `data/combined.yaml` references both sources without copying or modifying images and labels.

1. Open the dataset page and select **Download Dataset**.
2. Export the dataset in **YOLOv8** format.
3. Extract it directly into `data/`, preserving the `train/`, `valid/`, and `test/` folders.
4. Keep the first export in `data/` and extract the second export into `data/dataset2/`.
5. Generate the combined configuration and validate both sources:

```bash
python scripts/combine_datasets.py
```

The combining script prints the final split sizes and stops if the two positive sources do not have exactly the expected `bolt`/`nut` class order or contain invalid label IDs. When a negative source is supplied, it also verifies one empty label file per negative image.

Do not create augmented copies of validation or test images. Colour augmentation is applied online to training images only: hue, saturation, and brightness are varied in memory while each batch is loaded. This avoids data leakage and does not increase disk usage.

### Negative and hard-negative images

An image with no nut or bolt must have an **empty** YOLO label file, not an `unknown` bounding box. The project prepares a separate `data/negatives/` source and adds it to all three splits. Such images teach the detector to return zero objects instead of forcing a `nut` or `bolt` prediction.

The preparation script automatically downloads two public sources: 128 generic COCO128 scenes and exactly 500 content-unique images sampled across the object categories in [Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02). The Caltech archive checksum is verified, the background catch-all category is excluded, byte-identical files are removed, and every retained image receives an empty YOLO label. No `screw`, `washer`, or generic `unknown` class is introduced.

```bash
python scripts/prepare_negative_dataset.py --replace
python scripts/combine_datasets.py --negatives data/negatives
```

The `--replace` flag is intentional: it rebuilds only the generated `data/negatives/` directory. Downloaded source files remain cached in `data/raw_negatives/`, and `data/negatives/manifest.json` records source URLs and exact split counts. Hard negatives are ordinary unrelated images on which the current model produces a false `nut` or `bolt`; these can later be added through one or more `--generic-source` directories without changing the class list.

## Local installation

Python 3.11 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For a CUDA-enabled RTX 2060 Super, install the PyTorch build appropriate for the installed NVIDIA driver before installing or running Ultralytics. Follow the official PyTorch selector for the exact command, because the CUDA wheel must match the local environment.

## Fine-tuning

The service uses **pretrained YOLOv8n** (`yolov8n.pt`) as the baseline. It is small enough for an RTX 2060 Super while remaining a real object detector that returns boxes and confidence scores.

```bash
python scripts/train.py --data data/combined.yaml --epochs 150 --patience 7 --imgsz 640 --batch 8 --device 0 --name nut_bolt_combined --hsv-h 0.015 --hsv-s 0.5 --hsv-v 0.4
```

The training script enables moderate colour augmentation by default: `--hsv-h 0.015 --hsv-s 0.5 --hsv-v 0.4`. These transforms only affect the training split; validation and test images remain unchanged. Early stopping now uses `--patience 7`: a maximum of 150 epochs is allowed, but training stops after seven epochs without validation improvement and keeps the best validation checkpoint.

Training outputs are saved under `runs/detect/nut_bolt_combined/`. The script prints the exact checkpoint path when it finishes. Copy the best checkpoint for the API:

```powershell
New-Item -ItemType Directory -Force models
Copy-Item runs/detect/nut_bolt_combined/weights/best.pt models/best.pt
```

If CUDA runs out of memory, retry with `--batch 4`. Do not train from scratch for this task; the script fine-tunes public pretrained weights.

## Evaluation

Run evaluation only after training. It uses the `test` split declared in the dataset YAML and reports precision, recall, mAP@50, and mAP@50-95.

```bash
python scripts/evaluate.py --data data/combined.yaml --weights models/best.pt --imgsz 640 --device 0
```

Record the command output in the submission README after the training run. Do not copy metrics published by the dataset author: report only metrics produced by this checkpoint on the held-out test split.

Baseline run, trained with `yolov8n.pt` for 80 epochs on the original dataset only:

| Split | Precision | Recall | mAP@50 | mAP@50-95 |
| --- | ---: | ---: | ---: | ---: |
| Held-out test (82 images, 461 objects) | 0.949 | 0.948 | 0.951 | 0.701 |

The dataset contains a small number of polygon labels mixed with bounding boxes. Ultralytics converts the polygons to boxes and warns about the mixed annotations during detection evaluation. This is an identified data-quality limitation of the public dataset.

## Confidence-threshold calibration

Do not select `CONFIDENCE_THRESHOLD` by visual inspection of one image. After each training run, calibrate it on the validation split, which includes empty-label generic images and any mined false-positive examples:

```bash
python scripts/calibrate_threshold.py --data data/combined.yaml --weights models/best.pt --split val --imgsz 640 --device 0 --batch 16 --min-recall 0.95 --max-negative-image-fpr 0.0
```

The script tests thresholds from 0.25 to 0.95, reports the lowest value that preserves at least 95% recall while allowing no negative image to produce a `nut` or `bolt`, and writes all evidence to `runs/threshold_calibration.json`. If it reports no valid threshold, retraining data—not an arbitrary higher threshold—is required.

Use the reported value to run the API locally:

```powershell
$env:CONFIDENCE_THRESHOLD = "0.65"  # Replace 0.65 with the calibrated value.
uvicorn app.main:app --reload
```

Docker Compose reads the same environment variable, so the calibrated value can be passed without editing source code.

## Run the API locally

The API starts even before weights exist, but `/predict` returns HTTP 503 until `models/best.pt` is present. This makes model deployment failures visible rather than returning misleading results.

```bash
uvicorn app.main:app --reload
```

Check readiness:

```bash
curl http://127.0.0.1:8000/health
```

Run prediction from PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -F "image=@C:\path\to\parts.jpg"
```

Example response:

```json
{
  "objects": [
    {"class": "bolt", "confidence": 0.94, "bbox": [120, 80, 300, 145]},
    {"class": "nut", "confidence": 0.91, "bbox": [340, 95, 415, 175]}
  ],
  "counts": {"nuts": 1, "bolts": 1}
}
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Docker

Place trained weights at `models/best.pt`, then run:

```bash
docker compose up --build
```

The Compose configuration mounts `./models` as read-only into the container. The image intentionally does not bundle weights, which keeps the source repository small and lets deployments provide an approved model artifact.

## Tests

```bash
pytest -q
```

## Known limitations and next steps

- The model quality depends on the coverage and annotation quality of the public training dataset. New backgrounds, fastener types, severe occlusion, blur, or very small objects may lower recall.
- A confidence threshold is a product decision: raising it reduces false positives but can miss true objects. Detections near the threshold should be retained for error analysis or routed to manual review in a production workflow.
- The current API intentionally ignores model classes other than `nut` and `bolt`. Empty-label negatives and threshold calibration handle images with no task object; a generic `unknown` class cannot represent every possible object.
- Production hardening could add request authentication, rate limiting, structured logging, model/version metadata, monitoring of confidence distributions, and asynchronous batch inference.
