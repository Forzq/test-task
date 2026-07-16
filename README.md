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

Recommended dataset: [Bolts and Nuts by Kim on Roboflow Universe](https://universe.roboflow.com/kim-sxidz/bolts-and-nuts-vhkyw/dataset/1). It is an object-detection dataset with `nut` and `bolt` labels, an existing train/validation/test split, and the CC BY 4.0 licence.

1. Open the dataset page and select **Download Dataset**.
2. Export the dataset in **YOLOv8** format.
3. Extract it directly into `data/`, preserving the `train/`, `valid/`, and `test/` folders.
4. Use the exported `data/data.yaml`. In the downloaded version, class order is `0: bolt`, `1: nut`.

Do not use the validation or test images for manually generated augmentations. The selected dataset version already includes training-only transformations; Ultralytics also performs its standard online augmentations during fine-tuning.

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
python scripts/train.py --data data/data.yaml --epochs 80 --imgsz 640 --batch 8 --device 0
```

Training outputs are saved under `runs/detect/nut_bolt/`. The script prints the exact checkpoint path when it finishes. Copy the best checkpoint for the API:

```powershell
New-Item -ItemType Directory -Force models
Copy-Item runs/detect/nut_bolt/weights/best.pt models/best.pt
```

If CUDA runs out of memory, retry with `--batch 4`. Do not train from scratch for this task; the script fine-tunes public pretrained weights.

## Evaluation

Run evaluation only after training. It uses the `test` split declared in the dataset YAML and reports precision, recall, mAP@50, and mAP@50-95.

```bash
python scripts/evaluate.py --data data/data.yaml --weights models/best.pt --imgsz 640 --device 0
```

Record the command output in the submission README after the training run. Do not copy metrics published by the dataset author: report only metrics produced by this checkpoint on the held-out test split.

Current run, trained with `yolov8n.pt` for 80 epochs on the downloaded dataset:

| Split | Precision | Recall | mAP@50 | mAP@50-95 |
| --- | ---: | ---: | ---: | ---: |
| Held-out test (82 images, 461 objects) | 0.949 | 0.948 | 0.951 | 0.701 |

The dataset contains a small number of polygon labels mixed with bounding boxes. Ultralytics converts the polygons to boxes and warns about the mixed annotations during detection evaluation. This is an identified data-quality limitation of the public dataset.

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
- The current API intentionally ignores classes other than `nut` and `bolt`. A real production system should log these cases, collect representative images, annotate them, and periodically retrain the model.
- Production hardening could add request authentication, rate limiting, structured logging, model/version metadata, monitoring of confidence distributions, and asynchronous batch inference.
