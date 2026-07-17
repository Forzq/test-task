# Nut and Bolt Detection API

Python service that detects and counts `nut` and `bolt` objects in JPG or PNG images. It combines YOLO detection with an optional `bolt` / `nut` / `other` crop verifier so unrelated detector proposals can be rejected before they reach the API response.

## Scope and architecture

```text
JPG/PNG upload -> POST /predict -> image validation -> YOLO detector
    -> square crop with context -> bolt/nut/other classifier
    -> reject other/ambiguous crops -> JSON objects + counts
```

The API only exposes the two task classes: `nut` and `bolt`. `other` is an internal verifier class and never appears in JSON. The trained verifier is enabled by default; `ENABLE_CROP_CLASSIFIER=false` provides a detector-only fallback.

Before crop classification, detections smaller than `MIN_BOX_AREA_RATIO=0.001` are rejected. This prevents isolated texture details such as sugar sprinkles from being enlarged into convincing object crops. The threshold is relative to image area, so it works consistently across different upload resolutions.

## Project layout

```text
app/                 FastAPI application and inference layers
scripts/train.py     Fine-tunes pretrained YOLOv8s weights
scripts/prepare_classifier_dataset.py  Creates balanced crops and mines hard negatives
scripts/train_classifier.py            Fine-tunes pretrained YOLOv8n-cls weights
scripts/calibrate_classifier_threshold.py  Calibrates second-stage acceptance
scripts/evaluate.py  Evaluates best.pt on the held-out test split
scripts/select_checkpoint.py  Selects weights by recall and negative-image FPR
scripts/regression_check.py   Checks fixed false-positive/false-negative cases
data/                Dataset configuration example; downloaded data is ignored by Git
models/              Trained best.pt is placed here before API inference
tests/               HTTP-level tests using a fake detector
```

## Dataset

The original [Bolts and Nuts](https://universe.roboflow.com/kim-sxidz/bolts-and-nuts-vhkyw/dataset/8) dataset is stored in `data/`; the additional [Nuts and Bolts Detector](https://universe.roboflow.com/nutsandbolts/nuts-and-bolts-detector/dataset/2) dataset is stored in `data/dataset2/`. Both use the same class order: `0: bolt`, `1: nut`. The generated `data/external_positives/` source adds 1,746 annotated images converted from NPU-BOLT, MVTec Screws, and Bolts/Washers. The final `data/combined.yaml` references all positive and negative sources.

1. Open the dataset page and select **Download Dataset**.
2. Export the dataset in **YOLOv8** format.
3. Extract it directly into `data/`, preserving the `train/`, `valid/`, and `test/` folders.
4. Keep the first export in `data/` and extract the second export into `data/dataset2/`.
5. Prepare the external source and generate the combined configuration:

```bash
python scripts/prepare_external_positive_dataset.py --replace
python scripts/combine_datasets.py --additional data/external_positives --negatives data/negatives --negative-additional data/hard_negatives
```

The converter creates 775 target-centred NPU training crops with recalculated boxes, converts MVTec oriented boxes into enclosing axis-aligned boxes, and preserves source-level split separation. MVTec is licensed CC BY-NC-SA 4.0 and must be replaced before commercial use. The combining script rejects incompatible classes, malformed labels, and non-empty negative labels.

Do not create augmented copies of validation or test images. Colour augmentation is applied online to training images only: hue, saturation, and brightness are varied in memory while each batch is loaded. This avoids data leakage and does not increase disk usage.

### Negative and hard-negative images

An image with no nut or bolt must have an **empty** YOLO label file, not an `unknown` bounding box. The project prepares a separate `data/negatives/` source and adds it to all three splits. Such images teach the detector to return zero objects instead of forcing a `nut` or `bolt` prediction.

The preparation script automatically downloads two public sources: 128 generic COCO128 scenes and exactly 500 content-unique images sampled across the object categories in [Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02). The Caltech archive checksum is verified, the background catch-all category is excluded, byte-identical files are removed, and every retained image receives an empty YOLO label. No `screw`, `washer`, or generic `unknown` class is introduced.

```bash
python scripts/prepare_negative_dataset.py --replace
python scripts/mine_hard_negatives.py --weights models/best.pt --limit 200 --replace
python scripts/combine_datasets.py --additional data/external_positives --negatives data/negatives --negative-additional data/hard_negatives
```

The `--replace` flag rebuilds only a generated output directory. Downloaded files remain cached, and each generated dataset contains a provenance manifest. The current hard-negative source contains 200 unrelated images on which the previous model produced a false `nut` or `bolt` with confidence from 0.504 to 0.908.

Confirmed user-reported failures can be stored in `data/manual_hard_negatives/` with empty labels. The current source contains the donut/sprinkle regression image in the training split only. `combined.yaml` includes this source, while the same image is retained under `tests/fixtures/` for deterministic post-processing regression checks.

### Crop-classifier dataset

The second stage uses the Ultralytics classification directory format: `data/classifier/{train,val,test}/{bolt,nut,other}`. Positive crops come from existing YOLO annotations. The `other` class comes only from images with empty detection labels and combines full images, random regions, and low-confidence proposals produced by the current detector. The latter are hard negatives because they are precisely the unrelated regions YOLO considers bolt- or nut-like.

Build it with the deployed detector checkpoint:

```powershell
.venv313\Scripts\python.exe scripts\prepare_classifier_dataset.py --data data\combined.yaml --weights models\best.pt --device 0 --batch 4 --chunk-size 16 --replace
```

The current generated dataset contains 13,410 square 224×224 crops. Classes are balanced independently within every split:

| Split | bolt | nut | other | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 3,809 | 3,809 | 3,809 | 11,427 |
| val | 374 | 374 | 374 | 1,122 |
| test | 287 | 287 | 287 | 861 |

Of these, 567 `other` crops are current-detector false proposals, 3,119 are random negative regions, and 784 are complete negative scenes. Detector proposals are selected before generic negative crops, guaranteeing that mined hard negatives are retained during class balancing. The exact provenance and source counts are stored in `data/classifier/manifest.json`. No augmented copies are written to validation or test.

## Local installation

Python 3.11 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For a CUDA-enabled RTX 2060 Super, install the PyTorch build appropriate for the installed NVIDIA driver before installing or running Ultralytics. Follow the official PyTorch selector for the exact command, because the CUDA wheel must match the local environment.

## Fine-tuning

The service fine-tunes **pretrained YOLOv8s** (`yolov8s.pt`). Batch size 4 and two loader workers fit the RTX 2060 Super while limiting host RAM usage.

```bash
python scripts/train.py --data data/combined.yaml --model yolov8s.pt --epochs 80 --patience 7 --imgsz 640 --batch 4 --workers 2 --device 0 --name nut_bolt_yolov8s_v2 --save-period 5
```

The training script enables moderate online colour augmentation by default. Validation and test images remain unchanged. Early stopping uses `--patience 7`, and `--save-period 5` retains periodic checkpoints for task-specific selection.

Training outputs are saved under `runs/detect/nut_bolt_yolov8s_v2/`. Select the deployable checkpoint using positive recall and negative-image false-positive rate; the selector copies the winner to `models/best.pt`:

```powershell
python scripts/select_checkpoint.py --data data/combined.yaml --weights-dir runs/detect/nut_bolt_yolov8s_v2/weights --output models/best.pt --device 0 --batch 2
```

The selected checkpoint is `last.pt`: at threshold 0.62 on validation it reached precision 0.966, recall 0.924, and zero false-positive negative images. If CUDA runs out of memory, reduce batch size. Calibration and mining process images in bounded chunks to avoid oversized tensors.

### Crop-classifier fine-tuning

The verifier fine-tunes lightweight pretrained **YOLOv8n-cls** weights. Its custom transform pipeline resizes the already square crop without `RandomResizedCrop`, so a long bolt is not truncated. Moderate colour, rotation, scale, flip, and erasing augmentation is applied only in memory to train images. Early stopping uses patience 7.

```powershell
.venv313\Scripts\python.exe scripts\train_classifier.py --data data\classifier --model yolov8n-cls.pt --epochs 50 --patience 7 --imgsz 224 --batch 32 --workers 2 --device 0 --name crop_verifier_v1
```

Do not enable the verifier until training finishes. Copy the printed best checkpoint and calibrate its acceptance threshold on validation:

```powershell
Copy-Item runs\classify\crop_verifier_v1\weights\best.pt models\classifier_best.pt
.venv313\Scripts\python.exe scripts\calibrate_classifier_threshold.py --data data\classifier --weights models\classifier_best.pt --split val --device 0 --max-other-fpr 0.02 --min-positive-recall 0.90
```

Use the selected threshold printed by the calibration script, then enable the second stage:

```powershell
$env:ENABLE_CROP_CLASSIFIER = "true"
$env:CLASSIFIER_MODEL_PATH = "models/classifier_best.pt"
$env:CLASSIFIER_CONFIDENCE_THRESHOLD = "0.54"
uvicorn app.main:app --reload
```

After adding new manual negatives, fine-tune from the current classifier rather than restarting from generic pretrained weights:

```powershell
.venv313\Scripts\python.exe scripts\train_classifier.py --data data\classifier --model models\classifier_best.pt --epochs 20 --patience 5 --imgsz 224 --batch 32 --workers 2 --device 0 --name crop_verifier_hardneg_v2
```

## Evaluation

Run evaluation only after training. It uses the `test` split declared in the dataset YAML and reports precision, recall, mAP@50, and mAP@50-95.

```bash
python scripts/evaluate.py --data data/combined.yaml --weights models/best.pt --imgsz 640 --device 0
```

Record the command output in the submission README after the training run. Do not copy metrics published by the dataset author: report only metrics produced by this checkpoint on the held-out test split.

Selected YOLOv8s checkpoint on the combined held-out test split:

| Split | Precision | Recall | mAP@50 | mAP@50-95 |
| --- | ---: | ---: | ---: | ---: |
| Held-out test (410 images, 1,770 objects) | 0.946 | 0.943 | 0.965 | 0.798 |

The dataset contains a small number of polygon labels mixed with bounding boxes. Ultralytics converts the polygons to boxes and warns about the mixed annotations during detection evaluation. This is an identified data-quality limitation of the public dataset.

## Confidence-threshold calibration

Do not select `CONFIDENCE_THRESHOLD` by visual inspection of one image. After each training run, calibrate it on the validation split, which includes empty-label generic images and any mined false-positive examples:

```bash
python scripts/calibrate_threshold.py --data data/combined.yaml --weights models/best.pt --split val --imgsz 640 --device 0 --batch 2 --min-recall 0.90 --max-negative-image-fpr 0.0 --output runs/threshold_calibration_yolov8s_v2.json
```

The strict zero-FPR validation threshold is 0.61, with recall 0.927. The API default is intentionally 0.45: it raises validation recall to 0.949, keeps precision at 0.940, and produces a false detection on only 1 of 109 negative validation images. It also passes both fixed regression cases, whereas 0.61 misses the obvious bolt. This trade-off is based on validation and regression evidence rather than one visual example.

Use the reported value to run the API locally:

```powershell
$env:CONFIDENCE_THRESHOLD = "0.45"
uvicorn app.main:app --reload
```

Run fixed regression cases after changing either weights or threshold:

```bash
.venv313\Scripts\python.exe scripts\regression_check.py --weights models\best.pt --classifier-weights models\classifier_best.pt --confidence 0.45 --classifier-confidence 0.54 --min-box-area-ratio 0.001
```

Docker Compose reads the same environment variable, so the calibrated value can be passed without editing source code.

## Run the API locally

The API starts even before weights exist, but `/predict` returns HTTP 503 until `models/best.pt` is present. When `ENABLE_CROP_CLASSIFIER=true`, `models/classifier_best.pt` is also required. `/health` reports whether the verifier is enabled and which classifier path is configured.

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

Place detector weights at `models/best.pt` and, when enabled, classifier weights at `models/classifier_best.pt`, then run:

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
- The crop classifier reduces open-set false positives but cannot mathematically recognise every possible unknown object. Continue collecting production false proposals and regenerate or extend the `other` class for iterative hard-negative mining.
- Production hardening could add request authentication, rate limiting, structured logging, model/version metadata, monitoring of confidence distributions, and asynchronous batch inference.
