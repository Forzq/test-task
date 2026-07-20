# Nut and Bolt Detection API

Python service that detects and counts `nut` and `bolt` objects in JPG or PNG images. It combines YOLO detection with an optional `bolt` / `nut` / `other` crop verifier so unrelated detector proposals can be rejected before they reach the API response.

## Quick start with Docker

The repository includes the two deployed model files. Training datasets are
not required to run inference.

Requirements: Docker Desktop with Docker Compose.

```bash
git clone --depth 1 --single-branch --branch ready-for-inference \
  https://github.com/Forzq/test-task.git
cd test-task
docker compose up --build
```

Wait until the `api` service becomes healthy, then open:

- Swagger UI: <http://localhost:8000/docs>
- Model readiness: <http://localhost:8000/health>

The health endpoint must return `"status": "ready"`. Submit a JPG or PNG from
Swagger, or use PowerShell:

```powershell
curl.exe -X POST "http://localhost:8000/predict" `
  -F "image=@C:\path\to\parts.jpg;type=image/jpeg"
```

Stop the service with:

```bash
docker compose down
```

The Docker image uses CPU-only PyTorch for a portable submission. A local
CUDA environment is needed only for model training, not for API evaluation.

## Scope and architecture

```text
ApplicationFactory
    -> ModelRuntime (model lifecycle and application settings)
    -> PredictionController (HTTP validation and response mapping)
        -> DetectionService (pipeline orchestration)
            -> ImageDecoder -> YOLO detector
            -> CropCandidateVerifier -> bolt/nut/other classifier
            -> DetectionPostProcessor -> class-aware NMS and counts
```

The application is split into small classes with one responsibility each.
`app/main.py` is the FastAPI composition layer, `app/service.py` coordinates one
prediction, and `app/pipeline.py` contains reusable image, threshold, crop,
classification, and post-processing policies. Public functions and methods use
English NumPy-style docstrings that describe their purpose, inputs, and outputs.

The API only exposes the two task classes: `nut` and `bolt`. `other` is an internal verifier class and never appears in JSON. The trained verifier is enabled by default; `ENABLE_CROP_CLASSIFIER=false` provides a detector-only fallback.

Before crop classification, detections smaller than `MIN_BOX_AREA_RATIO=0.001` are rejected. This prevents isolated texture details such as sugar sprinkles from being enlarged into convincing object crops. The threshold is relative to image area, so it works consistently across different upload resolutions. Detector and verifier thresholds are configured independently for `bolt` and `nut`. Final class-aware NMS removes duplicate boxes only when they have the same final class, so an overlapping nut and bolt are retained.

## Project layout

```text
app/                 FastAPI application and inference layers
scripts/train.py     Fine-tunes pretrained YOLOv8s weights
scripts/prepare_classifier_dataset.py  Creates balanced crops and mines hard negatives
scripts/train_classifier.py            Fine-tunes pretrained YOLOv8n-cls weights
scripts/audit_dataset.py                Audits labels, splits, balance, and duplicates
scripts/refresh_hard_negatives.py       Rebuilds mining and dependent datasets
scripts/calibrate_class_thresholds.py   Calibrates bolt and nut independently
scripts/benchmark_pipeline.py           Benchmarks the complete two-stage API pipeline
scripts/compare_benchmarks.py           Applies promotion gates to two benchmarks
scripts/evaluate.py  Evaluates best.pt on the held-out test split
scripts/select_checkpoint.py  Selects weights by recall and negative-image FPR
scripts/regression_check.py   Checks fixed false-positive/false-negative cases
data/                Dataset configuration example; downloaded data is ignored by Git
models/              Deployed detector and crop-classifier weights
tests/               HTTP-level tests using a fake detector
```

## Dataset

The original [Bolts and Nuts](https://universe.roboflow.com/kim-sxidz/bolts-and-nuts-vhkyw/dataset/8) dataset is stored in `data/`; the additional [Nuts and Bolts Detector](https://universe.roboflow.com/nutsandbolts/nuts-and-bolts-detector/dataset/2) dataset is stored in `data/dataset2/`. Both use the same class order: `0: bolt`, `1: nut`. The generated `data/external_positives/` source contains 634 strictly converted MVTec and Bolts/Washers images. The final `data/combined.yaml` references positive and negative training sources but never the frozen external benchmark.

1. Open the dataset page and select **Download Dataset**.
2. Export the dataset in **YOLOv8** format.
3. Extract it directly into `data/`, preserving the `train/`, `valid/`, and `test/` folders.
4. Keep the first export in `data/` and extract the second export into `data/dataset2/`.
5. Prepare the external source and generate the combined configuration:

```bash
python scripts/prepare_external_positive_dataset.py --replace
python scripts/combine_datasets.py --additional data/external_positives --additional data/arg_fixings_training --negatives data/negatives --negative-additional data/hard_negatives --negative-additional data/manual_hard_negatives
```

The ontology requires one complete physical bolt or nut per box. NPU-BOLT is excluded because its `bolt head`, `bolt side`, and `blur bolt` annotations describe parts rather than complete physical bolts. MVTec maps `type_008` to bolt and `type_007`, `type_009`, `type_010`, and `type_011` to nut; its screw categories remain background. Bolts/Washers contributes complete bolt boxes while bottles and washers remain background. MVTec is licensed CC BY-NC-SA 4.0 and must be replaced before commercial use. The combining script rejects incompatible classes, malformed labels, non-empty negative labels, and any attempt to include `data/external_benchmark`.

ARG_FIXINGS2 is a separate CC BY 4.0 training source with complete Bolt/Nut boxes and Screw/Washer background. Its public Roboflow export requires an API key. Set the key only in the current shell or download the YOLOv8 ZIP manually; never commit it:

```powershell
$env:ROBOFLOW_API_KEY = "YOUR_PRIVATE_KEY"
.venv313\Scripts\python.exe scripts\prepare_arg_fixings_training.py --replace
python scripts\combine_datasets.py --additional data/external_positives --additional data/arg_fixings_training --negatives data/negatives --negative-additional data/hard_negatives --negative-additional data/manual_hard_negatives
```

The converter removes exact overlaps with the frozen benchmark and writes `data/arg_fixings_training/review.csv` for spot-checking the source's model/SAM-assisted annotations. Roboflow train filenames are grouped by their pre-augmentation source name, and only one offline-augmented variant is retained per original image by default; YOLO supplies fresh online augmentation during training.

Three additional CC BY 4.0 Roboflow sources can be imported together: Fastener Object Detection v1, screw-nut-bolt v1, and Stresstest Tool Tray v3. The importer requests the API key through a hidden terminal prompt, keeps it only in process memory, downloads immutable YOLOv8 exports, and never writes the key to the repository:

```powershell
.venv313\Scripts\python.exe scripts\import_roboflow_sources.py --replace
```

`hex-bolt` and `bolt` map to `bolt`; `hex-nut` and `nut` map to `nut` when those classes exist in the immutable export. Fastener Object Detection v1 currently declares `hex-bolt` but not `hex-nut` in its downloaded YOLOv8 YAML, despite the broader project overview, so that source contributes bolt positives only. Screw, washer, needle, scissor, and other source boxes are omitted from the target ontology, so images containing only those objects become empty-label background examples. The importer retains at most one Roboflow offline variant per filename family, removes SHA-256 duplicates against the current training data and frozen benchmark, creates a deterministic 80/10/10 family-level split for the source published without validation/test sets, and updates `data/combined.yaml` only after all checks succeed. It writes complete provenance to `data/roboflow_additions/manifest.json` and a manual spot-check worksheet to `data/roboflow_additions/review.csv`. It does not start model training.

Do not train directly on the complete raw addition. The `screw_nut_bolt` export has a median of 21 objects per scene and a median box area of about `0.0021`, so its 19,492 boxes dominate the loss despite reasonable image-level counts. Generate a separate balanced composition instead:

```powershell
.venv313\Scripts\python.exe scripts\prepare_balanced_detector_dataset.py --replace
.venv313\Scripts\python.exe scripts\audit_dataset.py --data data\combined_balanced.yaml --output runs\dataset_audit_balanced_v7.json
```

The builder keeps complete annotations but caps images from the dense source. It samples positive scenes across class composition, object density, and box scale, then applies class-specific scene quotas chosen to compensate for the older dataset's bolt imbalance. The raw source and `data/combined.yaml` remain unchanged. The generated `data/combined_balanced.yaml` contains 4,179 train images with 10,217 bolt and 10,097 nut boxes, 1,207 negative scenes, no invalid labels, and no exact cross-split leakage. Exact quotas and selected strata are recorded in `data/roboflow_balanced/manifest.json`.

If the balanced positive-source experiment fails the frozen external benchmark, use the imported datasets only as hard negatives. This preserves the deployed detector's positive ontology while adding screw, needle, scissor, and unrelated fastener scenes with empty labels:

```powershell
.venv313\Scripts\python.exe scripts\prepare_balanced_detector_dataset.py --quota-config data\roboflow_hard_negative_quotas.json --output data\roboflow_hard_negative_only --combined-output data\combined_hardneg_v8.yaml --replace
.venv313\Scripts\python.exe scripts\audit_dataset.py --data data\combined_hardneg_v8.yaml --output runs\dataset_audit_hardneg_v8.json
```

The JSON quota configuration explicitly sets every positive-scene quota to zero and retains 500/110/100 imported background scenes in train/validation/test. This is a separate experiment; it does not modify `data/combined.yaml`, `data/combined_balanced.yaml`, or deployed weights.

### Mechanical parts detection v3

The local `data/dataset3` export is the CC BY 4.0 Roboflow `mechanical-parts-detection-glf33-waais/3` source. Its 5,237 exported files use ten classes. The project maps complete `bolt` and `nut` boxes to the two detector classes; bearing, flange, gear, shaft, snap-ring, spacer, spring, and washer boxes become detector background and are retained in `ignored_boxes.jsonl` for future explicit classifier `other` crops.

Do not reference the raw ten-class export directly. Prepare a two-class, family-deduplicated copy and combine it with every previous training source:

```powershell
.venv313\Scripts\python.exe scripts\prepare_mechanical_parts_dataset.py --source data\dataset3 --output data\mechanical_parts_roboflow --existing-data data\combined.yaml --benchmark-images data\external_benchmark\images --validation-percent 10 --max-variants-per-family 1 --seed 42 --replace
.venv313\Scripts\python.exe scripts\combine_datasets.py --primary data --secondary data\dataset2 --additional data\external_positives --additional data\arg_fixings_training --additional data\roboflow_additions --additional data\mechanical_parts_roboflow --negatives data\negatives --negative-additional data\hard_negatives --negative-additional data\manual_hard_negatives --output data\combined_dataset3.yaml
.venv313\Scripts\python.exe scripts\audit_dataset.py --data data\combined_dataset3.yaml --output runs\dataset_audit_dataset3.json
```

Although the publisher export states that no augmentation was applied, the files contain visible offline rotation/noise variants under common pre-Roboflow filename families. The converter retains one representative per family and assigns the family as a unit to a deterministic 90/10 train/validation split. The prepared source contains 4,261 retained families: 3,815 train images and 446 validation images. Its train split contributes 2,241 bolt boxes, 2,993 nut boxes, and 2,502 target-negative scenes. It also converts 239 polygon annotations to enclosing detection boxes without modifying the raw export.

The final `combined_dataset3.yaml` contains all previous sources plus the prepared addition: 9,485 train images, 18,328 bolt boxes, 18,509 nut boxes, and 4,714 negative scenes. The audit reports no invalid labels and no exact cross-split duplicate groups.

The source shares 167 original filename families with the current Mechanical Parts Dataset 2022 benchmark. Once a model is trained on `combined_dataset3.yaml`, that benchmark is retired and must not be used for model selection or claimed as independent. Preserve the deployed checkpoints and build a new source-disjoint benchmark before the final old-versus-new comparison.

Do not create augmented copies of validation or test images. Colour augmentation is applied online to training images only: hue, saturation, and brightness are varied in memory while each batch is loaded. This avoids data leakage and does not increase disk usage.

### Negative and hard-negative images

An image with no nut or bolt must have an **empty** YOLO label file, not an `unknown` bounding box. The project prepares a separate `data/negatives/` source and adds it to all three splits. Such images teach the detector to return zero objects instead of forcing a `nut` or `bolt` prediction.

The preparation script automatically downloads two public sources: 128 generic COCO128 scenes and exactly 500 content-unique images sampled across the object categories in [Caltech 101](https://data.caltech.edu/records/mzrjq-6wc02). The Caltech archive checksum is verified, the background catch-all category is excluded, byte-identical files are removed, and every retained image receives an empty YOLO label. No `screw`, `washer`, or generic `unknown` class is introduced.

```bash
python scripts/prepare_negative_dataset.py --replace
python scripts/mine_hard_negatives.py --weights models/best.pt --limit 200 --replace
python scripts/combine_datasets.py --additional data/external_positives --negatives data/negatives --negative-additional data/hard_negatives --negative-additional data/manual_hard_negatives
```

The `--replace` flag rebuilds only a generated output directory. Downloaded files remain cached, and each generated dataset contains a provenance manifest. The current hard-negative source contains 200 unrelated images on which the previous model produced a false `nut` or `bolt` with confidence from 0.504 to 0.908.

Confirmed user-reported failures can be stored in `data/manual_hard_negatives/` with empty labels. The current source contains the donut/sprinkle regression image in the training split only. `combined.yaml` includes this source, while the same image is retained under `tests/fixtures/` for deterministic post-processing regression checks.

The complete refresh is automated. Run it after producing a new detector so proposals are mined by the model that will actually be deployed:

```powershell
.venv313\Scripts\python.exe scripts\refresh_hard_negatives.py --detector runs\detect\detector_hardneg_v3\weights\best.pt --device 0 --batch 4
```

Use `--skip-mining` to rebuild only the combined and classifier datasets, or `--dry-run` to print every command without changing generated data.

### Crop-classifier dataset

The second stage uses the Ultralytics classification directory format: `data/classifier/{train,val,test}/{bolt,nut,other}`. Positive crops come from existing YOLO annotations. The `other` class comes only from images with empty detection labels and combines full images, random regions, and low-confidence proposals produced by the current detector. The latter are hard negatives because they are precisely the unrelated regions YOLO considers bolt- or nut-like.

Build it with the deployed detector checkpoint:

```powershell
.venv313\Scripts\python.exe scripts\prepare_classifier_dataset.py --data data\combined.yaml --weights models\best.pt --contexts 0.2 1.0 --device 0 --batch 4 --chunk-size 16 --replace
```

The current generated dataset contains 15,822 square 224×224 crops. Every positive and detector-proposal crop is generated at context ratios `0.2` and `1.0`; classes are balanced independently within every split:

| Split | bolt | nut | other | Total |
| --- | ---: | ---: | ---: | ---: |
| train | 4,000 | 4,000 | 4,000 | 12,000 |
| val | 748 | 748 | 748 | 2,244 |
| test | 526 | 526 | 526 | 1,578 |

Of these, 1,134 `other` crops are current-detector false proposals, 3,699 are random negative regions, and 441 are complete negative scenes. Detector proposals are selected before generic negative crops, guaranteeing that mined hard negatives are retained during class balancing. The exact provenance, context, and source counts are stored in `data/classifier/manifest.json`. No stochastic augmented copies are written to validation or test.

#### Source-balanced classifier refresh

The first Dataset 3 classifier over-rejected external positives because Dataset 3 displaced too much of the legacy visual domain. Rebuild a 50/50 source-balanced dataset from the frozen legacy crops and the crops mined by `detector_dataset3_v9`:

```powershell
.venv313\Scripts\python.exe scripts\prepare_source_balanced_classifier_dataset.py --legacy data\classifier --dataset3 data\classifier_dataset3_v9 --output data\classifier_source_balanced_v10 --train-per-source-class 4000 --val-per-source-class 999 --test-per-source-class 572 --seed 42 --replace
```

The generated train split contains 4,000 legacy and 4,000 source-unique Dataset 3 crops for each of `bolt`, `nut`, and `other`. Validation contains 999 per source/class and test contains 572. Exact cross-source overlaps are removed before Dataset 3 sampling; the final 35,426 crops contain no exact duplicate, cross-split, or cross-class groups. Full provenance and fingerprint are stored in `data/classifier_source_balanced_v10/manifest.json`.

### Dataset audit

Run the repeatable audit before each training cycle:

```powershell
.venv313\Scripts\python.exe scripts\normalize_detection_labels.py --data data\combined.yaml --apply
.venv313\Scripts\python.exe scripts\audit_dataset.py --data data\combined.yaml --output runs\dataset_audit.json
```

Before ARG_FIXINGS2 is added, the strict combined detector dataset has 3,048 train, 388 validation, and 377 test images. Train contains 6,266 bolt boxes and 1,624 nut boxes, so training must not start yet: the imbalance ratio is `3.86`. It contains 955 negative images, no malformed labels, and no cross-split exact duplicates. The next audit must be run after ARG_FIXINGS2 conversion and combination.

## Local installation

Python 3.11 is recommended. Docker is the shortest evaluation path; use a
local environment when running tests, dataset tools, or training scripts.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For a CUDA-enabled RTX 2060 Super, install the PyTorch build appropriate for the installed NVIDIA driver before installing or running Ultralytics. Follow the official PyTorch selector for the exact command, because the CUDA wheel must match the local environment.

## Fine-tuning

The service fine-tunes **pretrained YOLOv8s** (`yolov8s.pt`). Batch size 4 and two loader workers fit the RTX 2060 Super while limiting host RAM usage.

To continue from the deployed best YOLOv8s detector on the expanded Dataset 3 composition, keep `models/best.pt` unchanged and write a separate candidate run:

```powershell
.venv313\Scripts\python.exe scripts\train.py --data data\combined_dataset3.yaml --model models\best.pt --epochs 30 --patience 7 --imgsz 640 --batch 4 --workers 2 --device 0 --optimizer AdamW --lr0 0.001 --warmup-epochs 1 --cos-lr --name detector_dataset3_v9 --save-period 5
```

This is fine-tuning of the existing detector, not a YOLO11 migration and not training from generic COCO weights. The command writes only to `runs/detect/detector_dataset3_v9`; it never replaces `models/best.pt`.

Before this run, the deployed detector and classifier are preserved as `models/candidates/detector_baseline_before_dataset3.pt` and `models/candidates/classifier_baseline_before_dataset3.pt`. Their hashes are documented in `models/candidates/README.md` for the future source-disjoint A/B benchmark.

```powershell
.venv313\Scripts\python.exe scripts\train.py --data data\combined.yaml --model models\best.pt --epochs 30 --patience 5 --imgsz 640 --batch 4 --workers 2 --device 0 --name detector_hardneg_v3 --save-period 5
```

The training script enables moderate online colour augmentation by default. Validation and test images remain unchanged. Early stopping uses `--patience 5`, and `--save-period 5` retains periodic checkpoints. This command writes only under `runs/detect/`; it does not overwrite `models/best.pt`.

Keep the best checkpoint as a candidate until calibration and pipeline benchmarking pass:

```powershell
Copy-Item runs\detect\detector_hardneg_v3\weights\best.pt models\candidates\detector_hardneg_v3.pt
```

If CUDA runs out of memory, reduce batch size to 2. Calibration and mining process images in bounded chunks to avoid oversized tensors.

### Crop-classifier fine-tuning

The verifier fine-tunes lightweight pretrained **YOLOv8n-cls** weights. Its custom transform pipeline resizes the already square crop without `RandomResizedCrop`, so a long bolt is not truncated. Moderate colour, rotation, scale, flip, and erasing augmentation is applied only in memory to train images. Early stopping uses patience 5.

```powershell
.venv313\Scripts\python.exe scripts\train_classifier.py --data data\classifier --model models\classifier_best.pt --epochs 25 --patience 5 --imgsz 224 --batch 32 --workers 2 --device 0 --name classifier_multiscale_v3
```

Run this command only after regenerating `data/classifier` with the candidate detector. It writes under `runs/classify/`; preserve it as a candidate instead of replacing the deployed model:

```powershell
Copy-Item runs\classify\classifier_multiscale_v3\weights\best.pt models\candidates\classifier_multiscale_v3.pt
```

Calibrate candidate detector and classifier thresholds independently by class on validation data:

```powershell
.venv313\Scripts\python.exe scripts\calibrate_class_thresholds.py --task detector --data data\combined.yaml --weights models\candidates\detector_hardneg_v3.pt --split val --imgsz 640 --device 0 --batch 4 --min-recall 0.90 --max-negative-fpr 0.02 --output runs\calibration\candidate_detector.json
.venv313\Scripts\python.exe scripts\calibrate_class_thresholds.py --task classifier --data data\classifier --weights models\candidates\classifier_multiscale_v3.pt --split val --imgsz 224 --device 0 --batch 32 --min-recall 0.90 --max-negative-fpr 0.02 --output runs\calibration\candidate_classifier.json
```

Apply the four selected values only to the candidate benchmark command. Promotion happens after comparison, not after calibration alone.

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

Mixed public-source polygon labels have been normalized to enclosing boxes by `normalize_detection_labels.py`, so detector training receives one annotation format.

## Confidence-threshold calibration

Do not select thresholds by visual inspection of one image. After each training run, calibrate the detector and classifier on validation data. Bolt and nut are optimized separately because their score distributions and class frequencies differ:

```powershell
.venv313\Scripts\python.exe scripts\calibrate_class_thresholds.py --task detector --data data\combined.yaml --weights models\best.pt --split val --imgsz 640 --device 0 --batch 4 --output runs\calibration\detector_classes.json
.venv313\Scripts\python.exe scripts\calibrate_class_thresholds.py --task classifier --data data\classifier --weights models\classifier_best.pt --split val --imgsz 224 --device 0 --batch 32 --output runs\calibration\classifier_classes.json
```

Use the selected values through four explicit environment variables. Keep `CLASSIFIER_CROP_CONTEXTS=0.20` for the currently deployed single-scale classifier; switch to `0.20,1.00` only after promoting the newly trained multi-scale classifier:

```powershell
$env:BOLT_CONFIDENCE_THRESHOLD = "0.43"
$env:NUT_CONFIDENCE_THRESHOLD = "0.56"
$env:BOLT_CLASSIFIER_THRESHOLD = "0.39"
$env:NUT_CLASSIFIER_THRESHOLD = "0.51"
$env:CLASSIFIER_CROP_CONTEXTS = "0.20"
$env:FINAL_NMS_IOU_THRESHOLD = "0.45"
uvicorn app.main:app --reload
```

Run fixed regression cases after changing either weights or threshold:

```powershell
.venv313\Scripts\python.exe scripts\regression_check.py --weights models\best.pt --classifier-weights models\classifier_best.pt --bolt-detector-threshold 0.43 --nut-detector-threshold 0.56 --bolt-classifier-threshold 0.39 --nut-classifier-threshold 0.51 --contexts 0.2 --min-box-area-ratio 0.001 --nms-iou 0.45
```

Docker Compose reads the same variables, so calibrated values can be passed without editing source code.

## End-to-end benchmark and promotion

The baseline report at `runs/benchmarks/baseline_v1_test.json` evaluates the actual detector, classifier, filters, and NMS on all 410 test images. It records model SHA-256 hashes and exact configuration. Current baseline: macro F1 `0.9394`, bolt recall `0.8935`, nut recall `0.9634`, negative-image FPR `0.0000`, and exact-count image accuracy `0.8415`.

After training and validation calibration, benchmark the candidate with its four selected thresholds and both crop contexts:

```powershell
.venv313\Scripts\python.exe scripts\benchmark_pipeline.py --data data\combined.yaml --split test --detector models\candidates\detector_hardneg_v3.pt --classifier models\candidates\classifier_multiscale_v3.pt --bolt-detector-threshold BOLT_VALUE --nut-detector-threshold NUT_VALUE --bolt-classifier-threshold BOLT_CLS_VALUE --nut-classifier-threshold NUT_CLS_VALUE --contexts 0.2 1.0 --min-box-area-ratio 0.001 --nms-iou 0.45 --output runs\benchmarks\candidate_v3_test.json
.venv313\Scripts\python.exe scripts\compare_benchmarks.py --baseline runs\benchmarks\baseline_v1_test.json --candidate runs\benchmarks\candidate_v3_test.json --output runs\benchmarks\comparison_v3.json
```

Only when `promotion_recommended` is true should both deployed checkpoints and environment thresholds be replaced. Retain the old files or their Git commit so rollback remains immediate.

### Independent external benchmark

`data/external_benchmark` is the retired v1 benchmark. It uses the 225-image test split from Mechanical Parts Dataset 2022 (CC BY 4.0, DOI `10.5281/zenodo.7504801`). It must not be used after training on `combined_dataset3.yaml`, because dataset3 shares 167 source families with it.

```powershell
curl.exe -L --fail --output data\external_benchmark_sources\mechanical_parts_yolo.rar "https://zenodo.org/records/7504801/files/Mechanical%20Parts%20Dataset.rar?download=1"
tar -xf data\external_benchmark_sources\mechanical_parts_yolo.rar -C data\external_benchmark_sources
.venv313\Scripts\python.exe scripts\prepare_external_benchmark.py --replace
```

Current external composition: 105 positive and 120 bearing/gear negative images, 326 bolt boxes, and 249 nut boxes. No exact duplicates were found within the benchmark or against `combined.yaml`.

| Pipeline | Macro F1 | Bolt recall | Nut recall | Negative-image FPR | Exact counts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deployed weights | 0.5508 | 0.2546 | 0.6867 | 0.1250 | 0.6267 |
| `detector_hardneg_v3-2` candidate + deployed classifier | 0.4930 | 0.1963 | 0.6627 | 0.1000 | 0.6178 |

The candidate is rejected: its lower background FPR does not compensate for the `0.0579` macro-F1 loss and recall regressions in both task classes. Reports are stored in `runs/benchmarks/external_*.json`; `data/external_benchmark/review.csv` is the source-annotation review worksheet.

#### Frozen BOLTS benchmark v2

`data/external_benchmark_v2` is the current evaluation-only benchmark and must never be included in training or threshold calibration. It is built from the BOLTS v1 Roboflow export (CC BY 4.0). The converter retains one image per pre-augmentation source family, converts polygon annotations to boxes, maps `bolt` and `nut` to the API classes, and treats lockwashers, washers, screws, and other objects as background. It rejects exact and perceptual overlaps with `combined_dataset3.yaml` and raw `dataset3`, removes internal near-duplicates, and applies the manually reviewed exclusions in `data/benchmark-dataset/benchmark_exclusions.json` without using model predictions.

```powershell
.venv313\Scripts\python.exe scripts\prepare_bolts_benchmark.py --source data\benchmark-dataset --output data\external_benchmark_v2 --training-data data\combined_dataset3.yaml --perceptual-distance 4 --max-variants-per-family 1 --replace
.venv313\Scripts\python.exe scripts\audit_dataset.py --data data\external_benchmark_v2\benchmark.yaml --output runs\external_benchmark_v2_audit.json
.venv313\Scripts\python.exe scripts\benchmark_pipeline.py --data data\external_benchmark_v2\benchmark.yaml --split test --detector models\best.pt --classifier models\classifier_best.pt --bolt-detector-threshold 0.45 --nut-detector-threshold 0.45 --bolt-classifier-threshold 0.54 --nut-classifier-threshold 0.54 --contexts 0.2 --min-box-area-ratio 0.001 --nms-iou 0.45 --output runs\benchmarks\external_v2_current_best.json
```

The frozen v2 fingerprint is `AB0F0B36C40DEFF84446345BF51A25C4FD2FB881B13DEF49DCE04069F71DE25B`. Its 260 manually reviewed images contain 254 bolt boxes, 371 nut boxes, and 111 background-only scenes. The current deployed pipeline reaches macro F1 `0.5035`, bolt F1 `0.3518`, nut F1 `0.6553`, negative-image FPR `0.4505`, and exact-count accuracy `0.3923`. These values are evaluation results only and must not be used to retune thresholds.

The Dataset 3 detector with the legacy classifier reaches macro F1 `0.5617`, bolt F1 `0.4463`, nut F1 `0.6772`, negative-image FPR `0.0541`, and exact-count accuracy `0.5269` with thresholds selected exclusively on `combined_dataset3.yaml` validation data. It was explicitly promoted as the `dataset3_hybrid_v1` deployment on 2026-07-20: the accepted tradeoff is a nut-recall decrease from `0.5687` to `0.5202` in exchange for much lower background FPR and higher macro F1. Exact hashes, thresholds, and rollback paths are stored in `models/deployment.json`.

The first Dataset 3 crop-classifier candidate is rejected. Although it eliminates false positives on all 111 frozen negative scenes, it reduces bolt recall to `0.1969`, nut recall to `0.4717`, and macro F1 to `0.4806`. This is external-domain over-rejection, so the frozen benchmark must not be used to lower its thresholds. The candidate remains archived as `models/candidates/classifier_dataset3_v9.pt` for error analysis only.

The 50/50 source-balanced v10 classifier is also rejected. Its regression result is macro F1 `0.4783`, bolt recall `0.2126`, nut recall `0.4582`, and negative-image FPR `0.0360`. Because this training strategy was chosen after inspecting benchmark v2, v2 is now a development regression suite rather than a pristine final holdout; a new source-disjoint hidden benchmark is required for any final production claim.

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

The submission already contains `models/best.pt` and
`models/classifier_best.pt`. Start the complete hybrid pipeline with:

```bash
docker compose up --build
```

The Compose configuration mounts `./models` read-only into the container. The
image intentionally does not copy weights into an image layer, while a normal
repository clone still contains both approved checkpoints. The container uses
CPU-only PyTorch; this keeps evaluation portable and avoids downloading CUDA
runtime libraries that are unused by the default Compose configuration.

## Tests

```bash
pytest -q
```

## Known limitations and next steps

- The model quality depends on the coverage and annotation quality of the public training dataset. New backgrounds, fastener types, severe occlusion, blur, or very small objects may lower recall.
- A confidence threshold is a product decision: raising it reduces false positives but can miss true objects. Detections near the threshold should be retained for error analysis or routed to manual review in a production workflow.
- The crop classifier reduces open-set false positives but cannot mathematically recognise every possible unknown object. Continue collecting production false proposals and regenerate or extend the `other` class for iterative hard-negative mining.
- Production hardening could add request authentication, rate limiting, structured logging, model/version metadata, monitoring of confidence distributions, and asynchronous batch inference.
