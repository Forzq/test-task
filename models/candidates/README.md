# Candidate model checkpoints

This directory documents trained candidates evaluated during development.
Candidate `.pt` files are intentionally excluded from Git to keep the
submission compact. Only `models/best.pt` and `models/classifier_best.pt` are
required to run the API.

## classifier_hardneg_v2

- Source: `runs/classify/crop_verifier_hardneg_v2/weights/best.pt`
- Training data: balanced `bolt`, `nut`, and `other` crop dataset with prioritised hard negatives
- Best reported validation top-1 accuracy: `0.98663`
- Status: candidate; threshold calibration and regression comparison are required before deployment

The deployed classifier remained `models/classifier_best.pt` because this
candidate was not promoted.

## Dataset 3 baseline snapshots

The deployed pipeline was snapshotted immediately before fine-tuning on `data/combined_dataset3.yaml`:

- `detector_baseline_before_dataset3.pt` — SHA-256 `9A9B63893DE71DA40513CB2F354E2396DF7898625D76ED2CFFB17D2BAD43EA3A`.
- `classifier_baseline_before_dataset3.pt` — SHA-256 `96EBF3F8139E0BD420063DD1F420B333118F1A054D049F915666ED7A56993F46`.

The hashes preserve experiment provenance. Binary snapshots are excluded from
the submission; the baseline source state is recoverable from Git commit
`ffbcc10`.

## detector_dataset3_v9.pt

- Source: `runs/detect/detector_dataset3_v9/weights/best.pt`
- Parent checkpoint: `models/best.pt`
- Training data: `data/combined_dataset3.yaml`
- Training: 40 epochs, AdamW, batch 4, image size 640
- SHA-256: `91AA27AEDC03635E042777B1436E581AFEA40CB7877CAE5375FBED90F4814C8C`
- Validation-calibrated detector thresholds: bolt `0.43`, nut `0.56`
- Validation-calibrated classifier thresholds with deployed classifier: bolt `0.39`, nut `0.51`
- Frozen benchmark v2: macro F1 `0.5617`, negative-image FPR `0.0541`, exact-count accuracy `0.5269`
- Status: explicitly promoted as the detector in `dataset3_hybrid_v1`; the legacy classifier remains deployed

`models/best.pt` now contains this checkpoint. The previous detector is
recoverable from Git commit `ffbcc10`; exact active deployment settings are
recorded in `models/deployment.json`.

## classifier_dataset3_v9.pt

- Source: `runs/classify/classifier_dataset3_v9/weights/best.pt`
- Parent checkpoint: `models/classifier_best.pt`
- Training data: `data/classifier_dataset3_v9` with 24,000 balanced train crops
- SHA-256: `9EF24A8003581CB998A80BE67CC7834273FAE23BC92919F26491925E85E04264`
- Standalone validation thresholds: bolt `0.95`, nut `0.95`
- Complete-pipeline validation choice: contexts `0.2,1.0`, bolt `0.44`, nut `0.01`
- Frozen benchmark v2: macro F1 `0.4806`, negative-image FPR `0.0000`, exact-count accuracy `0.5000`
- Status: rejected; external bolt and nut recall regressed by `0.0906` and `0.0970`

The classifier is overly conservative outside its training domains. Do not deploy it or recalibrate it on the frozen benchmark. A future classifier dataset should source-balance legacy positives, Dataset 3 positives, and mined `other` crops to reduce catastrophic forgetting.

## classifier_source_balanced_v10.pt

- Source: `runs/classify/classifier_source_balanced_v10/weights/best.pt`
- Parent checkpoint: `models/classifier_best.pt`
- Training data: `data/classifier_source_balanced_v10`, balanced 50/50 by legacy and Dataset 3 source
- SHA-256: `010D5274E3DC4C02D3CCDDA774CDC648E861943B54F23595B448AA6A707E9154`
- Standalone validation thresholds: bolt `0.99`, nut `0.95`
- Complete-pipeline validation choice: contexts `0.2,1.0`, bolt `0.44`, nut `0.01`
- Benchmark v2 regression: macro F1 `0.4783`, negative-image FPR `0.0360`, exact-count accuracy `0.4885`
- Status: rejected; bolt and nut recall remain below the deployed pipeline

Source balancing improved internal validation but did not solve external-domain over-rejection. The strongest measured pipeline remains `detector_dataset3_v9.pt` with the deployed `classifier_best.pt`; promoting that hybrid requires an explicit decision to accept its `0.0485` nut-recall regression in exchange for much lower background FPR and higher macro F1.
